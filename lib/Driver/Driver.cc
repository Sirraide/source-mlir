/// ===========================================================================
///                                   DRIVER
///
/// This file implements the Source compiler driver. The driver is the main
/// entry point for the compiler and is responsible for creating the compiler
/// context, dispatching work amongst threads and managing the compilation
/// process.
///
/// Some invariants:
///
///   - All public member functions are thread-safe and must lock the
///     driver mutex upon entry.
///
///   - Public member functions MUST NOT be called by other member functions.
///
///   - No private member function may lock the driver mutex.
///
/// ===========================================================================

#include <llvm/Support/ThreadPool.h>
#include <source/CG/CodeGen.hh>
#include <source/Core.hh>
#include <source/Driver/Driver.hh>
#include <source/Frontend/Parser.hh>
#include <source/Frontend/Sema.hh>
#include <source/Support/DisjointSets.hh>
#include <source/Support/TopologicalSort.hh>

/// Implemented in HLIR.cc
namespace mlir::hlir {
void InitContext(MLIRContext& mctx);
}

namespace src {
struct FileSource {
    virtual ~FileSource() = default;

    /// Get the next file to compile.
    virtual auto next(Context* ctx) -> File* = 0;

    /// Get the number of files to compile.
    virtual usz size() = 0;
};

class DriverImpl : public Driver {
public:
    DriverImpl(u32 thread_count) : threads{llvm::ThreadPoolStrategy{thread_count}} {}

    /// Thread pool for this driver.
    llvm::ThreadPool threads;

    /// To make this thread-safe.
    std::mutex mtx;

    /// Compile options.
    CompileOptions opts;

    /// Import paths.
    std::vector<fs::path> import_paths;

    /// Implements add_import_path().
    void AddImportPath(fs::path path);

    /// Implements the overloads of compile().
    auto Compile(FileSource& source) -> int;

    /// Implements describe_module().
    bool DescribeModule(StringRef name);

    /// Create a new context.
    void InitContext(Context& ctx);
};
} // namespace src

/// ===========================================================================
///  Internals
/// ===========================================================================
void src::DriverImpl::AddImportPath(fs::path path) {
    if (path.is_relative()) path = fs::current_path() / path;
    import_paths.push_back(std::move(path));
}

auto src::DriverImpl::Compile(FileSource& source) -> int {
    using Action = CompileOptions::Action;
    enum : int {
        Ok = 0,
        Error = -1,
    };

    /// Nothing to do.
    if (source.size() == 0) return Ok;

    /// Create context for this compilation.
    Context ctx;
    InitContext(ctx);

    /// Parse input files in parallel.
    SmallVector<std::shared_future<Module*>> parsed_modules;
    for (usz i = 0; i < source.size(); i++) {
        parsed_modules.push_back(threads.async([&] -> Module* {
            auto file = source.next(&ctx);
            auto mod = Parser::Parse(ctx, *file);
            if (ctx.has_error()) return nullptr;
            return mod;
        }));
    }

    threads.wait();
    if (ctx.has_error()) return opts.action == Action::Sema ? Ok : Error;

    /// Sort the modules on their module name; use a stable sort
    /// for deterministic ordering based on the order of the input
    /// files on the command line.
    rgs::stable_sort(parsed_modules, [](auto& a, auto& b) {
        return a.get()->name < b.get()->name;
    });

    /// Combine modules with the same name and also collect module names.
    StringMap<usz> name_indices;
    SmallVector<Module*> combined_modules;
    for (const auto& parsed_module : parsed_modules) {
        auto mod = parsed_module.get();
        Assert(mod, "Parser should never return null unless the error flag is set");

        /// If this is the first or a different module, just add it.
        if (
            combined_modules.empty() or
            combined_modules.back()->name != mod->name
        ) {
            if (mod->is_logical_module) name_indices[mod->name] = combined_modules.size();
            combined_modules.push_back(mod);
            continue;
        }

        /// Otherwise, merge them.
        combined_modules.back()->assimilate(mod);
    }

    /// Print the AST of the modules, if requested.
    if (opts.syntax_only) {
        if (opts.action == Action::PrintAST)
            for (auto mod : combined_modules)
                mod->print_ast();
        return Ok;
    }

    /// Import runtime if requested.
    if (opts.include_runtime) {
        for (auto mod : combined_modules) mod->add_import(
            __SRCC_RUNTIME_NAME,
            __SRCC_RUNTIME_NAME,
            {},
            false,
            false
        );
    }

    /// Perform a topological sort on the modules to determine what order
    /// we need to build them in and to detect cyclic dependencies, while
    /// also maximising parallelism.
    ///
    /// Build a dependency graph (represented here as an adjacency matrix);
    /// each row R represents some module; the column entries C in that column
    /// are all modules such that R must be built before C, that is, such that
    /// C imports R.
    ///
    /// This only considers modules that are also built by this compilation
    /// process.
    SmallVector<SmallVector<usz>> task_groups;
    {
        auto mcount = combined_modules.size();
        auto storage = Buffer<bool>(mcount * mcount, false);
        utils::TSortInputGraph dep_graph(storage, mcount);
        for (auto [j, m] : llvm::enumerate(combined_modules)) {
            for (auto& i : m->imports) {
                auto idx = name_indices.find(i.linkage_name);
                if (idx == name_indices.end()) continue;
                dep_graph[idx->second, j] = 1;
            }
        }

        if (not utils::TopologicalSort(dep_graph, task_groups)) {
            SmallVector<std::string_view> cycle;

            /// If the topological sort fails, we have a cycle; find it
            /// using Kruskal’s algorithm.
            int nodes = int(dep_graph.nodes);
            utils::DisjointSets<int> ds{nodes};
            for (int i = 0; i < nodes; i++) {
                for (int j = 0; j < nodes; j++) {
                    if (dep_graph[usz(i), usz(j)] and not ds.unite(i, j)) {
                        for (int elem : ds.elements(i))
                            cycle.push_back(combined_modules[usz(elem)]->name);
                        break;
                    }
                }
            }

            Diag::Error("Cyclic dependency between modules: {}", fmt::join(cycle, ", "));
            return opts.action == Action::Sema ? Ok : Error;
        }
    }

    /// Perform an action on every module and wait until all threads are done.
    auto ForEachModule = [&](auto task) {
        for (auto mod : combined_modules) threads.async([=] { task(mod); });
        threads.wait();
    };

    /// Compile modules in parallel.
    for (auto& g : task_groups) {
        auto Analyse = [&](usz module_index) {
            auto mod = combined_modules[module_index];

            /// Resolve any imports that we’ve already built.
            for (auto& i : mod->imports) {
                auto idx = name_indices.find(i.linkage_name);
                if (idx == name_indices.end()) continue;
                i.mod = combined_modules[usz(idx->second)];
            }

            Sema::Analyse(mod, opts.debug_cxx);
        };

        for (auto mod : g) threads.async([=] { return Analyse(mod); });
        threads.wait();
        if (ctx.has_error()) return opts.action == Action::Sema ? Ok : Error;
    }

    if (opts.action == Action::PrintAST or opts.action == Action::Sema) {
        if (opts.action == Action::PrintAST)
            for (auto m : combined_modules)
                m->print_ast();
        return Ok;
    }

    /// Print exports if requested.
    if (opts.action == Action::PrintExports) {
        for (auto mod : combined_modules) {
            if (mod->is_logical_module) {
                fmt::print("Module '{}':\n", mod->name);
                for (auto& exps : mod->exports)
                    for (auto e : exps.second)
                        e->print(false);
            }
        }
        return Ok;
    }

    /// Generate HLIR for each module. If this fails, that’s an ICE,
    /// so no need for error checking here.
    mlir::MLIRContext mlir{mlir::MLIRContext::Threading::DISABLED};
    mlir.setThreadPool(threads);
    mlir::hlir::InitContext(mlir);
    ForEachModule([&](auto mod) { CodeGenModule(&mlir, mod, not opts.verify_hlir); });
    if (opts.action == Action::PrintHLIR) {
        for (auto mod : combined_modules)
            mod->print_hlir(opts.use_generic_assembly_format);
        return Ok;
    }

    /// Lower HLIR to lowered HLIR.
    ForEachModule([&](auto mod) { LowerHLIR(&mlir, mod); });
    if (opts.action == Action::PrintLoweredHLIR) {
        for (auto mod : combined_modules)
            mod->print_hlir(opts.use_generic_assembly_format);
        return Ok;
    }

    /// Lower HLIR to LLVM IR.
    const bool debug_lowering = opts.action == Action::PrintLLVMLowering;
    ForEachModule([&](auto mod) { LowerToLLVM(&mlir, mod, debug_lowering, not opts.verify_hlir); });
    if (debug_lowering or opts.action == Action::PrintLLVM) {
        for (auto mod : combined_modules)
            mod->print_llvm(int(opts.opt_level));
        return Ok;
    }

    /// Run the code if requested.
    if (opts.action == Action::Execute) {
        /// Because the modules are sorted by name, and the empty string
        /// is the first string in lexicographical order, the executable
        /// must be the first module, if any.
        if (not combined_modules.front()->is_logical_module) {
            Diag::Error("No executable found");
            return Error;
        }

        return i8(combined_modules.front()->run(int(opts.opt_level)));
    }

    /// Emit modules to disk.
    ForEachModule([&](Module* mod) {
        if (mod->is_logical_module) {
            Assert(not mod->name.empty());
            auto name = opts.module_output_dir / mod->name;
            name += __SRCC_OBJ_FILE_EXT;
            mod->emit_object_file(int(opts.opt_level), name);
        }

        /// Emit executables in the cwd.
        else {
            mod->emit_executable(int(opts.opt_level), opts.executable_output_name);
        }
    });

    return Ok;
}

bool src::DriverImpl::DescribeModule(StringRef name) {
    Context ctx;
    InitContext(ctx);
    Module* mod{};

    /// C++ header.
    if (name.starts_with("<") and name.ends_with(">")) {
        name = name.substr(1, name.size() - 2);
        mod = Module::ImportCXXHeaders(
            &ctx,
            {name},
            std::string{name},
            opts.debug_cxx,
            {}
        );
    }

    else {
        fs::path path;

        /// Runtime.
        if (name == __SRCC_RUNTIME_NAME and opts.include_runtime) {
            path = __SRCC_BUILTIN_MODULE_PATH;
            path /= __SRCC_RUNTIME_NAME;
            path += __SRCC_OBJ_FILE_EXT;
        } else {
            path = std::string_view{name};
        }

        auto& f = ctx.get_or_load_file(std::move(path));
        mod = Module::Deserialise(
            &ctx,
            auto{f.path()}.filename().replace_extension(""),
            {},
            ArrayRef(
                reinterpret_cast<const u8*>(f.data()),
                f.size()
            )
        );
    }

    /// Check for errors.
    if (ctx.has_error()) return false;
    mod->print_exports();
    return true;
}

void src::DriverImpl::InitContext(Context& ctx) {
    ctx.should_use_colours.store(opts.use_colours, std::memory_order_relaxed);
    ctx.module_import_paths = import_paths;
}

auto src::Driver::Impl() {
    return static_cast<DriverImpl*>(this);
}

/// ===========================================================================
///  API
/// ===========================================================================
auto src::Driver::Create(CompileOptions opts) -> std::unique_ptr<Driver> {
    auto driver = std::make_unique<DriverImpl>(opts.threads ?: std::thread::hardware_concurrency());
    driver->opts = std::move(opts);
    if (opts.use_default_import_paths) driver->AddImportPath(__SRCC_BUILTIN_MODULE_PATH);
    return std::unique_ptr<Driver>{driver.release()};
}

void std::default_delete<src::Driver>::operator()(src::Driver* ptr) const noexcept {
    delete ptr->Impl();
}

void src::Driver::add_import_path(fs::path path) {
    std::unique_lock _{Impl()->mtx};
    Impl()->AddImportPath(std::move(path));
}

int src::Driver::compile(std::vector<File*> files) {
    struct Files : FileSource {
        std::vector<File*> files;
        std::vector<File*>::iterator it{files.begin()};

        Files(std::vector<File*> files) : files{std::move(files)} {}

        auto next(Context*) -> File* override {
            Assert(it != files.end());
            return *it++;
        }

        usz size() override { return files.size(); }
    };

    std::unique_lock _{Impl()->mtx};
    Files f{std::move(files)};
    return Impl()->Compile(f);
}

int src::Driver::compile(std::vector<fs::path> file_paths) {
    struct Files : FileSource {
        std::vector<fs::path> files;
        std::vector<fs::path>::iterator it{files.begin()};

        Files(std::vector<fs::path> files) : files{std::move(files)} {}

        auto next(Context* context) -> File* override {
            Assert(it != files.end());
            return &context->get_or_load_file(*it++);
        }

        usz size() override { return files.size(); }
    };

    std::unique_lock _{Impl()->mtx};
    Files f{std::move(file_paths)};
    return Impl()->Compile(f);
}

bool src::Driver::describe_module(llvm::StringRef name) {
    std::unique_lock _{Impl()->mtx};
    return Impl()->DescribeModule(name);
}
