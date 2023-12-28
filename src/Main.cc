#include <clopts.hh>
#include <csignal>
#include <source/Driver/Driver.hh>

namespace detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    multiple<positional<"file", "The file to compile", std::string, true>>,
    option<"--colour", "Enable coloured output (default: auto)", values<"always", "auto", "never">>,
    option<"--dir", "Set module output directory">,
    option<"-o", "Output file name. Ignored for modules">,
    option<"--target-features", "Set a feature flag for the backend (e.g. +avx2)">,
    multiple<option<"-I", "Add module import directory">>,
    experimental::short_option<"-O", "Optimisation level", values<0, 1, 2, 3, 4>>,
    experimental::short_option<"-j", "Number of threads to use for compilation", std::int64_t>,
    flag<"-r", "JIT-compile and run the program after compiling">,
    flag<"-s", "Emit assembly code">,
    flag<"--ast", "Print the AST of the module after parsing">,
    flag<"--lowered", "Print lowered HLIR">,
    flag<"--debug-llvm", "Debug LLVM lowering process">,
    flag<"--debug-cxx", "Debug C++ imports">,
    flag<"--describe-module", "Load file as a module and print its exports">,
    flag<"--exports", "Show exported declarations">,
    flag<"--hlir", "Print the HLIR of the module">,
    flag<"--llvm", "Print the LLVM IR of the module">,
    flag<"--no-verify", "Disable MLIR verification; CAUTION: this may lead to miscompilations">,
    flag<"--nostdinc", "Do not add standard module directories to the module search path">,
    flag<"--nostdrt", "Do not import the standard runtime module">,
    flag<"--sema", "Run sema only and always exit with code 0 unless there is an ICE">,
    flag<"--syntax-only", "Skip the semantic analysis step">,
    flag<"--use-generic-assembly-format", "Print HLIR using the generic assembly format">,
    help<>
>; // clang-format on
}
using detail::options;

/// TODO: Optional `program <name>` directive in the first line of a file
/// to allow specifying the name of the executable (as well as building
/// multiple executables at the same time)

int main(int argc, char** argv) {
    /// Parse options.
    auto opts = options::parse(argc, argv, [](auto&& s) -> bool { src::Diag::Fatal("{}", s); });

    /// Check if we want to use colours.
    bool use_colour = isatty(fileno(stdout));
    if (auto c = opts.get<"--colour">()) {
        if (*c == "always") use_colour = true;
        else if (*c == "never") use_colour = false;
    }

    /// Enable them globally.
    src::EnableAssertColours(use_colour);

    /// Disallow filenames starting with '-'; users can still write `./-`.
    for (auto& f : *opts.get<"file">()) {
        if (f.starts_with('-')) src::Diag::Fatal(
            "Invalid option: '{}'. Write './{}' to treat it as a filename.",
            f,
            f
        );
    }

    /// -o - is only valid for assembly.
    if (opts.get_or<"-o">("") == "-" and not opts.get<"-s">())
        src::Diag::Fatal("'-o -' is only valid in combination with '-s'");

    /// Parse target features.
    llvm::StringMap<bool> target_features;
    if (auto features = opts.get<"--target-features">()) {
        for (auto feature : src::rgs::subrange(*features) | src::vws::split(',')) {
            if (
                feature.size() < 2 or
                (feature.front() != '+' and feature.front() != '-')
            ) src::Diag::Fatal(
                "Argument to '--target-features' must be a comma-separated list of "
                "strings, each starting with either '+' or '-'"
            );

            auto f = feature | src::vws::drop(1);
            target_features[{f.data(), f.size()}] = feature.front() == '+';
        }
    }

    /// Create driver.
    using Action = src::CompileOptions::Action;
    auto driver = src::Driver::Create({
        .module_output_dir = opts.get_or<"--dir">(src::fs::current_path()),
        .executable_output_name = opts.get_or<"-o">("a.out"),
        .target_features = std::move(target_features),
        .action = opts.get<"--ast">()        ? Action::PrintAST
                : opts.get<"--debug-llvm">() ? Action::PrintLLVMLowering
                : opts.get<"--exports">()    ? Action::PrintExports
                : opts.get<"--hlir">()       ? Action::PrintHLIR
                : opts.get<"--llvm">()       ? Action::PrintLLVM
                : opts.get<"--lowered">()    ? Action::PrintLoweredHLIR
                : opts.get<"-r">()           ? Action::Execute
                : opts.get<"-s">()           ? Action::EmitASM
                : opts.get<"--sema">()       ? Action::Sema
                                             : Action::Compile,

        .opt_level = src::u8(opts.get_or<"-O">(0)),
        .threads = src::u16(opts.get_or<"-j">(0)),
        .debug_cxx = opts.get<"--debug-cxx">(),
        .include_runtime = not opts.get<"--nostdrt">(),
        .syntax_only = opts.get<"--syntax-only">(),
        .use_default_import_paths = not opts.get<"--nostdinc">(),
        .use_colours = use_colour,
        .use_generic_assembly_format = opts.get<"--use-generic-assembly-format">(),
        .verify_hlir = not opts.get<"--no-verify">(),
    });

    /// Add import paths.
    for (auto& path : *opts.get<"-I">())
        driver->add_import_path(path);

    /// Describe module, if requested.
    if (opts.get<"--describe-module">()) {
        const bool ok = driver->describe_module(opts.get<"file">()->front());
        std::exit(ok ? 0 : 1);
    }

    /// Collect files.
    std::vector<std::filesystem::path> files;
    for (auto& f : *opts.get<"file">()) files.emplace_back(f);

    /// Dew it.
    return driver->compile(std::move(files));

    /*    /// Notes:
        /// - ‘freeze’ keyword that makes a value const rather than forcing
        ///   it to be const in the declaration?
        ///
        /// - When parsing a declaration, the declaration itself is attached
        ///   to the nearest enclosing ‘declaration context’; the occurrence
        ///   of the declaration is replaced with an already resolved NameRef
        ///   to that declaration.

        /// Create a function that returns void and takes no arguments.
        auto mod = mlir::ModuleOp::create(builder.getUnknownLoc(), "bla");
        builder.setInsertionPointToEnd(mod.getBody());
        auto funcType = builder.getFunctionType({}, builder.getI32Type());
        auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);

        /// Add a block to the function.
        auto& entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        /// Create a string constant and print it.
        auto s = builder.create<mlir::hlir::StringOp>(builder.getUnknownLoc(), "Hello, World!");
        builder.create<mlir::hlir::PrintOp>(builder.getUnknownLoc(), s);

        /// Return 0.
        auto i = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 42, 32);
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{i});

        fmt::print("=== Module before lowering ===\n");
        if (not mlir::succeeded(mod.verify())) return 1;
        mod->print(llvm::outs());

        /// Lower the module.
        pm.addPass(mlir::hlir::CreateLowerToLLVMPass());
        if (mlir::failed(pm.run(mod))) return 1;

        fmt::print("\n=== Module after lowering ===\n");
        mod->print(llvm::outs());

        /// Convert to LLVM IR.
        mlir::registerLLVMDialectTranslation(*mod->getContext());
        llvm::LLVMContext llvm_ctx;
        auto llvm_mod = mlir::translateModuleToLLVMIR(mod, llvm_ctx);

        fmt::print("\n=== Module after conversion to LLVM IR ===\n");
        llvm_mod->dump();*/
}
