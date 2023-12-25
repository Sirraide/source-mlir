#ifndef SOURCE_INCLUDE_CONTEXT_HH
#define SOURCE_INCLUDE_CONTEXT_HH

#include <clang/Frontend/CompilerInstance.h>
#include <llvm/IR/Module.h>
#include <mlir/IR/MLIRContext.h>
#include <source/Frontend/Token.hh>
#include <source/Support/StringTable.hh>
#include <source/Support/Utils.hh>

namespace llvm {
class Target;
}

namespace mlir {
class ModuleOp;
class Operation;
class Value;
class Block;
class Type;
}

namespace src {
class DriverImpl;
class Context;
class Module;
class Decl;
class ProcDecl;
class Expr;
class BlockExpr;
class StructType;
class IntType;

/// Enable colours in Assert()/Todo()/Unreachable().
void EnableAssertColours(bool enable);

/// A file in the context.
class File {
    /// Context handle.
    Context& ctx;

    /// The name of the file.
    fs::path file_path;

    /// The contents of the file.
    std::unique_ptr<llvm::MemoryBuffer> contents;

    /// The id of the file.
    u32 id;

public:
    /// Get a temporary file path.
    static auto TempPath(std::string_view extension) -> fs::path;

    /// Write to a file on disk.
    [[nodiscard]] static bool Write(void* data, usz size, const fs::path& file);

    /// Write to a file on disk and terminate on error.
    static void WriteOrDie(void* data, usz size, const fs::path& file);

    /// We cannot move or copy files.
    File(const File&) = delete;
    File(File&&) = delete;
    File& operator=(const File&) = delete;
    File& operator=(File&&) = delete;

    /// Get an iterator to the beginning of the file.
    [[nodiscard]] auto begin() const { return contents->getBufferStart(); }

    /// Get the file data.
    [[nodiscard]] auto data() const -> const char* { return contents->getBufferStart(); }

    /// Get an iterator to the end of the file.
    [[nodiscard]] auto end() const { return contents->getBufferEnd(); }

    /// Get the id of this file.
    [[nodiscard]] auto file_id() const { return id; }

    /// Get the file path.
    [[nodiscard]] auto path() const -> const fs::path& { return file_path; }

    /// Get the size of the file.
    [[nodiscard]] auto size() const -> usz { return contents->getBufferSize(); }

private:
    /// Construct a file from a name and source.
    explicit File(
        Context& _ctx,
        fs::path _name,
        std::unique_ptr<llvm::MemoryBuffer> _contents
    );

    /// Load a file from disk.
    static auto LoadFileData(const fs::path& path) -> std::unique_ptr<llvm::MemoryBuffer>;

    /// The context is the only thing that can create files.
    friend Context;
};

/// Context.
///
/// This stores anything that is shared between modules. Operations
/// on the context are thread-safe.
class Context {
    friend DriverImpl;

    /// The files owned by the context.
    std::vector<std::unique_ptr<File>> owned_files;

    /// Import paths.
    std::span<const fs::path> module_import_paths;

    /// Modules in the context. Sometimes, we have to create modules
    /// during the compilation process, so we have to store them all
    /// in the context so we always have a place where we can put them.
    std::vector<std::unique_ptr<Module>> modules;

    /// For thread safety.
    mutable std::recursive_mutex mtx;

    /// Whether to use colours in diagnostics.
    std::atomic<bool> should_use_colours = true;

    /// Error flag. This is set-only.
    mutable std::atomic_flag error_flag;

    /// Some built-in types are stored here.
    std::array<std::unique_ptr<IntType>, 6> type_storage;

    Align int_align;
    Align pointer_align;
    Size int_size;
    Size pointer_size;
public:
    /// Whether to use colours in diagnostics.
    readonly_const(bool, use_colours, return should_use_colours.load(std::memory_order_relaxed));

    /// Paths to search for modules.
    readonly_const(std::span<const fs::path>, import_paths, return module_import_paths);

    readonly_const(Align, align_of_int, return int_align);
    readonly_const(Align, align_of_pointer, return pointer_align);
    readonly_const(Size, size_of_int, return int_size);
    readonly_const(Size, size_of_pointer, return pointer_size);

    /// FFI types.
    IntType* ffi_char;
    IntType* ffi_short;
    IntType* ffi_int;
    IntType* ffi_long;
    IntType* ffi_long_long;
    IntType* ffi_size_t;

    /// So we don’t interleave diagnostics.
    mutable std::mutex diags_mutex;

    /// Create a context for the host target.
    explicit Context();

    /// Do not allow copying or moving the context.
    Context(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;

    /// Delete context data.
    ~Context();

    /// Add a module to the context.
    void add_module(std::unique_ptr<Module> mod);

    /// Create a new file from a name and contents.
    File& create_file(fs::path name, std::unique_ptr<llvm::MemoryBuffer> contents) {
        std::unique_lock _{mtx};
        return MakeFile(
            std::move(name),
            std::move(contents)
        );
    }

    /// Get a file by id.
    [[nodiscard]] auto file(usz id) const -> File* {
        std::unique_lock _{mtx};
        if (id >= owned_files.size()) return nullptr;
        return owned_files[id].get();
    }

    /// Get the number of files in the context.
    [[nodiscard]] auto file_count() const -> usz {
        std::unique_lock _{mtx};
        return owned_files.size();
    }

    /// Get a file from disk.
    ///
    /// This loads a file from disk or returns a reference to it if
    /// is has already been loaded.
    ///
    /// \param path The path to the file.
    /// \return A reference to the file.
    File& get_or_load_file(fs::path path);

    /// Check if the error flag is set.
    [[nodiscard]] bool has_error() const {
        return error_flag.test(std::memory_order_acquire);
    }

    /// Create a clang compiler instance.
    ///
    /// This may be called concurrently from multiple threads.
    void init_clang(clang::CompilerInstance& ci);

    /// Set the error flag.
    ///
    /// \return The previous value of the error flag.
    bool set_error() const {
        return error_flag.test_and_set(std::memory_order_release);
    }

private:
    /// Initialise the context.
    void Initialise();

    /// Register a file in the context.
    File& MakeFile(fs::path name, std::unique_ptr<llvm::MemoryBuffer> contents);
};

/// Reference to an imported module.
struct ImportedModuleRef {
    /// The actual name of the module for linkage purposes.
    String linkage_name;

    /// The name of the module in code.
    String logical_name;

    /// Location of the import declaration.
    Location import_location;

    /// Whether this is an open module.
    bool is_open;

    /// Whether this is actually a C++ header.
    bool is_cxx_header;

    /// Resolved module path.
    fs::path resolved_path{};

    /// The module.
    Module* mod{};

    friend bool operator==(const ImportedModuleRef& a, const ImportedModuleRef& b) {
        return a.logical_name == b.logical_name and
               a.linkage_name == b.linkage_name and
               a.is_open == b.is_open;
    }
};

/// A Source module.
///
/// Unlike the context, a module is NOT thread-safe; it may only be
/// accessed from one thread at a time.
class Module {
    llvm::LLVMContext llvm_context;

public:
    Context* const context;

    /// Allocator for AST nodes, strings, etc.
    std::unique_ptr<llvm::BumpPtrAllocator> alloc = std::make_unique<llvm::BumpPtrAllocator>();

    /// Tokens from which this module was parsed.
    std::unique_ptr<TokenStream> tokens = std::make_unique<TokenStream>(*alloc);

    /// Owner objects that were added to this after merging a module that
    /// need to be deleted when this module is deleted.
    SmallVector<utils::OpaqueHandle> owned_objects;

    /// Modules imported by this module.
    SmallVector<ImportedModuleRef> imports;

    /// Exported declarations.
    StringMap<llvm::TinyPtrVector<Expr*>> exports;

    /// Named structs.
    SmallVector<StructType*, 64> named_structs;

    /// Top-level module function.
    ProcDecl* top_level_func{};

    /// Module string table for string literals.
    StringTable strtab;

    /// AST nodes in this module.
    SmallVector<Expr*> exprs;

    /// Functions that are part of this module.
    SmallVector<ProcDecl*> functions;

    /// Static assertions that are not part of a template go here.
    SmallVector<Expr*, 32> static_assertions;

    /// Module name.
    String name;

    /// Location of the module declaration.
    Location module_decl_location;

    /// Whether this is a logical module.
    readonly(bool, is_logical_module, return not name.empty());

    /// Whether this is a C++ header.
    bool is_cxx_header{};

    /// Get the global scope of this module.
    readonly_decl(BlockExpr*, global_scope);

    /// Accessor so we don’t have to include everything required
    /// to bring mlir::ModuleOp into scope here. Implemented in
    /// CodeGen.cc.
    readonly_decl(mlir::ModuleOp, mlir);

    /// Associated MLIR module op.
    mlir::Operation* mlir_module_op;

    /// Associated LLVM module. This is null for
    /// imported modules.
    std::unique_ptr<llvm::Module> llvm;

private:
    explicit Module(Context* ctx);

public:
    Module(const Module&) = delete;
    Module(Module&&) = delete;
    Module& operator=(const Module&) = delete;
    Module& operator=(Module&&) = delete;
    ~Module();

    /// Create a new module.
    static auto Create(
        Context* ctx,
        StringRef name,
        bool is_cxx_header = false,
        Location module_decl_location = {}
    ) -> Module*;

    /// Create a new, uninitialised module. The module must be initialised
    /// with a call to init() and cannot be used for anything else until that
    /// has happened. The exception to this is that any container members
    /// (e.g. `alloc`) may be accessed at any time.
    static auto CreateUninitialised(Context* ctx) -> Module*;

    /// Add a function to this module.
    void add_function(ProcDecl* func) { functions.push_back(func); }

    /// Add an import to this module.
    bool add_import(
        StringRef linkage_name,
        StringRef logical_name,
        Location import_location,
        bool is_open = false,
        bool is_cxx_header = false
    );

    /// Merge another module into this one.
    ///
    /// The other module will be left empty and should not be used
    /// anymore. The only operation that is supported on an assimilated
    /// module is calling its destructor.
    void assimilate(Module* other);

    /// Get the name to use for the module description section in the object file.
    [[nodiscard]] auto description_section_name() const -> std::string {
        return fmt::format(".__src_module__description__.{}", name);
    }

    /// Emit an executable. Implemented in Emit.cc.
    void emit_executable(int opt_level, const fs::path& location);

    /// Emit code to an object file. Implemented in Emit.cc
    void emit_object_file(int opt_level, const fs::path& location);

    /// Initialise an uninitialised module. It is illegal to call this more
    /// than once or on a module not created with CreateUninitialised().
    void init(StringRef name, bool is_cxx_header = false, Location module_decl_location = {});

    /// Get the name to use for the guard variable for this module’s initialiser.
    [[nodiscard]] auto init_guard_name() const -> std::string {
        return fmt::format("__src_init_guard.{}", name);
    }

    /// Get the name to use for the module initialiser of this module.
    [[nodiscard]] auto module_initialiser_name() const -> std::string {
        return fmt::format("__src_static_init.{}", name);
    }

    /// Print the AST of the module to stdout. Implemented
    /// in AST.cc
    void print_ast() const;

    /// Print the AST of any exported symbols.
    void print_exports() const;

    /// Print the HLIR of the module. Implemented in CodeGen.cc.
    void print_hlir(bool use_generic_assembly_format) const;

    /// Print the module as LLVM IR. Implemented in HLIRLowering.cc.
    void print_llvm(int opt_level);

    /// Execute the module. Implemented in HLIRLowering.cc.
    int run(int opt_level);

    /// Save a string. This is not to be used for string literals.
    auto save(StringRef str) -> String { return tokens->save(str); }

    /// Serialise the module to a description that can be saved and
    /// loaded later. Implemented in Endec.cc.
    auto serialise() -> SmallVector<u8>;

    /// Deserialise a module from a module description. Implemented in Endec.cc.
    static auto Deserialise(
        Context* ctx,
        StringRef module_name,
        Location loc,
        ArrayRef<u8> description
    ) -> Module*;

    /// Import a C++ header.
    static auto ImportCXXHeaders(
        Context* ctx,
        ArrayRef<StringRef> header_names,
        StringRef module_name,
        bool debug_cxx,
        Location loc
    ) -> Module*;
private:
    /// Generate LLVM IR for this module.
    void GenerateLLVMIR(int opt_level);
};

/// A diagnostic. The diagnostic is issued when the destructor is called.
struct Diag {
    /// Diagnostic severity.
    enum struct Kind : u8 {
        None,    ///< Not an error. Do not emit this diagnostic.
        Note,    ///< Informational note.
        Warning, ///< Warning, but no hard error.
        Error,   ///< Hard error. Program is ill-formed.
        FError,  ///< Fatal (system) error. NOT a compiler bug.
        ICError, ///< Compiler bug.
    };

private:
    const Context* ctx;
    Kind kind;
    bool include_stack_trace = true;
    Location where;
    std::source_location sloc{};
    std::string msg;

    /// Handle fatal error codes.
    void HandleFatalErrors(utils::Colours);

    /// Print a diagnostic with no (valid) location info.
    void PrintDiagWithoutLocation(utils::Colours);

    /// Do not print a stack trace.
    void NoTrace() { include_stack_trace = false; }

public:
    static constexpr u8 ICEExitCode = 17;
    static constexpr u8 FatalExitCode = 18;

    Diag(Diag&& other)
        : ctx(other.ctx), kind(other.kind), where(other.where), msg(std::move(other.msg)) {
        other.kind = Kind::None;
    }

    Diag& operator=(Diag&& other) {
        if (this == &other) return *this;
        ctx = other.ctx;
        kind = other.kind;
        where = other.where;
        msg = std::move(other.msg);
        other.kind = Kind::None;
        return *this;
    }

    /// Create an empty diagnostic.
    explicit Diag()
        : ctx(nullptr), kind(Kind::None), where(), msg() {}

    /// Disallow copying.
    Diag(const Diag&) = delete;
    Diag& operator=(const Diag&) = delete;

    /// The destructor prints the diagnostic, if it hasn’t been moved from.
    ~Diag() {
        if (kind != Kind::None) print();
    }

    /// Issue a diagnostic.
    Diag(const Context* ctx, Kind kind, Location where, std::string msg)
        : ctx(ctx), kind(kind), where(where), msg(std::move(msg)) {}

    /// Issue a diagnostic with no location.
    Diag(Kind _kind, std::string&& msg, std::source_location sloc = {})
        : ctx(nullptr), kind(_kind), where(), sloc(sloc), msg(std::move(msg)) {}

    /// Issue a diagnostic with a format string and arguments.
    template <typename... Args>
    Diag(
        const Context* ctx,
        Kind kind,
        Location where,
        fmt::format_string<Args...> fmt,
        Args&&... args
    ) : Diag{ctx, kind, where, fmt::format(fmt, std::forward<Args>(args)...)} {}

    /// Issue a diagnostic with a format string and arguments, but no location.
    template <typename... Args>
    Diag(Kind kind, fmt::format_string<Args...> fmt, Args&&... args)
        : Diag{kind, fmt::format(fmt, std::forward<Args>(args)...)} {}

    /// Print this diagnostic now. This resets the diagnostic.
    void print();

    /// Suppress this diagnostic.
    void suppress() { kind = Kind::None; }

    /// Emit a note.
    template <typename... Args>
    static Diag Note(fmt::format_string<Args...> fmt, Args&&... args) {
        return Diag{Kind::Note, fmt::format(fmt, std::forward<Args>(args)...)};
    }

    /// Emit a note.
    template <typename... Args>
    static Diag Note(
        const Context* ctx,
        Location where,
        fmt::format_string<Args...> fmt,
        Args&&... args
    ) {
        return Diag{ctx, Kind::Note, where, fmt::format(fmt, std::forward<Args>(args)...)};
    }

    /// Emit a warning.
    template <typename... Args>
    static Diag Warning(fmt::format_string<Args...> fmt, Args&&... args) {
        return Diag{Kind::Warning, fmt::format(fmt, std::forward<Args>(args)...)};
    }

    /// Emit a warning.
    template <typename... Args>
    static Diag Warning(
        const Context* ctx,
        Location where,
        fmt::format_string<Args...> fmt,
        Args&&... args
    ) {
        return Diag{ctx, Kind::Warning, where, fmt::format(fmt, std::forward<Args>(args)...)};
    }

    /// Emit an error.
    template <typename... Args>
    static Diag Error(
        fmt::format_string<Args...> fmt,
        Args&&... args
    ) {
        return Diag{Kind::Error, fmt::format(fmt, std::forward<Args>(args)...)};
    }

    /// Emit an error.
    template <typename... Args>
    static Diag Error(
        const Context* ctx,
        Location where,
        fmt::format_string<Args...> fmt,
        Args&&... args
    ) {
        return Diag{ctx, Kind::Error, where, fmt::format(fmt, std::forward<Args>(args)...)};
    }

    /// Raise an internal compiler error and exit.
    template <typename... Args>
    [[noreturn]] static void ICE(
        utils::FStringWithSrcLoc<Args...> fmt,
        Args&&... args
    ) {
        Diag{Kind::ICError, fmt::format(fmt.fmt, std::forward<Args>(args)...), fmt.sloc};
        std::terminate(); /// Should never be reached.
    }

    /// Raise an internal compiler error at a location and exit.
    template <typename... Args>
    [[noreturn]] static void ICE(
        const Context* ctx,
        Location where,
        fmt::format_string<Args...> fmt,
        Args&&... args
    ) {
        Diag{ctx, Kind::ICError, where, fmt::format(fmt, std::forward<Args>(args)...)};
        std::terminate(); /// Should never be reached.
    }

    /// Raise a fatal error and exit.
    ///
    /// This is NOT an ICE; instead it is an error that is probably caused by
    /// the underlying system, such as attempting to output to a directory that
    /// isn’t accessible to the user.
    template <typename... Args>
    [[noreturn]] static void Fatal(
        utils::FStringWithSrcLoc<Args...> fmt,
        Args&&... args
    ) {
        Diag{Kind::FError, fmt::format(fmt.fmt, std::forward<Args>(args)...), fmt.sloc};
        std::terminate(); /// Should never be reached.
    }
};

} // namespace src

#endif // SOURCE_INCLUDE_CONTEXT_HH
