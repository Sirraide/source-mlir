#ifndef SOURCE_INCLUDE_CONTEXT_HH
#define SOURCE_INCLUDE_CONTEXT_HH

#include <clang/Frontend/CompilerInstance.h>
#include <llvm/IR/Module.h>
#include <mlir/IR/MLIRContext.h>
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
/// This is to avoid having to include Value.h everywhere.
inline constexpr usz AlignOfMLIRType = 8;
inline constexpr usz AlignOfMLIRValue = 8;
inline constexpr usz SizeOfMLIRType = 8;
inline constexpr usz SizeOfMLIRValue = 8;

#define SOURCE_MLIR_VALUE_MEMBER(name)                                          \
    alignas(::src::AlignOfMLIRValue) char _##name##_[::src::SizeOfMLIRValue]{}; \
    property_decl(mlir::Value, name)

#define SOURCE_MLIR_TYPE_MEMBER(name)                                         \
    alignas(::src::AlignOfMLIRType) char _##name##_[::src::SizeOfMLIRType]{}; \
    property_decl(mlir::Type, name)

class Context;
class Module;
class Decl;
class ProcDecl;
class Expr;
class BlockExpr;
class StructType;

/// A file in the context.
class File {
    /// Context handle.
    Context& ctx;

    /// The name of the file.
    fs::path file_path;

    /// The contents of the file.
    std::vector<char> contents;

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
    [[nodiscard]] auto begin() const { return contents.begin(); }

    /// Get the file data.
    [[nodiscard]] auto data() const -> const char* { return contents.data(); }

    /// Get an iterator to the end of the file.
    [[nodiscard]] auto end() const { return contents.end(); }

    /// Get the id of this file.
    [[nodiscard]] auto file_id() const { return id; }

    /// Get the file path.
    [[nodiscard]] auto path() const -> const fs::path& { return file_path; }

    /// Get the size of the file.
    [[nodiscard]] auto size() const -> usz { return contents.size(); }

private:
    /// Construct a file from a name and source.
    explicit File(Context& _ctx, fs::path _name, std::vector<char>&& _contents);

    /// Load a file from disk.
    static auto LoadFileData(const fs::path& path) -> std::vector<char>;

    /// The context is the only thing that can create files.
    friend Context;
};

class Context {
    /// The files owned by the context.
    std::vector<std::unique_ptr<File>> owned_files;

    /// Error flag. This is set-only.
    mutable bool error_flag = false;

public:
    /// Contexts.
    mlir::MLIRContext mlir;
    llvm::LLVMContext llvm;
    clang::CompilerInstance clang;
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> file_system;
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> in_memory_fs;

    /// Import paths.
    std::vector<fs::path> import_paths;

    /// Modules in the context.
    std::vector<std::unique_ptr<Module>> modules;

public:
    friend Module;

    /// Create a context for the host target.
    explicit Context();

    /// Do not allow copying or moving the context.
    Context(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;

    /// Delete context data.
    ~Context();

    /// Create a new file from a name and contents.
    template <typename Buffer>
    File& create_file(fs::path name, Buffer&& contents) {
        return MakeFile(
            std::move(name),
            std::vector<char>{std::forward<Buffer>(contents)}
        );
    }

    /// Get a list of all files owned by the context.
    [[nodiscard]] auto files() const -> const decltype(owned_files)& {
        return owned_files;
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
    [[nodiscard]] bool has_error() const { return error_flag; }

    /// Set the error flag.
    ///
    /// \return The previous value of the error flag.
    bool set_error() const {
        auto old = error_flag;
        error_flag = true;
        return old;
    }

private:
    /// Initialise the context.
    void Initialise();

    /// Initialise the MLIR context. Implemented in HLIR.cc.
    void InitialiseMLIRContext();

    /// Register a file in the context.
    File& MakeFile(fs::path name, std::vector<char>&& contents);
};

/// A decoded source location.
struct LocInfo {
    usz line;
    usz col;
    const char* line_start;
    const char* line_end;
};

/// A short decoded source location.
struct LocInfoShort {
    usz line;
    usz col;
};

/// A source range in a file.
struct Location {
    u32 pos{};
    u16 len{};
    u16 file_id{};

    constexpr Location() = default;
    constexpr Location(u32 pos, u16 len, u16 file_id)
        : pos(pos), len(len), file_id(file_id) {}

    /// Create a new location that spans two locations.
    constexpr Location(Location a, Location b) {
        if (a.file_id != b.file_id) return;
        if (not a.is_valid() or not b.is_valid()) return;
        pos = std::min<u32>(a.pos, b.pos);
        len = u16(std::max<u32>(a.pos + a.len, b.pos + b.len) - pos);
    }

    /// Shift a source location to the left.
    [[nodiscard]] constexpr auto operator<<(isz amount) const -> Location {
        Location l = *this;
        l.pos = u32(pos - u32(amount));
        return l;
    }

    /// Shift a source location to the right.
    [[nodiscard]] constexpr auto operator>>(isz amount) const -> Location {
        Location l = *this;
        l.pos = u32(pos + u32(amount));
        return l;
    }

    /// Extend a source location to the left.
    [[nodiscard]] constexpr auto operator<<=(isz amount) const -> Location {
        Location l = *this << amount;
        l.len = u16(l.len + amount);
        return l;
    }

    /// Extend a source location to the right.
    [[nodiscard]] constexpr auto operator>>=(isz amount) const -> Location {
        Location l = *this;
        l.len = u16(l.len + amount);
        return l;
    }

    /// Contract a source location to the left.
    [[nodiscard]] constexpr auto contract_left(isz amount) const -> Location {
        if (amount > len) return {};
        Location l = *this;
        l.len = u16(l.len - amount);
        return l;
    }

    /// Contract a source location to the right.
    [[nodiscard]] constexpr auto contract_right(isz amount) const -> Location {
        if (amount > len) return {};
        Location l = *this;
        l.pos = u32(l.pos + u32(amount));
        l.len = u16(l.len - amount);
        return l;
    }

    /// Encode a location as a 64-bit number.
    [[nodiscard]] constexpr u64 encode() { return std::bit_cast<u64>(*this); }

    [[nodiscard]] constexpr bool is_valid() const { return len != 0; }

    /// Get this location as an MLIR Location. Implemented in CodeGen.cc.
    [[nodiscard]] auto mlir(Context* ctx) const -> mlir::Location;

    /// Seek to a source location.
    [[nodiscard]] auto seek(const Context* ctx) const -> LocInfo;

    /// Seek to a source location, but only return the line and column.
    [[nodiscard]] auto seek_line_column(const Context* ctx) const -> LocInfoShort;

    /// Check if the source location is seekable.
    [[nodiscard]] bool seekable(const Context* ctx) const;

    /// Get the text pointed to by this source location.
    [[nodiscard]] auto text(const Context* ctx) const -> std::string_view;

    /// Decode a source location from a 64-bit number.
    static constexpr auto Decode(u64 loc) -> Location {
        return std::bit_cast<Location>(loc);
    }
};

/// Reference to an imported module.
struct ImportedModuleRef {
    /// The actual name of the module for linkage purposes.
    std::string linkage_name;

    /// The name of the module in code.
    std::string logical_name;

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
};

/// A Source module.
class Module {
public:
    Context* const context;

    /// Modules imported by this module.
    SmallVector<ImportedModuleRef> imports;

    /// Exported declarations.
    StringMap<SmallVector<Expr*, 1>> exports;

    /// Named structs.
    SmallVector<StructType*, 64> named_structs;

    /// Top-level module function.
    ProcDecl* top_level_func;

    /// Module string table.
    StringTable strtab;

    /// AST nodes in this module.
    SmallVector<Expr*> exprs;

    /// Functions that are part of this module.
    SmallVector<ProcDecl*> functions;

    /// Static assertions that are not part of a template go here.
    SmallVector<Expr*, 32> static_assertions;

    /// Module name.
    std::string name;

    /// Location of the module declaration.
    Location module_decl_location;

    /// Whether this is a logical module.
    readonly(bool, is_logical_module, return not name.empty());

    /// Whether this is a C++ header.
    bool is_cxx_header;

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

    /// An empty name means this isn’t a logical module.
    explicit Module(
        Context* ctx,
        std::string name,
        bool is_cxx_header = false,
        Location module_decl_location = {}
    );
    ~Module();

    /// Add a function to this module.
    void add_function(ProcDecl* func) { functions.push_back(func); }

    /// Get the name to use for the module description section in the object file.
    [[nodiscard]] auto description_section_name() const -> std::string {
        return fmt::format(".__src_module__description__.{}", name);
    }

    /// Emit an executable. Implemented in Emit.cc.
    void emit_executable(int opt_level, const fs::path& location);

    /// Emit code to an object file. Implemented in Emit.cc
    void emit_object_file(int opt_level, const fs::path& location);

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
    void print_ast(bool use_colour) const;

    /// Print the AST of any exported symbols.
    void print_exports(bool use_colour) const;

    /// Print the HLIR of the module. Implemented in CodeGen.cc.
    void print_hlir(bool use_generic_assembly_format) const;

    /// Print the module as LLVM IR. Implemented in HLIRLowering.cc.
    void print_llvm(int opt_level);

    /// Execute the module. Implemented in HLIRLowering.cc.
    int run(int opt_level);

    /// Serialise the module to a description that can be saved and
    /// loaded later. Implemented in Endec.cc.
    auto serialise() -> SmallVector<u8>;


    /// Deserialise a module from a module description. Implemented in Endec.cc.
    static auto Deserialise(
        Context* ctx,
        std::string module_name,
        Location loc,
        ArrayRef<u8> description
    ) -> Module*;

    /// Import a C++ header.
    static auto ImportCXXHeaders(
        Context* ctx,
        ArrayRef<StringRef> header_names,
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
    std::string msg;

    /// Handle fatal error codes.
    void HandleFatalErrors();

    /// Print a diagnostic with no (valid) location info.
    void PrintDiagWithoutLocation();

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
    Diag(Kind _kind, std::string&& msg)
        : ctx(nullptr), kind(_kind), where(), msg(std::move(msg)) {}

    /// Issue a diagnostic with a format string and arguments.
    template <typename... Args>
    Diag(
        const Context* ctx,
        Kind kind,
        Location where,
        fmt::format_string<Args...> fmt,
        Args&&... args
    )
        : Diag{ctx, kind, where, fmt::format(fmt, std::forward<Args>(args)...)} {}

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
    static Diag Error(fmt::format_string<Args...> fmt, Args&&... args) {
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
    [[noreturn]] static void ICE(fmt::format_string<Args...> fmt, Args&&... args) {
        Diag{Kind::ICError, fmt::format(fmt, std::forward<Args>(args)...)};
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
    [[noreturn]] static void Fatal(fmt::format_string<Args...> fmt, Args&&... args) {
        Diag{Kind::FError, fmt::format(fmt, std::forward<Args>(args)...)};
        std::terminate(); /// Should never be reached.
    }

    /// Same as Fatal(), but do not print a stacktrace.
    template <typename... Args>
    [[noreturn]] static void FatalNoTrace(fmt::format_string<Args...> fmt, Args&&... args) {
        Diag{Kind::FError, fmt::format(fmt, std::forward<Args>(args)...)}.NoTrace();
        std::terminate(); /// Should never be reached.
    }
};

} // namespace src

#endif // SOURCE_INCLUDE_CONTEXT_HH
