#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Unicode.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <random>
#include <source/Core.hh>
#include <source/Frontend/AST.hh>
#include <thread>

#ifndef _WIN32
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

#ifdef __linux__
#    include <execinfo.h>
#    include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#endif

/// ===========================================================================
///  Context
/// ===========================================================================
src::Context::~Context() = default;

src::Context::Context(const llvm::Target* tgt)
    : tgt(tgt) {
    Initialise();
}

src::Context::Context() {
    Initialise();

    std::string error;
    auto target_triple = llvm::sys::getDefaultTargetTriple();
    llvm::Triple triple{target_triple};
    auto target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if (not target) Diag::Fatal("Failed to lookup target '{}': {}", target_triple, error);
    tgt = target;
}

auto src::Context::get_or_load_file(fs::path path) -> File& {
    auto f = rgs::find_if(owned_files, [&](const auto& e) { return e->path() == path; });
    if (f != owned_files.end()) return **f;

    /// Load the file.
    auto contents = src::File::LoadFileData(path);
    return MakeFile(std::move(path), std::move(contents));
}

void src::Context::Initialise() {
    llvm::InitializeNativeTarget();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    mlir::registerAllDialects(mlir);
    mlir::registerBuiltinDialectTranslation(mlir);
    mlir::registerLLVMDialectTranslation(mlir);
    mlir.loadDialect< // clang-format off
        mlir::func::FuncDialect,
        mlir::scf::SCFDialect,
        mlir::affine::AffineDialect,
        mlir::LLVM::LLVMDialect,
        mlir::index::IndexDialect
    >(); // clang-format on
    mlir.printOpOnDiagnostic(true);
    mlir.printStackTraceOnDiagnostic(true);
}

auto src::Context::MakeFile(fs::path name, std::vector<char>&& contents) -> File& {
    /// Create the file.
    auto fptr = new File(*this, std::move(name), std::move(contents));
    fptr->id = u32(owned_files.size());
    Assert(fptr->id <= std::numeric_limits<u16>::max());
    owned_files.emplace_back(fptr);
    return *fptr;
}

/// ===========================================================================
///  File
/// ===========================================================================
auto src::File::TempPath(std::string_view extension) -> fs::path {
    std::mt19937 rd(std::random_device{}());

    /// Get the temporary directory.
    auto tmp_dir = std::filesystem::temp_directory_path();

    /// Use the pid on Linux, and another random number on Windows.
#ifndef _WIN32
    auto pid = std::to_string(u32(::getpid()));
#else
    auto pid = std::to_string(rd());
#endif

    /// Get the current time and tid.
    auto now = chr::system_clock::now().time_since_epoch().count();
    auto tid = std::to_string(u32(std::hash<std::thread::id>{}(std::this_thread::get_id())));

    /// And some random letters too.
    /// Do NOT use `char` for this because it’s signed on some systems (including mine),
    /// which completely breaks the modulo operation below... Thanks a lot, C.
    std::array<u8, 8> rand{};
    rgs::generate(rand, [&] { return rd() % 26 + 'a'; });

    /// Create a unique file name.
    auto tmp_name = fmt::format(
        "{}.{}.{}.{}",
        pid,
        tid,
        now,
        std::string_view{(char*) rand.data(), rand.size()}
    );

    /// Append it to the temporary directory.
    auto f = tmp_dir / tmp_name;
    if (not extension.empty()) {
        if (not extension.starts_with('.')) f += '.';
        f += extension;
    }
    return f;
}

bool src::File::Write(void* data, usz size, const fs::path& file) {
    auto f = std::fopen(file.string().c_str(), "wb");
    if (not f) return false;
    defer { std::fclose(f); };
    for (;;) {
        auto written = std::fwrite(data, 1, size, f);
        if (written == size) break;
        if (written < 1) return false;
        data = (char*) data + written;
        size -= written;
    }
    return true;
}

void src::File::WriteOrDie(void* data, usz size, const fs::path& file) {
    if (not src::File::Write(data, size, file))
        Diag::Fatal("Failed to write to file '{}': {}", file.string(), std::strerror(errno));
}

src::File::File(Context& ctx, fs::path name, std::vector<char>&& contents)
    : ctx(ctx), file_path(std::move(name)), contents(std::move(contents)) {}

auto src::File::LoadFileData(const fs::path& path) -> std::vector<char> {
#ifdef __linux__
    /// Open the file.
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) Diag::Fatal("Could not open file \"{}\": {}", path.string(), strerror(errno));
    defer { close(fd); };

    /// Determine the file size.
    struct stat st {};
    if (fstat(fd, &st) == -1) Diag::Fatal("Could not stat file \"{}\": {}", path.string(), strerror(errno));

    /// If the file is empty, return an empty string.
    if (st.st_size == 0) return {};

    /// Map the file into memory.
    void* ptr = mmap(nullptr, static_cast<usz>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) Diag::Fatal("Could not map file \"{}\": {}", path.string(), strerror(errno));

    /// Copy the file into a vector.
    std::vector<char> ret(static_cast<usz>(st.st_size));
    memcpy(ret.data(), ptr, static_cast<usz>(st.st_size));

    /// Unmap the file.
    munmap(ptr, static_cast<usz>(st.st_size));

#else
    /// Read the file manually.
    std::unique_ptr<FILE, decltype(&std::fclose)> f{std::fopen(path.string().c_str(), "rb"), std::fclose};
    if (not f) Diag::Fatal("Could not open file \"{}\": {}", path.string(), strerror(errno));

    /// Get the file size.
    std::fseek(f.get(), 0, SEEK_END);
    auto sz = std::size_t(std::ftell(f.get()));
    std::fseek(f.get(), 0, SEEK_SET);

    /// Read the file.
    std::vector<char> ret;
    ret.resize(sz);
    std::size_t n_read = 0;
    while (n_read < sz) {
        errno = 0;
        auto n = std::fread(ret.data() + n_read, 1, sz - n_read, f.get());
        if (errno) Diag::Fatal("Error reading file \"{}\": {}", path.string(), strerror(errno));
        if (n == 0) break;
        n_read += n;
    }
#endif

    /// Construct the file data.
    return ret;
}

/// ===========================================================================
///  Location
/// ===========================================================================
bool src::Location::seekable(const Context* ctx) const {
    auto& files = ctx->files();
    if (file_id >= files.size()) return false;
    const auto* f = files[file_id].get();
    return pos + len <= f->size() and is_valid();
}

/// Seek to a source location. The location must be valid.
auto src::Location::seek(const Context* ctx) const -> LocInfo {
    LocInfo info{};

    /// Get the file that the location is in.
    auto& files = ctx->files();
    const auto* f = files.at(file_id).get();

    /// Seek back to the start of the line.
    const char* const data = f->data();
    info.line_start = data + pos;
    while (info.line_start > data and *info.line_start != '\n') info.line_start--;
    if (*info.line_start == '\n') info.line_start++;

    /// Seek forward to the end of the line.
    const char* const end = data + f->size();
    info.line_end = data + pos + len;
    while (info.line_end < end and *info.line_end != '\n') info.line_end++;

    /// Determine the line and column number.
    info.line = 1;
    for (const char* d = data; d < data + pos; d++) {
        if (*d == '\n') {
            info.line++;
            info.col = 0;
        } else {
            info.col++;
        }
    }

    /// Done!
    return info;
}

/// TODO: Lexer should create map that counts where in a file the lines start so
/// we can do binary search on that instead of iterating over the entire file.
auto src::Location::seek_line_column(const Context* ctx) const -> LocInfoShort {
    LocInfoShort info{};

    /// Get the file that the location is in.
    auto& files = ctx->files();
    const auto* f = files.at(file_id).get();

    /// Seek back to the start of the line.
    const char* const data = f->data();

    /// Determine the line and column number.
    info.line = 1;
    for (const char* d = data; d < data + pos; d++) {
        if (*d == '\n') {
            info.line++;
            info.col = 0;
        } else {
            info.col++;
        }
    }

    /// Done!
    return info;
}

/// ===========================================================================
///  Module
/// ===========================================================================
src::Module::Module(Context* ctx, std::string name, Location module_decl_location)
    : context(ctx),
      name(std::move(name)),
      module_decl_location(module_decl_location) {
    /// Create the global scope. The scope is automatically
    /// added to our list of scopes by operator new.
    new (this) Scope{nullptr, this};
    top_level_func = new (this) ProcDecl{
        this,
        is_logical_module ? fmt::format("_S.static.initialisation.{}", name) : "__src_main",
        new (this) ProcType({}, BuiltinType::Void(this), {}),
        {},
        new (this) BlockExpr{global_scope, {}, {}},
        Linkage::Exported,
        Mangling::None,
        {},
    };

}

/// ===========================================================================
///  Diagnostics
/// ===========================================================================
namespace {
/// Get the colour of a diagnostic.
static constexpr auto Colour(src::Diag::Kind kind) {
    using Kind = src::Diag::Kind;
    switch (kind) {
        case Kind::ICError: return fmt::fg(fmt::terminal_color::magenta) | fmt::emphasis::bold;
        case Kind::Warning: return fmt::fg(fmt::terminal_color::yellow) | fmt::emphasis::bold;
        case Kind::Note: return fmt::fg(fmt::terminal_color::green) | fmt::emphasis::bold;

        case Kind::FError:
        case Kind::Error:
            return fmt::fg(fmt::terminal_color::red) | fmt::emphasis::bold;

        default:
            return fmt::text_style{};
    }
}

/// Get the name of a diagnostic.
static constexpr std::string_view Name(src::Diag::Kind kind) {
    using Kind = src::Diag::Kind;
    switch (kind) {
        case Kind::ICError: return "Internal Compiler Error";
        case Kind::FError: return "Fatal Error";
        case Kind::Error: return "Error";
        case Kind::Warning: return "Warning";
        case Kind::Note: return "Note";
        default: return "Diagnostic";
    }
}

#ifdef __linux__
/// Print the current stack trace.
void PrintBacktrace() {
    /// Get the backtrace.
    static void* trace[128];
    int n = backtrace(trace, 128);

    /// Convert to strings.
    std::vector<std::string> trace_strs;
    trace_strs.reserve(src::usz(n));
    for (int i = 0; i < n; i++) trace_strs.emplace_back(fmt::format("{:p}", trace[i]));

    /// Symboliser path.
    std::string sym = std::getenv("SYMBOLIZER_PATH") ?: "";
    if (sym.empty()) sym = "llvm-symbolizer";

    /// Use llvm-symbolizer to print the backtrace.
    auto cmd = fmt::format(
        "{} {} -e {} -s -p -C -i --color --output-style=GNU | awk '{{ print \"#\" NR, $0 }}'",
        sym,
        fmt::join(trace_strs, " "),
        src::fs::canonical("/proc/self/exe").native()
    );
    std::system(cmd.c_str());
}
#else
void PrintBacktrace() {
    /// TODO: Implement this for other platforms.
}
#endif
} // namespace

/// Abort due to assertion failure.
[[noreturn]] void src::detail::AssertFail(std::string&& msg) {
    Diag::ICE("{}", std::move(msg));
}

void src::Diag::HandleFatalErrors() {
    /// Abort on ICE.
    if (kind == Kind::ICError) {
        PrintBacktrace();
        std::exit(ICEExitCode);
    }

    /// Exit on a fatal error.
    if (kind == Kind::FError)
        std::exit(FatalExitCode); /// Separate line so we can put a breakpoint here.
}

/// Print a diagnostic with no (valid) location info.
void src::Diag::PrintDiagWithoutLocation() {
    /// Print the message.
    fmt::print(stderr, Colour(kind), "{}: ", Name(kind));
    fmt::print(stderr, "{}\n", msg);
    HandleFatalErrors();
}

src::Diag::~Diag() { print(); }

void src::Diag::print() {
    using fmt::fg;
    using enum fmt::emphasis;
    using enum fmt::terminal_color;

    /// If this diagnostic is suppressed, do nothing.
    if (kind == Kind::None) return;

    /// Don’t print the same diagnostic twice.
    defer { kind = Kind::None; };

    /// If the diagnostic is an error, set the error flag.
    if (kind == Kind::Error and ctx)
        ctx->set_error(); /// Separate line so we can put a breakpoint here.

    /// If there is no context, then there is also no location info.
    if (not ctx) {
        PrintDiagWithoutLocation();
        return;
    }

    /// If the location is invalid, either because the specified file does not
    /// exists, its position is out of bounds or 0, or its length is 0, then we
    /// skip printing the location.
    const auto& fs = ctx->files();
    if (not where.seekable(ctx)) {
        /// Even if the location is invalid, print the file name if we can.
        if (where.file_id < fs.size()) {
            const auto& file = *fs[where.file_id].get();
            fmt::print(stderr, bold, "{}: ", file.path().string());
        }

        /// Print the message.
        PrintDiagWithoutLocation();
        return;
    }

    /// If the location is valid, get the line, line number, and column number.
    const auto [line, col, line_start, line_end] = where.seek(ctx);

    /// Split the line into everything before the range, the range itself,
    /// and everything after.
    std::string before(line_start, col);
    std::string range(line_start + col, where.len);
    std::string after(line_start + col + where.len, line_end);

    /// Replace tabs with spaces. We need to do this *after* splitting
    /// because this invalidates the offsets.
    utils::ReplaceAll(before, "\t", "    ");
    utils::ReplaceAll(range, "\t", "    ");
    utils::ReplaceAll(after, "\t", "    ");

    /// Print the file name, line number, and column number.
    const auto& file = *fs[where.file_id].get();
    fmt::print(stderr, bold, "{}:{}:{}: ", file.path().string(), line, col);

    /// Print the diagnostic name and message.
    fmt::print(stderr, Colour(kind), "{}: ", Name(kind));
    fmt::print(stderr, "{}\n", msg);

    /// Print the line up to the start of the location, the range in the right
    /// colour, and the rest of the line.
    fmt::print(stderr, " {} | {}", line, before);
    fmt::print(stderr, Colour(kind), "{}", range);
    fmt::print(stderr, "{}\n", after);

    /// Determine the number of digits in the line number.
    const auto digits = utils::NumberWidth(line);

    /// Underline the range. For that, we first pad the line based on the number
    /// of digits in the line number and append more spaces to line us up with
    /// the range.
    for (usz i = 0, end = digits + usz(llvm::sys::unicode::columnWidthUTF8(before)) + sizeof("  | ") - 1; i < end; i++)
        fmt::print(stderr, " ");

    /// Finally, underline the range.
    for (usz i = 0, end = usz(llvm::sys::unicode::columnWidthUTF8(range)); i < end; i++)
        fmt::print(stderr, Colour(kind), "~");
    fmt::print(stderr, "\n");

    /// Handle fatal errors.
    HandleFatalErrors();
}

/// ===========================================================================
///  Utils
/// ===========================================================================
void src::utils::ReplaceAll(
    std::string& str,
    std::string_view from,
    std::string_view to
) {
    if (from.empty()) return;
    for (usz i = 0; i = str.find(from, i), i != std::string::npos; i += to.length())
        str.replace(i, from.length(), to);
}

auto src::utils::NumberWidth(usz number, usz base) -> usz {
    return number == 0 ? 1 : usz(std::log(number) / std::log(base) + 1);
}
