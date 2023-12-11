#include <cpptrace/cpptrace.hpp>
#include <llvm/ADT/StringExtras.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Unicode.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <random>
#include <source/Core.hh>
#include <source/Frontend/AST.hh>
#include <thread>
#include <zstd.h>

#ifndef _WIN32
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

/// ===========================================================================
///  Context
/// ===========================================================================
src::Context::~Context() = default;

src::Context::Context() {
    Initialise();
}

auto src::Context::get_or_load_file(fs::path path) -> File& {
    auto f = rgs::find_if(owned_files, [&](const auto& e) { return e->path() == path; });
    if (f != owned_files.end()) return **f;

    /// Load the file.
    auto contents = src::File::LoadFileData(path);
    return MakeFile(std::move(path), std::move(contents));
}

/// Implemented in HLIR.cc
namespace mlir::hlir {
void InitContext(MLIRContext& mctx);
}

void src::Context::Initialise() {
    llvm::InitializeNativeTarget();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    mlir::hlir::InitContext(mlir);

    clang.createDiagnostics();
    clang.getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();
    clang.createTarget();
    clang.createSourceManager(*clang.createFileManager());
    clang.createPreprocessor(clang::TU_Prefix);
    clang.createASTContext();
    clang.getDiagnostics().setShowColors(true);
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
    info.line_end = data + pos;
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

auto src::Location::text(const Context* ctx) const -> std::string_view {
    auto& files = ctx->files();
    auto* f = files[file_id].get();
    return std::string_view{f->data(), f->size()}.substr(pos, len);
}

/// ===========================================================================
///  Module
/// ===========================================================================
src::Module::Module(Context* ctx, std::string name, bool is_cxx_header, Location module_decl_location)
    : context(ctx),
      name(std::move(name)),
      module_decl_location(module_decl_location),
      is_cxx_header(is_cxx_header) {
    top_level_func = new (this) ProcDecl{
        this,
        nullptr,
        is_logical_module ? module_initialiser_name() : "__src_main",
        new (this) ProcType({}, BuiltinType::Void(this), false, {}),
        {},
        Linkage::Exported,
        Mangling::None,
        {},
    };

    top_level_func->body = new (this) BlockExpr{this, {}};
}

src::Module::~Module() {
    for (auto ex : exprs) utils::Deallocate(ex);
}

auto src::Module::_global_scope() -> BlockExpr* {
    return top_level_func->body;
}

/// ===========================================================================
///  Diagnostics
/// ===========================================================================
namespace src {
namespace {
/// Get the colour of a diagnostic.
static constexpr auto Colour(Diag::Kind kind) {
    using Kind = Diag::Kind;
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
static constexpr std::string_view Name(Diag::Kind kind) {
    using Kind = Diag::Kind;
    switch (kind) {
        case Kind::ICError: return "Internal Compiler Error";
        case Kind::FError: return "Fatal Error";
        case Kind::Error: return "Error";
        case Kind::Warning: return "Warning";
        case Kind::Note: return "Note";
        default: return "Diagnostic";
    }
}

/// Remove project directory from filename.
auto NormaliseFilename(std::string_view filename) -> std::string_view {
    if (auto pos = filename.find(__SRCC_PROJECT_DIR_NAME); pos != std::string_view::npos) {
        static constexpr std::string_view name{__SRCC_PROJECT_DIR_NAME};
        filename.remove_prefix(pos + name.size() + 1);
    }
    return filename;
}

/// Print the current stack trace.
void PrintBacktrace() {
    std::fflush(stdout);
    std::fflush(stderr);

    utils::Colours C{true};
    using enum utils::Colour;

    auto trace = cpptrace::generate_trace(2);
    for (auto&& [i, frame] : vws::enumerate(trace.frames)) {
        std::string_view symbol = frame.symbol;
        std::string sym;

        /// Helper to remove template parameters.
        auto SkipTemplateParams = [&] {
            /// Skip matching angle brackets.
            usz brackets = 1;
            usz j = 0;
            for (; j < symbol.size(); j++) {
                if (symbol[j] == '<') brackets++;
                else if (symbol[j] == '>') {
                    brackets--;
                    if (brackets == 0) break;
                }
            }

            /// Remove everything up to and including the closing angle bracket.
            symbol.remove_prefix(std::min(j + 1, symbol.size()));
        };

        /// Clean up the symbol name.
        for (;;) {
            /// Seek next occurrence of `std::`, ` >`, or `src::.
            auto pos = std::min({
                symbol.find("std::"sv),
                symbol.find(" >"sv),
                symbol.find("src::"sv),
                symbol.find("fmt::"sv),
            });

            /// Nothing found.
            if (pos == std::string_view::npos) {
                sym += symbol;
                break;
            }

            /// Append everything up to the position.
            sym += symbol.substr(0, pos);
            symbol.remove_prefix(pos);

            /// If we found ` >` or `src::`, simply remove them and continue.
            if (symbol.starts_with(" >")) {
                symbol.remove_prefix(" >"sv.size());
                sym += ">";
                continue;
            } else if (symbol.starts_with("src::")) {
                symbol.remove_prefix("src::"sv.size());
                continue;
            }

            /// Otherwise, append and remove `std::`, but keep going.
            if (symbol.starts_with("std::")) {
                symbol.remove_prefix("std::"sv.size());
                sym += "std::";

                /// Remove `__cxx11::`.
                if (symbol.starts_with("__cxx11::")) symbol.remove_prefix("__cxx11::"sv.size());

                /// Prettify std::string.
                if (symbol.starts_with("basic_string<char")) {
                    symbol.remove_prefix("basic_string<char"sv.size());
                    SkipTemplateParams();
                    sym += "string";
                }
                continue;
            }

            /// Prettify fmt::format_string.
            if (symbol.starts_with("fmt::")) {
                symbol.remove_prefix("fmt::"sv.size());
                sym += "fmt::";

                /// Remove version.
                if (symbol.size() >= 2 and symbol[0] == 'v' and std::isdigit(u8(symbol[1]))) {
                    auto nmsp = symbol.find("::"sv);
                    if (nmsp == std::string_view::npos) continue;
                    symbol.remove_prefix(nmsp + "::"sv.size());
                }

                /// If we’re looking at `format_string`, remove its template parameters.
                if (symbol.starts_with("basic_format_string<")) {
                    symbol.remove_prefix("basic_format_string<"sv.size());
                    SkipTemplateParams();
                    sym += "format_string<...>";
                }
            }
        }

        /// Use a relative filepath for less noise.
        std::string_view filename = NormaliseFilename(frame.filename);

        /// Print the line.
        fmt::print(
            stderr,
            "#{:<{}} {}{} {}in {}{} {}at {}{}{}:{}{}{}\n",
            i,
            utils::NumberWidth(trace.frames.size()),
            C(Blue),
            fmt::ptr((void*)frame.address),
            C(Reset),
            C(Yellow),
            sym,
            C(Reset),
            C(Green),
            filename,
            C(Reset),
            C(Blue),
            u64(frame.line.raw_value),
            C(Reset)
        );

        /// Stop at main.
        if (frame.symbol == "main") break;
    }

    std::fflush(stdout);
    std::fflush(stderr);
}
} // namespace
} // namespace src

/// Abort due to assertion failure.
void src::detail::AssertFail(
    AssertKind k,
    std::string_view condition,
    std::string_view file,
    int line,
    std::string&& message
) {
    using fmt::fg;
    using enum fmt::emphasis;
    using enum fmt::terminal_color;

    /// Print filename and ICE title.
    fmt::print(stderr, bold, "{}:{}:", NormaliseFilename(file), line);
    fmt::print(stderr, Colour(Diag::Kind::ICError), " {}: ", Name(Diag::Kind::ICError));

    /// Print the condition, if any.
    switch (k) {
        case AssertKind::AK_Assert:
            fmt::print(stderr, "Assertion failed: '{}'", condition);
            break;

        case AssertKind::AK_Todo:
            fmt::print(stderr, "TODO");
            break;

        case AssertKind::AK_Unreachable:
            fmt::print(stderr, "Unreachable code reached");
            break;
    }

    /// Print the message.
    if (not message.empty()) fmt::print(stderr, ": {}", message);
    fmt::print(stderr, "\n");

    /// Print the backtrace and exit.
    PrintBacktrace();
    std::exit(Diag::ICEExitCode);
}

void src::Diag::HandleFatalErrors() {
    /// Abort on ICE.
    if (kind == Kind::ICError) {
        if (include_stack_trace) PrintBacktrace();
        std::exit(ICEExitCode);
    }

    /// Exit on a fatal error.
    if (kind == Kind::FError) {
        if (include_stack_trace)  PrintBacktrace();
        std::exit(FatalExitCode);
    }
}

/// Print a diagnostic with no (valid) location info.
void src::Diag::PrintDiagWithoutLocation() {
    /// Print the message.
    fmt::print(stderr, Colour(kind), "{}: ", Name(kind));
    fmt::print(stderr, "{}\n", msg);
    HandleFatalErrors();
}

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

    /// First, reset the colour.
    fmt::print(stderr, "\033[m");

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
    std::string range(line_start + col, std::min<u64>(where.len, u64(line_end - (line_start + col))));
    auto after = line_start + col + where.len > line_end
                   ? std::string{}
                   : std::string(line_start + col + where.len, line_end);

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

    /// LLVM’s columnWidthUTF8() function returns -1 for non-printable characters
    /// for some ungodly reason, so guard against that.
    static const auto ColumnWidth = [](StringRef text) {
        auto wd = llvm::sys::unicode::columnWidthUTF8(text);
        return wd < 0 ? 0 : usz(wd);
    };

    /// Underline the range. For that, we first pad the line based on the number
    /// of digits in the line number and append more spaces to line us up with
    /// the range.
    for (usz i = 0, end = digits + ColumnWidth(before) + sizeof("  | ") - 1; i < end; i++)
        fmt::print(stderr, " ");

    /// Finally, underline the range.
    for (usz i = 0, end = ColumnWidth(range); i < end; i++)
        fmt::print(stderr, Colour(kind), "~");
    fmt::print(stderr, "\n");

    /// Handle fatal errors.
    HandleFatalErrors();
}

/// ===========================================================================
///  Utils
/// ===========================================================================
void src::utils::Compress(
    SmallVectorImpl<u8>& into,
    ArrayRef<u8> data,
    int compression_level
) {
    const auto oldsz = into.size();
    const auto bound = ::ZSTD_compressBound(data.size());
    into.resize_for_overwrite(bound + oldsz);

    /// We invoke ZSTD manually here instead of using the
    /// LLVM wrapper since that one clears the buffer before
    /// writing to it, which is not what we want.
    const usz sz = ::ZSTD_compress(
        into.data() + oldsz,
        bound,
        data.data(),
        data.size(),
        compression_level
    );

    if (::ZSTD_isError(sz)) Diag::Fatal("compression failed: {}", sz);
    into.resize(oldsz + sz);
}

void src::utils::Decompress(
    SmallVectorImpl<u8>& into,
    ArrayRef<u8> data,
    usz uncompressed_size
) {
    const auto oldsz = into.size();
    into.resize_for_overwrite(oldsz + uncompressed_size);

    auto sz = ::ZSTD_decompress(
        into.data() + oldsz,
        uncompressed_size,
        data.data(),
        data.size()
    );

    if (::ZSTD_isError(sz)) Diag::Fatal("decompression failed: {}", sz);
    if (sz != uncompressed_size) Diag::Fatal("Invalid uncompressed size");
}

auto src::utils::Escape(StringRef str) -> std::string {
    std::string s;
    llvm::raw_string_ostream os{s};
    os.write_escaped(str, true);
    return s;
}

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
