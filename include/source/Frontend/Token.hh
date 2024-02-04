#ifndef SOURCE_TOKEN_HH
#define SOURCE_TOKEN_HH

#include <source/Support/Utils.hh>

namespace src {
class Context;

/// ===========================================================================
///  Source Locations
/// ===========================================================================
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

/// ===========================================================================
///  Tokens.
/// ===========================================================================
enum struct Tk {
    Invalid,
    Eof,
    Identifier,
    CXXHeaderName,
    Integer,
    StringLiteral,
    MacroParameter,

    /// Keywords.
    Alias,
    And,
    As,
    AsBang,
    Asm,
    Assert,
    Bool,
    Break,
    Continue,
    CShort,
    Defer,
    Delete,
    Do,
    Dynamic,
    Elif,
    Else,
    Enum,
    Export,
    F32,
    F64,
    Fallthrough,
    False,
    For,
    ForReverse,
    Goto,
    If,
    Import,
    In,
    Init,
    Int,
    IntegerType,
    Is,
    Land,
    Lor,
    Match,
    NoReturn,
    Not,
    Or,
    Pragma,
    Proc,
    Return,
    Static,
    Struct,
    Then,
    True,
    Try,
    Type,
    Typeof,
    Unreachable,
    Val,
    Var,
    Variant,
    Void,
    While,
    With,
    Xor,

    /// Extension keywords.
    CChar,
    CChar8T,
    CChar16T,
    CChar32T,
    CInt,
    CLong,
    CLongDouble,
    CLongLong,
    CSizeT,
    CWCharT,

    /// Punctuation.
    Semicolon,
    Colon,
    ColonColon,
    Comma,
    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Ellipsis,
    Dot,
    LArrow,
    RArrow,
    RDblArrow,
    Question,

    /// Operators.
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Ampersand,
    VBar,
    Tilde,
    Bang,
    Assign,
    DotDot,
    DotDotLess,
    DotDotEq,
    MinusMinus,
    PlusPlus,
    StarStar,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Neq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    ShiftLeft,
    ShiftRight,
    ShiftRightLogical,
    ShiftLeftEq,
    ShiftRightEq,
    ShiftRightLogicalEq,
    StarStarEq,
};

/// A token.
struct Token {
    /// The type of the token.
    Tk type = Tk::Invalid;

    /// Token text.
    String text{};

    /// Number.
    APInt integer{};

    /// Source location.
    Location location{};

    /// Whether this token was produced by __id.
    bool artificial = false;

    /// Get the source code spelling of this token.
    readonly_decl(std::string, spelling);

    /// Compare two tokens for equality. This only checks if their
    /// types and values are equal and ignores e.g. whether they are
    /// artificial
    friend bool operator==(const Token& a, const Token& b);

    /// Helper to make creating tokens easier.
    template <typename... arguments>
    static auto Make(Tk t, arguments&&... args) -> Token {
        return Token{t, std::forward<arguments>(args)...};
    }
};

/// Stringify a token type.
constexpr auto Spelling(Tk t) -> String {
    switch (t) {
        case Tk::Invalid: return "<invalid>";
        case Tk::Eof: return "<eof>";
        case Tk::Identifier: return "<identifier>";
        case Tk::CXXHeaderName: return "<cxx header name>";
        case Tk::MacroParameter: return "<macro parameter>";
        case Tk::StringLiteral: return "<string literal>";
        case Tk::Integer: return "<integer>";
        case Tk::IntegerType: return "<integer type>";

        case Tk::Alias: return "alias";
        case Tk::And: return "and";
        case Tk::As: return "as";
        case Tk::AsBang: return "as!";
        case Tk::Asm: return "asm";
        case Tk::Assert: return "assert";
        case Tk::Bool: return "bool";
        case Tk::Break: return "break";
        case Tk::Defer: return "defer";
        case Tk::Delete: return "delete";
        case Tk::Do: return "do";
        case Tk::Dynamic: return "dynamic";
        case Tk::Elif: return "elif";
        case Tk::Else: return "else";
        case Tk::Enum: return "enum";
        case Tk::Export: return "export";
        case Tk::F32: return "f32";
        case Tk::F64: return "f64";
        case Tk::Fallthrough: return "fallthrough";
        case Tk::False: return "false";
        case Tk::For: return "for";
        case Tk::ForReverse: return "for~";
        case Tk::Goto: return "goto";
        case Tk::If: return "if";
        case Tk::In: return "in";
        case Tk::Init: return "init";
        case Tk::Int: return "int";
        case Tk::Import: return "import";
        case Tk::Is: return "is";
        case Tk::Land: return "land";
        case Tk::Lor: return "lor";
        case Tk::Match: return "match";
        case Tk::NoReturn: return "noreturn";
        case Tk::Not: return "not";
        case Tk::Or: return "or";
        case Tk::Pragma: return "pragma";
        case Tk::Proc: return "proc";
        case Tk::Return: return "return";
        case Tk::Static: return "static";
        case Tk::Struct: return "struct";
        case Tk::Then: return "then";
        case Tk::True: return "true";
        case Tk::Try: return "try";
        case Tk::Type: return "type";
        case Tk::Typeof: return "typeof";
        case Tk::Unreachable: return "unreachable";
        case Tk::Val: return "val";
        case Tk::Var: return "var";
        case Tk::Variant: return "variant";
        case Tk::Void: return "void";
        case Tk::While: return "while";
        case Tk::With: return "with";
        case Tk::Xor: return "xor";
        case Tk::CChar8T: return "__srcc_ffi_char8";
        case Tk::CChar16T: return "__srcc_ffi_char16";
        case Tk::CChar32T: return "__srcc_ffi_char32";
        case Tk::CChar: return "__srcc_ffi_char";
        case Tk::CInt: return "__srcc_ffi_int";
        case Tk::CLong: return "__srcc_ffi_long";
        case Tk::CLongDouble: return "__srcc_ffi_longdouble";
        case Tk::CLongLong: return "__srcc_ffi_longlong";
        case Tk::Continue: return "continue";
        case Tk::CShort: return "__srcc_ffi_short";
        case Tk::CSizeT: return "__srcc_ffi_size";
        case Tk::CWCharT: return "__srcc_ffi_wchar";

        case Tk::Semicolon: return ";";
        case Tk::Colon: return ":";
        case Tk::ColonColon: return "::";
        case Tk::Comma: return ",";
        case Tk::LParen: return "(";
        case Tk::RParen: return ")";
        case Tk::LBrack: return "[";
        case Tk::RBrack: return "]";
        case Tk::LBrace: return "{";
        case Tk::RBrace: return "}";
        case Tk::Ellipsis: return "...";
        case Tk::Dot: return ".";
        case Tk::LArrow: return "<-";
        case Tk::RArrow: return "->";
        case Tk::RDblArrow: return "=>";
        case Tk::Question: return "?";
        case Tk::Plus: return "+";
        case Tk::Minus: return "-";
        case Tk::Star: return "*";
        case Tk::Slash: return "/";
        case Tk::Percent: return "%";
        case Tk::Caret: return "^";
        case Tk::Ampersand: return "&";
        case Tk::VBar: return "|";
        case Tk::Tilde: return "~";
        case Tk::Bang: return "!";
        case Tk::Assign: return "=";
        case Tk::DotDot: return "..";
        case Tk::DotDotLess: return "..<";
        case Tk::DotDotEq: return "..=";
        case Tk::MinusMinus: return "--";
        case Tk::PlusPlus: return "++";
        case Tk::StarStar: return "**";
        case Tk::Lt: return "<";
        case Tk::Le: return "<=";
        case Tk::Gt: return ">";
        case Tk::Ge: return ">=";
        case Tk::EqEq: return "==";
        case Tk::Neq: return "!=";
        case Tk::PlusEq: return "+=";
        case Tk::MinusEq: return "-=";
        case Tk::StarEq: return "*=";
        case Tk::SlashEq: return "/=";
        case Tk::PercentEq: return "%=";
        case Tk::ShiftLeft: return "<<";
        case Tk::ShiftRight: return ">>";
        case Tk::ShiftRightLogical: return ">>>";
        case Tk::ShiftLeftEq: return "<<=";
        case Tk::ShiftRightEq: return ">>=";
        case Tk::ShiftRightLogicalEq: return ">>>=";
        case Tk::StarStarEq: return "**=";
    }

    Unreachable();
}

class Lexer;
class TokenStream {
    using Storage = std::deque<Token>;
    Storage token_storage;
    llvm::UniqueStringSaver saver;

    /// Lexer may have to coalesce multiple tokens.
    friend Lexer;

public:
    using iterator = Storage::iterator;

    /// Construct a token stream.
    TokenStream(llvm::BumpPtrAllocator &alloc) : saver(alloc) {}

    /// Allocate a token.
    ///
    /// This returns a stable pointer that may be retained.
    auto allocate() -> Token* {
        return &token_storage.emplace_back();
    }

    /// Get the last token.
    auto back() -> Token* { return &token_storage.back(); }

    /// Get an iterator to the first token.
    auto begin() { return token_storage.begin(); }

    /// Get an iterator to the last token.
    auto end() { return token_storage.end(); }

    /// Save a string in the stream.
    ///
    /// \param str The string to store.
    /// \return A stable reference to the stored string.
    auto save(StringRef str) -> String {
        return String{saver.save(str)};
    }

    /// Get the number of tokens in the stream.
    auto size() const -> usz { return token_storage.size(); }

    /// Access a token by index.
    auto operator[](usz idx) -> Token& {
        Assert(idx < token_storage.size(), "Token index out of bounds");
        return token_storage[idx];
    }
};
} // namespace src

#endif // SOURCE_TOKEN_HH
