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
    Nil,
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
    std::string text{};

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

using TokenStream = std::deque<Token>;
} // namespace src

#endif // SOURCE_TOKEN_HH
