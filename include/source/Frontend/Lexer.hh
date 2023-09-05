#ifndef SOURCE_INCLUDE_FRONTEND_LEXER_HH
#define SOURCE_INCLUDE_FRONTEND_LEXER_HH

#include <source/Core.hh>

namespace src {

/// ===========================================================================
///  Tokens.
/// ===========================================================================
enum struct Tk {
    Invalid,
    Eof,
    Identifier,
    Integer,
    StringLiteral,
    MacroParameter,

    /// Keywords.
    Module,
    Export,
    Import,
    Pragma,
    Assert,
    Asm,
    If,
    Then,
    Elif,
    Else,
    Match,
    While,
    Do,
    For,
    In,
    With,
    Try,
    Return,
    Defer,
    Break,
    Continue,
    Fallthrough,
    Unreachable,
    Variant,
    Extern,
    Static,
    Is,
    As,
    AsBang,
    Not,
    And,
    Or,
    Xor,
    True,
    False,
    Null,
    Proc,
    Var,
    Val,
    Enum,
    Struct,
    Init,
    Type,
    Typeof,
    NoReturn,
    Bool,
    Void,
    I8,
    I16,
    I32,
    I64,
    Int,
    F32,
    F64,
    CChar,
    CChar8T,
    CChar16T,
    CChar32T,
    CWCharT,
    CShort,
    CInt,
    CLong,
    CLongLong,
    CLongDouble,
    CSizeT,
    StringKw,

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
    AmpersandEq,
    VBarEq,
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
    isz integer{};

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
auto Spelling(Tk t) -> std::string_view;

/// A lexer that reads a source file and provides tokens from it.
class Lexer {
    struct Macro {
        std::string name{};
        Location location{};
        SmallVector<Token> parameter_list{};
        SmallVector<Token> expansion{};
    };

    class MacroExpansion {
        Macro* m{};
        decltype(m->expansion.begin()) it{};
        StringMap<Token> bound_parameters{};

    public:
        MacroExpansion(Lexer& l, Macro& m, StringMap<Token> args, Location loc)
            : m(&m), it(m.expansion.begin()), bound_parameters(std::move(args)) {}

        /// Check if the macro is done expanding.
        bool done() const { return it == m->expansion.end(); }

        /// Get the next token from the expansion.
        auto operator++() -> Token;
    };

protected:
    /// Source context.
    Context* ctx;

private:
    /// The file being lexed.
    File& f;

protected:
    /// The current token.
    Token tok{};

    /// The last character lexer.
    char lastc = ' ';

private:
    /// Current position in the source code.
    const char* curr{};
    const char* end{};

    /// Lookahead tokens.
    std::deque<Token> lookahead_tokens;

    /// Pragma handler.
    std::function<bool()> pragma_handler;

    /// Macro definitions.
    std::deque<Macro> macro_definitions;
    StringMap<Macro*> macro_definitions_by_name;

    /// Macro expansion stack.
    std::vector<MacroExpansion> macro_expansion_stack;

    /// Whether we’re currently looking ahead in the token stream.
    bool looking_ahead = false;

protected:
    /// Disable special handling for tokens. This is used for lexing __id.
    bool raw_mode = false;

private:
    /// Whether we’re reading a macro definition.
    bool in_macro_definition = false;

public:
    /// Construct a lexer to lex a file.
    explicit Lexer(Context* ctx, File& f);

    /// Copying/Moving is disallowed.
    Lexer(const Lexer&) = delete;
    Lexer(Lexer&&) = delete;
    Lexer& operator=(const Lexer&) = delete;
    Lexer& operator=(Lexer&&) = delete;

    auto CurrLoc() const -> Location;
    auto CurrOffs() const -> u32;

protected:
    /// Look ahead in the token list.
    ///
    /// Lookahead tokens are 1-based. LookAhead(0) returns the
    /// current token.
    auto LookAhead(usz n) -> Token&;

    /// Read the next token.
    void Next();

    /// Read the next character.
    void NextChar();

private:
    auto AllocateMacroDefinition(
        std::string name,
        Location location,
        SmallVector<Token>&& expansion = {}
    ) -> Macro&;

    /// Issue a diagnostic.
    template <typename... arguments>
    Diag Error(Location l, fmt::format_string<arguments...> fmt, arguments&&... args) {
        tok.type = Tk::Invalid;
        return Diag::Error(ctx, l, fmt, std::forward<arguments>(args)...);
    }

    void LexEscapedId();
    void LexIdentifier();
    void LexMacroDefinition();
    void LexMacroExpansion(Macro* m);
    void LexNumber();
    void LexString(char delim);

    void SetEnd();

    void SkipLine();
    void SkipWhitespace();
};

} // namespace src

/// Token formatter.
template <>
struct fmt::formatter<src::Token> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const src::Token& val, FormatContext& ctx) const -> decltype(ctx.out()) {
        return format_to(ctx.out(), "{}", Spelling(val.type));
    }
};


#endif // SOURCE_INCLUDE_FRONTEND_LEXER_HH
