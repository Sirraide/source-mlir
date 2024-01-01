#ifndef SOURCE_INCLUDE_FRONTEND_LEXER_HH
#define SOURCE_INCLUDE_FRONTEND_LEXER_HH

#include <source/Core.hh>
#include <source/Frontend/Token.hh>

namespace src {
/// A lexer that reads a source file and provides tokens from it.
class Lexer {
    struct Macro {
        StringRef name{};
        Location location{};
        SmallVector<Token> parameter_list{};
        SmallVector<Token> expansion{};
    };

    class MacroExpansion {
        Macro* m{};
        decltype(m->expansion.begin()) it{};
        StringMap<Token> bound_parameters{};

    public:
        MacroExpansion(Lexer&, Macro& m, StringMap<Token> args, Location)
            : m(&m), it(m.expansion.begin()), bound_parameters(std::move(args)) {}

        /// Check if the macro is done expanding.
        bool done() const { return it == m->expansion.end(); }

        /// Get the next token from the expansion.
        auto operator++() -> Token;
    };

    /// Source context.
    Context* ctx;

    /// The file being lexed.
    File& f;

    /// The last character lexed.
    char lastc = ' ';

    /// Tokens that we’re lexing.
    TokenStream& tokens;

    /// Get the current token.
    readonly(Token&, tok, return *tokens.back());

    /// Get the previous token, if there is one.
    readonly(Token*, prev, return tokens.size() > 1 ? &tokens[tokens.size() - 2] : nullptr);

    /// Current position in the source code.
    const char* curr{};
    const char* end{};

    /// Macro definitions.
    std::deque<Macro> macro_definitions;
    StringMap<Macro*> macro_definitions_by_name;

    /// Macro expansion stack.
    std::vector<MacroExpansion> macro_expansion_stack;

    /// Disable special handling for tokens. This is used for lexing __id.
    bool raw_mode = false;

    /// Whether we’re reading a macro definition.
    bool in_macro_definition = false;

public:
    /// Read all tokens in a file.
    static void LexEntireFile(Context* ctx, TokenStream& into, File&);

private:
    /// Construct a lexer to lex a file.
    explicit Lexer(Context* ctx, TokenStream& into, File& f);

    /// Copying/Moving is disallowed.
    Lexer(const Lexer&) = delete;
    Lexer(Lexer&&) = delete;
    Lexer& operator=(const Lexer&) = delete;
    Lexer& operator=(Lexer&&) = delete;

    auto AllocateMacroDefinition(
        StringRef name,
        Location location,
        SmallVector<Token>&& expansion = {}
    ) -> Macro&;

    auto CurrLoc() const -> Location;
    auto CurrOffs() const -> u32;

    /// Issue a diagnostic.
    template <typename... arguments>
    Diag Error(Location l, fmt::format_string<arguments...> fmt, arguments&&... args) {
        tok.type = Tk::Invalid;
        return Diag::Error(ctx, l, fmt, std::forward<arguments>(args)...);
    }

    void Next();
    void NextChar();
    void NextImpl();
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
