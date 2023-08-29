#ifndef SOURCE_INCLUDE_FRONTEND_PARSER_HH
#define SOURCE_INCLUDE_FRONTEND_PARSER_HH

#include <frontend/lexer.hh>
#include <result.hh>

class Parser : Lexer {
    Module* mod;
    std::unique_ptr<Module> mod_ptr{};
    std::vector<Scope*> scope_stack;

    /// Current function.
    FunctionDecl* curr_func{};

    /// Current declaration context.
    ExprList* decl_context{};

    readonly(Scope*, global_scope, return scope_stack[0]);
    readonly(Scope*, curr_scope, return scope_stack.back());
    readonly(Scope*, top_level_scope, return scope_stack[1]);
    readonly(Location, curr_loc, return tok.location);

public:
    /// Parse a file into a module.
    static auto Parse(Context* ctx, File& f) -> std::unique_ptr<Module>;

private:
    explicit Parser(Context* ctx, File& f);

    /// Parser functions.
    void ParseExpressions(ExprList& into);
    void ParsePreamble();
    void ParseFile();

    [[nodiscard]] auto ParseBlockExpr() -> Result<BlockExpr*>;
    [[nodiscard]] auto ParseExpr(int operator_precedence = 0) -> Result<Expr*>;

    static bool MayStartAnExpression(Tk k);
    bool AtStartOfExpression() { return MayStartAnExpression(tok.type); }

    /// Parser primitives.
    bool At(std::same_as<Tk> auto... tks) { return ((tok.type == tks) or ...); }
    bool Consume(Tk t) {
        if (At(t)) {
            Next();
            return true;
        }
        return false;
    }

    /// Issue an error.
    template <typename... Args>
    Diag Error(Location loc, fmt::format_string<Args...> fmt, Args&&... args) {
        return Diag::Error(ctx, loc, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    Diag Error(fmt::format_string<Args...> fmt, Args&&... args) {
        return Diag::Error(ctx, curr_loc, fmt, std::forward<Args>(args)...);
    }

    void Synchronise();
};

#endif // SOURCE_INCLUDE_FRONTEND_PARSER_HH
