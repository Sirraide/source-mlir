#ifndef SOURCE_INCLUDE_FRONTEND_PARSER_HH
#define SOURCE_INCLUDE_FRONTEND_PARSER_HH

#include <source/Frontend/Lexer.hh>
#include <source/Support/Result.hh>

namespace src {

class Parser : Lexer {
    Module* mod;
    std::unique_ptr<Module> mod_ptr{};
    std::vector<Scope*> scope_stack;

    /// Current function.
    FunctionDecl* curr_func{};

    /// Static assert context.
    ///
    /// Static assertions in templates should only be evaluated when the
    /// template is instantiated, so we store.
    SmallVectorImpl<Expr*>* static_assertions{};

    readonly(Scope*, global_scope, return scope_stack[0]);
    readonly(Scope*, top_level_scope, return scope_stack[1]);
    readonly(Scope*, curr_scope, return scope_stack.back());
    readonly(Location, curr_loc, return tok.location);

public:
    /// Parse a file into a module.
    static auto Parse(Context* ctx, File& f) -> std::unique_ptr<Module>;

private:
    class ScopeRAII {
        Parser& p;
        property_r(Scope*, scope);


    public:
        ScopeRAII(Parser* parser)
            : p(*parser),
              scope_field(new Scope(p.curr_scope, p.mod)) { p.scope_stack.push_back(scope); }

        ~ScopeRAII() { pop(); }

        ScopeRAII(const ScopeRAII&) = delete;
        ScopeRAII& operator=(const ScopeRAII&) = delete;

        ScopeRAII(ScopeRAII&& o) : p(o.p), scope_field(std::exchange(o.scope, nullptr)) {}
        ScopeRAII& operator=(ScopeRAII&& o) {
            if (this == std::addressof(o)) return *this;
            pop();
            scope_field = std::exchange(o.scope, nullptr);
            return *this;
        }

        void pop() {
            if (scope) p.scope_stack.pop_back();
            release();
        }

        void release() { scope_field = nullptr; }
    };

    static constexpr int NullPrecedence = 0;

    explicit Parser(Context* ctx, File& f);

    template <bool parsing_control_expr_of_match>
    auto ParseExprImpl(int operator_precedence) -> Result<Expr*>;

    /// Parser functions.
    auto ParseAssertExpr(bool is_static) -> Result<AssertExpr*>;
    auto ParseBlockExpr() -> Result<BlockExpr*>;
    auto ParseDeclBase() -> Result<Decl*>;
    auto ParseExpr(int operator_precedence = NullPrecedence) -> Result<Expr*>;
    auto ParseExprInNewScope() -> Result<Expr*>;
    void ParseExpressions(ExprList& into);
    void ParseFile();
    auto ParseForExpr(bool is_static) -> Result<Expr*>;
    auto ParseIfExpr(bool is_static) -> Result<IfExpr*>;
    auto ParseInlineAsm() -> Result<Expr*>;
    auto ParseMatchExpr(bool is_static) -> Result<MatchExpr*>;
    void ParsePreamble();
    auto ParseWhileExpr() -> Result<WhileExpr*>;
    auto ParseWithExpr() -> Result<WithExpr*>;

    /// Check if we’re at the start of an expression.
    bool AtStartOfExpression();

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

} // namespace src
#endif // SOURCE_INCLUDE_FRONTEND_PARSER_HH
