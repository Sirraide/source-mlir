#ifndef SOURCE_FRONTEND_PARSER_HH
#define SOURCE_FRONTEND_PARSER_HH

#include <source/Frontend/AST.hh>
#include <source/Frontend/Lexer.hh>
#include <source/Support/Result.hh>

namespace src {
class Parser : Lexer {
    template <typename T>
    using SVI = SmallVectorImpl<T>;

    Module* mod{};
    std::unique_ptr<Module> mod_ptr{};
    std::vector<Scope*> scope_stack;

    /// Current function.
    ProcDecl* curr_func{};

    readonly(Scope*, global_scope, return scope_stack[0]);
    readonly(Scope*, top_level_scope, return scope_stack[1]);
    readonly(Scope*, curr_scope, return scope_stack.back());
    readonly(Location, curr_loc, return tok.location);

public:
    /// Parse a file into a module.
    static auto Parse(Context& ctx, File& f) -> std::unique_ptr<Module>;

private:
    friend class MatchedDelimiterTracker;

    /// RAII wrapper that pushes and pops a scope.
    struct ScopeRAII {
        Parser& p;
        Scope* scope;

        ScopeRAII(Parser* parser)
            : p(*parser),
              scope(new(p.mod) Scope(p.curr_scope, p.mod)) { p.scope_stack.push_back(scope); }

        ~ScopeRAII() { pop(); }

        ScopeRAII(const ScopeRAII&) = delete;
        ScopeRAII& operator=(const ScopeRAII&) = delete;

        ScopeRAII(ScopeRAII&& o) : p(o.p), scope(std::exchange(o.scope, nullptr)) {}
        ScopeRAII& operator=(ScopeRAII&& o) {
            if (this == std::addressof(o)) return *this;
            pop();
            scope = std::exchange(o.scope, nullptr);
            return *this;
        }

        void pop() {
            if (scope) p.scope_stack.pop_back();
            release();
        }

        void release() { scope = nullptr; }
    };

    /// Helper that skips a balanced set of delimiters.
    class MatchedDelimiterTracker {
        Parser& P;
        int parens{};
        int bracks{};
        int braces{};

        int skip_until(int lookahead_index) {
            while (parens or bracks or braces) {
                /// Stop at EOF.
                auto& tok = P.LookAhead(usz(lookahead_index++));
                if (P.Is(tok, Tk::Eof)) return lookahead_index;

                /// Handle parens/brackets/braces.
                switch (tok.type) {
                    case Tk::RParen: parens = std::max(0, parens - 1); break;
                    case Tk::RBrack: bracks = std::max(0, bracks - 1); break;
                    case Tk::RBrace: braces = std::max(0, braces - 1); break;
                    case Tk::LParen: parens++; break;
                    case Tk::LBrack: bracks++; break;
                    case Tk::LBrace: braces++; break;
                    default: break;
                }
            }

            /// Return index of first token after the matching delimiter.
            return lookahead_index;
        }

    public:
        MatchedDelimiterTracker(Parser* P) : P(*P) {}

        /// Skip to the next matching '}'. Returns the index of the
        /// first token after the matching delimiter.
        int skip_braces(int lookahead_index) {
            braces++;
            return skip_until(lookahead_index);
        }

        /// Skip to the next matching ']'. Returns the index of the
        /// first token after the matching delimiter.
        int skip_bracks(int lookahead_index) {
            bracks++;
            return skip_until(lookahead_index);
        }

        /// Skip to the next matching ')'. Returns the index of the
        /// first token after the matching delimiter.
        int skip_parens(int lookahead_index) {
            parens++;
            return skip_until(lookahead_index);
        }
    };

    struct Signature {
        Expr* type;
        std::string name;
        SmallVector<ParamDecl*> param_decls;
        Location loc{};
        bool is_extern{};
        bool is_nomangle{};
    };

    static constexpr int NullPrecedence = 0;

    explicit Parser(Context* ctx, File& f);

    /// Parser primitives.
    bool At(std::same_as<Tk> auto... tks) { return Is(tok, tks...); }

    bool Consume(std::same_as<Tk> auto... tks) {
        if (At(tks...)) {
            Next();
            return true;
        }
        return false;
    }

    bool Is(const Token& t, std::same_as<Tk> auto... tks) { return ((t.type == tks) or ...); }

    /// Issue an error.
    template <typename... Args>
    Diag Error(Location loc, fmt::format_string<Args...> fmt, Args&&... args) {
        return Diag::Error(ctx, loc, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    Diag Error(fmt::format_string<Args...> fmt, Args&&... args) {
        return Diag::Error(ctx, curr_loc, fmt, std::forward<Args>(args)...);
    }

    /// Parser functions.
    auto ParseAssertion() -> Result<Expr*>;
    auto ParseBlock() -> Result<BlockExpr*>;
    auto ParseExpr(int curr_prec = 0) -> Result<Expr*>;
    auto ParseExprs(Tk until, SmallVector<Expr*>& into) -> Result<void>;
    void ParseFile();
    auto ParseIf() -> Result<Expr*>;
    auto ParseParamDeclList(SVI<ParamDecl*>& param_decls, SVI<Expr*>& param_types) -> Location;
    auto ParseProc() -> Result<Expr*>;
    auto ParseSignature() -> Signature;
    auto ParseType() -> Result<Expr*>;

    /// Synchronise in case of errors.
    void Synchronise(Tk token = Tk::Semicolon, std::same_as<Tk> auto... tks) {
        while (not At(Tk::Eof, token, tks...)) Next();
    }
};

} // namespace src
#endif // SOURCE_FRONTEND_PARSER_HH
