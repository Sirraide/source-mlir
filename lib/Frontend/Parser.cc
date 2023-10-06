#include <source/Frontend/AST.hh>
#include <source/Frontend/Parser.hh>

#define bind SRC_BIND Parser::

src::Parser::Parser(Context* ctx, File& f) : Lexer(ctx, f) {}

/// ===========================================================================
///  Helpers
/// ===========================================================================
namespace src {
namespace {
constexpr int BinaryOrPostfixPrecedence(Tk t) {
    switch (t) {
        case Tk::Dot:
            return 10'000;

        case Tk::LParen:
            return 1'000;

            /// Prefix operator precedence: 900.

        case Tk::StarStar:
            return 100;

        case Tk::Star:
        case Tk::Slash:
        case Tk::Percent:
            return 95;

        case Tk::Plus:
        case Tk::Minus:
            return 90;

        /// Shifts have higher precedence than logical/bitwise
        /// operators so e.g.  `a and 1 << 3` works properly.
        case Tk::ShiftLeft:
        case Tk::ShiftRight:
        case Tk::ShiftRightLogical:
            return 85;

        case Tk::Lt:
        case Tk::Gt:
        case Tk::Le:
        case Tk::Ge:
            return 80;

        case Tk::EqEq:
        case Tk::Neq:
            return 75;

        case Tk::And:
        case Tk::Or:
        case Tk::Xor:
            return 70;

        /// Naked invoke precedence is very low.
        case Tk::RParen:
            return 10;

        /// Assignment has the lowest precedence.
        case Tk::Assign:
        case Tk::RDblArrow:
            return 1;

        default:
            return -1;
    }
}

constexpr bool IsRightAssociative(Tk t) {
    switch (t) {
        case Tk::StarStar:
        case Tk::Assign:
        case Tk::RDblArrow:
            return true;

        default:
            return false;
    }
}

constexpr bool IsPostfix(Tk t) {
    switch (t) {
        default:
            return false;
    }
}

constexpr bool MayStartAnExpression(Tk k) {
    switch (k) {
        case Tk::Assert:
        case Tk::False:
        case Tk::Identifier:
        case Tk::If:
        case Tk::Int:
        case Tk::Integer:
        case Tk::IntegerType:
        case Tk::LBrace:
        case Tk::Proc:
        case Tk::Star:
        case Tk::StarStar:
        case Tk::StringLiteral:
        case Tk::True:
            return true;

        default:
            return false;
    }
}

constexpr inline int PrefixPrecedence = 900;
constexpr inline int InvokePrecedence = BinaryOrPostfixPrecedence(Tk::LParen);
constexpr inline int NakedInvokePrecedence = BinaryOrPostfixPrecedence(Tk::RParen);

} // namespace
} // namespace src

/// ===========================================================================
///  Parse Functions
/// ===========================================================================
auto src::Parser::Parse(Context& ctx, File& f) -> std::unique_ptr<Module> {
    Parser p{&ctx, f};
    p.ParseFile();
    return ctx.has_error() ? nullptr : std::move(p.mod_ptr);
}

/// <expr-assert> ::= ASSERT <expr> [ ","  <expr> ]
auto src::Parser::ParseAssertion() -> Result<Expr*> {
    auto start = curr_loc;
    Assert(Consume(Tk::Assert));

    /// Parse condition.
    auto cond = ParseExpr();
    auto mess = Consume(Tk::Comma) ? ParseExpr() : Result<Expr*>::Null();
    if (IsError(cond, mess)) return Diag();
    return new (mod) AssertExpr(*cond, *mess, {start, *mess ? mess->location : cond->location});
}

/// <expr-block> ::= "{" { <expr> ";" } "}"
auto src::Parser::ParseBlock() -> Result<BlockExpr*> {
    ScopeRAII sc{this};
    auto loc = curr_loc;
    Assert(Consume(Tk::LBrace));

    SmallVector<Expr*> exprs;
    if (not ParseExprs(Tk::RBrace, exprs)) return Diag();

    auto block = new (mod) BlockExpr(sc.scope, std::move(exprs), {loc, tok.location});
    if (not Consume(Tk::RBrace)) Error("Expected '}}'");
    return block;
}

/// <expr-decl-ref> ::= IDENTIFIER
/// <expr-access>   ::= <expr> "." IDENTIFIER
/// <expr-literal>  ::= INTEGER_LITERAL | STRING_LITERAL
/// <expr-invoke>   ::= <expr> [ "(" ] <expr> { "," <expr> } [ ")" ]
auto src::Parser::ParseExpr(int curr_prec) -> Result<Expr*> {
    auto lhs = Result<Expr*>::Null();

    /// See below.
    const auto start_token = tok.type;
    switch (start_token) {
        default: return Error("Expected expression");

        case Tk::IntegerType:
        case Tk::Int:
            lhs = ParseType();
            break;

        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        case Tk::Identifier:
            lhs = new (mod) DeclRefExpr(tok.text, curr_scope, tok.location);
            Next();
            break;

        case Tk::Integer:
            lhs = new (mod) IntLitExpr(tok.integer, tok.location);
            Next();
            break;

        case Tk::True:
        case Tk::False:
            lhs = new (mod) BoolLitExpr(tok.type == Tk::True, tok.location);
            Next();
            break;

        case Tk::StringLiteral:
            lhs = new (mod) StrLitExpr(mod->strtab.intern(tok.text), tok.location);
            Next();
            break;

        case Tk::Assert:
            lhs = ParseAssertion();
            break;

        case Tk::Proc:
            lhs = ParseProc();
            break;

        case Tk::If:
            lhs = ParseIf();
            break;

        case Tk::Star:
        case Tk::StarStar: {
            auto start = Next();
            auto operand = ParseExpr(PrefixPrecedence);
            if (IsError(operand)) return operand.diag;
            if (start_token == Tk::StarStar) {
                lhs = new (mod) UnaryPrefixExpr(Tk::Star, *operand, {start.contract_right(1), operand->location});
                lhs = new (mod) UnaryPrefixExpr(Tk::Star, *lhs, {start, operand->location});
            } else {
                lhs = new (mod) UnaryPrefixExpr(start_token, *operand, {start, operand->location});
            }
        } break;
    }

    /// Stop here if there was an error.
    if (lhs.is_diag) return lhs.diag;

    /// Make sure that the rest of the parser knows that this
    /// token can start an expression. This is a sanity check
    /// because I keep forgetting to add a token to that function
    /// when adding a new expression.
    Assert(
        MayStartAnExpression(start_token),
        "Add '{}' to MayStartAnExpression()",
        Spelling(start_token)
    );

    /// Parse anything that looks like a binary operator or
    /// postfix operator.
    while (not IsError(lhs)) {
        /// Some operators need special parsing.
        switch (tok.type) {
            /// If the token could be the start of an expression, but not a block,
            /// then this is an invoke expression, which means that this is either
            /// a declaration, a function call, a struct literal, or a template
            /// instantiation.
            ///
            /// We specifically disallow blocks in this position so that, e.g.
            /// in `if a {`, `a {` does not get parsed as an invoke expression.
            default: {
                if (
                    not MayStartAnExpression(tok.type) or
                    NakedInvokePrecedence < curr_prec or /// Right-associative
                    At(Tk::LBrace) or
                    tok.type == Tk::Star or /// Prefer binary '*' over unary '*' parse.
                    tok.type == Tk::StarStar or
                    IsPostfix(tok.type)
                ) break;

                /// Parse the arguments of the invoke expression.
                SmallVector<Expr*> args;
                do {
                    auto arg = ParseExpr(NakedInvokePrecedence);
                    if (not IsError(arg)) args.push_back(*arg);
                } while (Consume(Tk::Comma));

                /// An assignment expression after a naked invocation is bound to it
                /// in case it turns out to be a declaration.
                auto init = Result<Expr*>::Null();
                if (Consume(Tk::Assign)) {
                    auto expr = ParseExpr();
                    if (IsError(expr)) return expr.diag;
                    init = *expr;
                }

                Location loc = {lhs->location, not args.empty() ? args.back()->location : Location{}};
                lhs = new (mod) InvokeExpr(
                    *lhs,
                    std::move(args),
                    true,
                    *init,
                    loc
                );
                continue;
            }

            /// Delimited invoke.
            case Tk::LParen: {
                /// Yeet '('
                if (InvokePrecedence < curr_prec) return lhs;
                Next();

                /// Parse args.
                SmallVector<Expr*> args;
                while (not At(Tk::RParen, Tk::Eof)) {
                    auto arg = ParseExpr(InvokePrecedence);
                    if (not IsError(arg)) args.push_back(*arg);
                    if (not Consume(Tk::Comma)) break;
                }

                /// Create the invoke expression.
                lhs = new (mod) InvokeExpr(
                    *lhs,
                    std::move(args),
                    false,
                    {},
                    {lhs->location, curr_loc}
                );

                /// Yeet ')'
                if (not Consume(Tk::RParen)) Error("Expected ')'");
                continue;
            }

            /// Member access.
            case Tk::Dot: {
                Next();
                if (not At(Tk::Identifier)) Error("Expected identifier");
                lhs = new (mod) MemberAccessExpr(*lhs, tok.text, {lhs->location, Next()});
                continue;
            }
        }

        /// Check if we should keep parsing.
        auto prec = BinaryOrPostfixPrecedence(tok.type);
        if (prec < curr_prec or (prec == curr_prec and not IsRightAssociative(tok.type))) break;

        /// Save operator and parse rhs.
        auto op = tok.type;
        Next();
        auto rhs = ParseExpr(prec);
        if (IsError(rhs)) return rhs.diag;
        lhs = new (mod) BinaryExpr(op, *lhs, *rhs, {lhs->location, rhs->location});
    }

    /// Done parsing.
    return lhs;
}

/// <exprs> ::= { [ <expr> ]  ";" }
auto src::Parser::ParseExprs(Tk until, SmallVector<Expr*>& into) -> Result<void> {
    while (not At(Tk::Eof, until)) {
        /// Yeet excess semicolons.
        while (Consume(Tk::Semicolon)) continue;

        /// Parse an expression.
        auto e = ParseExpr();
        if (IsError(e)) {
            Synchronise();
            Consume(Tk::Semicolon);
            continue;
        }

        into.push_back(*e);
        Consume(Tk::Semicolon);
    }

    return {};
}

/// <file> ::= { <expr> | ";" }
void src::Parser::ParseFile() {
    /// Parse preamble; this also creates the module.
    mod_ptr = std::make_unique<Module>(ctx, "");
    mod = mod_ptr.get();

    /// Set up scopes.
    scope_stack.push_back(mod->global_scope);
    scope_stack.emplace_back(new (mod) Scope{global_scope, mod});

    /// Parse expressions.
    std::ignore = ParseExprs(Tk::Eof, mod->top_level_func->body->exprs);
}

/// <expr-if> ::= IF <expr> <then> { ELIF <expr> <then> } [ ELSE <expr> ]
/// <then> ::= [ THEN ] <expr>
auto src::Parser::ParseIf() -> Result<Expr*> {
    auto start = curr_loc;
    Assert(Consume(Tk::If, Tk::Elif));

    /// Parse condition, elif clauses, and else clause.
    auto cond = ParseExpr();
    auto then = (Consume(Tk::Then), ParseExpr());
    auto else_ = Consume(Tk::Elif) ? ParseExpr()
               : Consume(Tk::Else) ? ParseExpr()
                                   : Result<Expr*>::Null();

    /// Create the expression.
    if (IsError(cond, then, else_)) return Diag();
    return new (mod) IfExpr(*cond, *then, *else_, {start, curr_loc});
}

/// <proc-args>  ::= "(" <param-decl> { "," <param-decl> } ")"
/// <param-decl> ::= <type> [ IDENTIFIER ]
auto src::Parser::ParseParamDeclList(
    SVI<ParamDecl*>& param_decls,
    SVI<Expr*>& param_types
) -> Location {
    auto loc = curr_loc;
    Assert(Consume(Tk::LParen));

    /// List may be empty.
    if (At(Tk::RParen)) return {loc, Next()};

    /// Parse param decls.
    do {
        auto param_type = ParseType();
        if (IsError(param_type)) {
            /// Skip to comma or ')'.
            Synchronise(Tk::Comma, Tk::RParen);
            continue;
        }

        /// Name is optional.
        std::string name;
        if (At(Tk::Identifier)) {
            name = tok.text;
            Next();
        }

        /// TODO: Default values go in their own scope.

        param_decls.push_back(new (mod) ParamDecl(
            std::move(name),
            *param_type,
            tok.location
        ));
        param_types.push_back(*param_type);
    } while (Consume(Tk::Comma));

    /// Parse closing paren.
    loc = {loc, curr_loc};
    if (not Consume(Tk::RParen)) Error("Expected ')'");
    return loc;
}

/// <proc-extern>    ::= PROC IDENTIFIER <proc-signature>
/// <proc-named>     ::= PROC IDENTIFIER <proc-signature> <proc-body>
/// <proc-body>      ::= <expr-block> | "=" <expr>
auto src::Parser::ParseProc() -> Result<Expr*> {
    /// Procedure signatures are rather complicated, so
    /// we’ll parse them separately.
    auto sig = ParseSignature();

    /// Parse the body if the procedure is not external.
    auto body = Result<BlockExpr*>::Null();
    if (not sig.is_extern) {
        if (Consume(Tk::Assign)) {
            ScopeRAII sc{this};
            if (auto res = ParseExpr(); res.is_diag) return Diag();
            else body = new (mod) BlockExpr(sc.scope, {*res}, {sig.loc, res->location});
        } else if (At(Tk::LBrace)) body = ParseBlock();
        else body = Error("Expected '=' or '{{' at start of procedure body");
    }

    /// Create the procedure.
    ///
    /// Note: Variable declarations are only added to a scope in Sema.
    if (IsError(body)) return body;
    auto proc = new (mod) ProcDecl(
        mod,
        std::move(sig.name),
        sig.type,
        std::move(sig.param_decls),
        *body,
        sig.is_extern ? Linkage::Imported : Linkage::Internal,
        sig.is_nomangle ? Mangling::None : Mangling::Source,
        *body ? Location{sig.loc, body->location} : sig.loc
    );

    /// Add it to the current scope.
    curr_scope->declare(proc->name, proc);
    return proc;
}

/// <proc-signature> ::= [ <proc-args> ] [ <proc-ret> ] { <proc-attrs> }
/// <proc-ret>       ::= "->" <type>
/// <proc-attrs>     ::= EXTERN | NOMANGLE
auto src::Parser::ParseSignature() -> Signature {
    Signature sig;
    sig.loc = curr_loc;
    Assert(Consume(Tk::Proc));

    /// We currently only have named procedures.
    if (not At(Tk::Identifier)) {
        Error("Expected identifier");
        sig.name = "<error>";
    } else {
        sig.name = tok.text;
        Next();
    }

    /// Parse the arguments if there are any. The argument
    /// list may be omitted altogether.
    SmallVector<Expr*> param_types;
    if (At(Tk::LParen)) ParseParamDeclList(sig.param_decls, param_types);

    /// Helper to parse attributes.
    const auto ParseAttr = [&](std::string_view attr_name, bool& flag) {
        if (tok.text == attr_name) {
            if (flag) Diag::Warning(
                ctx,
                curr_loc,
                "Duplicate '{}' attribute ignored",
                attr_name
            );

            Next();
            return flag = true;
        }

        return false;
    };

    /// Parse attributes.
    while (At(Tk::Identifier) and ( // clang-format off
        ParseAttr("nomangle", sig.is_nomangle) or
        ParseAttr("extern", sig.is_extern)
    )); // clang-format on

    /// Finally, parse the return type.
    Expr* ret_type = Type::Void;
    if (Consume(Tk::RArrow)) {
        auto res = ParseType();
        if (not IsError(res)) ret_type = *res;
    }

    /// Create the procedure type.
    sig.type = new (mod) ProcType(
        std::move(param_types),
        ret_type,
        sig.loc
    );
    return sig;
}

/// <type>           ::= <type-prim> | <type-qualified>
/// <type-prim>      ::= INTEGER_TYPE | INT
/// <type-qualified> ::= <type> { <type-qual> }
/// <type-qual>      ::= "&"
auto src::Parser::ParseType() -> Result<Expr*> {
    /// Parse base type.
    Expr* base_type = nullptr;
    switch (tok.type) {
        default:
            return Error("Expected type");

        case Tk::IntegerType:
            base_type = new (mod) IntType(tok.integer, tok.location);
            Next();
            break;

        case Tk::Int:
            base_type = BuiltinType::Int(mod, tok.location);
            Next();
            break;
    }

    /// Parse qualifiers.
    while (At(Tk::Ampersand)) {
        base_type = new (mod) ReferenceType(base_type, {base_type->location, curr_loc});
        Next();
    }

    return base_type;
}
