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
        case Tk::ColonColon:
            return 100'000;

        case Tk::Dot:
            return 10'000;

        case Tk::LBrack:
            return 5'000;

            /// InvokePrecedence = 1000;
            /// PrefixPrecedence = 900.

        case Tk::As:
        case Tk::AsBang:
            return 200;

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
        /// operators so e.g.  `a land 1 << 3` works properly.
        case Tk::ShiftLeft:
        case Tk::ShiftRight:
        case Tk::ShiftRightLogical:
            return 85;

        case Tk::Land:
        case Tk::Lor:
            return 82;

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
        /// NakedInvokePrecedence = 10;

        /// Assignment has the lowest precedence.
        case Tk::Assign:
        case Tk::RDblArrow:
        case Tk::PlusEq:
        case Tk::MinusEq:
        case Tk::StarEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq:
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
        case Tk::PlusEq:
        case Tk::MinusEq:
        case Tk::StarEq:
        case Tk::StarStarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq:
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
        case Tk::Alias:
        case Tk::Assert:
        case Tk::Bool:
        case Tk::Break:
        case Tk::Continue:
        case Tk::Defer:
        case Tk::Dot:
        case Tk::Export:
        case Tk::False:
        case Tk::For:
        case Tk::ForReverse:
        case Tk::Goto:
        case Tk::Identifier:
        case Tk::If:
        case Tk::Int:
        case Tk::Integer:
        case Tk::IntegerType:
        case Tk::LBrace:
        case Tk::LBrack:
        case Tk::LParen:
        case Tk::Nil:
        case Tk::NoReturn:
        case Tk::Not:
        case Tk::Proc:
        case Tk::Return:
        case Tk::Star:
        case Tk::StarStar:
        case Tk::StringLiteral:
        case Tk::Struct:
        case Tk::True:
        case Tk::Var:
        case Tk::Void:
        case Tk::While:
        case Tk::With:
            return true;

        default:
            return false;
    }
}

constexpr inline int InvokePrecedence = 1'000;
constexpr inline int PrefixPrecedence = 900;
constexpr inline int NakedInvokePrecedence = 10;

} // namespace
} // namespace src

/// ===========================================================================
///  Parse Functions
/// ===========================================================================
auto src::Parser::Parse(Context& ctx, File& f) -> Module* {
    Parser p{&ctx, f};
    p.ParseFile();
    return ctx.has_error() ? nullptr : p.mod;
}

/// <expr-alias> ::= ALIAS IDENTIFIER "=" <expr>
auto src::Parser::ParseAlias() -> Result<AliasExpr*> {
    auto start = curr_loc;
    Assert(Consume(Tk::Alias));

    /// Parse name.
    if (not At(Tk::Identifier)) return Error("Expected identifier");
    auto name = tok.text;
    Next();

    /// Parse value.
    if (not Consume(Tk::Assign)) Error("Expected '='");
    auto value = ParseExpr();
    if (IsError(value)) return value.diag;
    return new (mod) AliasExpr(std::move(name), *value, {start, value->location});
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
    ScopeRAII sc{this, curr_loc};
    Assert(Consume(Tk::LBrace));
    if (not ParseExprs(Tk::RBrace, sc.scope->exprs)) return Diag();
    sc.scope->location = {sc.scope->location, curr_loc};
    if (not Consume(Tk::RBrace)) Error("Expected '}}'");
    return sc.scope;
}

/// <decl> ::= <proc-named> | <proc-extern> | <param-decl> | <var-decl>
auto src::Parser::ParseDecl() -> Result<Decl*> {
    if (At(Tk::Proc)) return cast<Decl>(ParseProc());

    /// Parse decl type.
    auto ty = ParseType();
    if (IsError(ty)) return ty.diag;

    /// Parse name.
    if (not At(Tk::Identifier)) return Error("Expected identifier");
    auto name = tok.text;
    auto loc = Next();

    /// Create a variable declaration, but don’t add it to any scope.
    return new (mod) LocalDecl(
        curr_func,
        std::move(name),
        *ty,
        {},
        LocalKind::Variable,
        loc
    );
}

/// <expr-decl-ref>  ::= IDENTIFIER
/// <expr-access>    ::= [ <expr> ] "." IDENTIFIER
/// <expr-literal>   ::= INTEGER_LITERAL | STRING_LITERAL
/// <expr-invoke>    ::= <expr> [ "(" ] <expr> { "," <expr> } [ ")" ]
/// <expr-paren>     ::= "(" <expr> ")"
/// <expr-subscript> ::= <expr> "[" <expr> "]"
/// <expr-tuple>     ::= "(" { <expr> "," } [ <expr> ] ")"
/// <array-literal>  ::= "[" { <expr>  "," } [ <expr> ] "]"
auto src::Parser::ParseExpr(int curr_prec) -> Result<Expr*> {
    /// A ProcDecl must be wrapped in DeclRefExpr if it is not a full
    /// expression or not preceded by 'export'.
    const bool wrap_procedure_in_decl_ref = not full_expr_or_export;
    tempset full_expr_or_export = false;

    /// See below.
    const auto start_token = tok.type;

    /// Parse the LHS of a binary expression.
    auto lhs = Result<Expr*>::Null();
    switch (start_token) {
        default: return Error("Expected expression");

        case Tk::IntegerType:
        case Tk::Int:
        case Tk::Bool:
        case Tk::Void:
        case Tk::NoReturn:
        case Tk::Nil:
        case Tk::Var:
            lhs = ParseType() >> [&](Type t) { return static_cast<Expr*>(t); };
            break;

        case Tk::Semicolon:
            return new (mod) EmptyExpr(curr_loc);

        /// Struct type or decl.
        ///
        /// Note that a named struct decl cannot be used directly as a type,
        /// so we return here if this is one. This is so that we don’t have
        /// to put a semicolon after the closing brace of a struct without
        /// causing the struct to gobble up any name after it.
        case Tk::Struct:
            lhs = ParseStruct();
            if (lhs.is_value and not cast<StructType>(*lhs)->name.empty()) return lhs;
            break;

        case Tk::LBrace:
            lhs = ParseBlock();
            break;

        case Tk::Identifier:
            lhs = ParseIdentExpr();
            break;

        /// Member access.
        case Tk::Dot: {
            auto loc = Next();
            if (not At(Tk::Identifier, Tk::Init)) return Error("Expected identifier");
            lhs = new (mod) MemberAccessExpr(nullptr, tok.text, {loc, curr_loc});
            Next();
        } break;

        /// <expr-loop-ctrl> ::= ( BREAK | CONTINUE ) [ IDENTIFIER ]
        case Tk::Break:
        case Tk::Continue: {
            const bool is_continue = tok.type == Tk::Continue;
            auto start = Next();

            /// Parse optional label.
            std::string label;
            if (At(Tk::Identifier)) {
                label = tok.text;
                start = {start, tok.location};
                Next();
            }

            lhs = new (mod) LoopControlExpr(std::move(label), is_continue, {start, curr_loc});
        } break;

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

        case Tk::Alias:
            lhs = ParseAlias();
            break;

        case Tk::Assert:
            lhs = ParseAssertion();
            break;

        case Tk::Proc: {
            lhs = ParseProc();

            /// The backend doesn’t like ProcDecls that are referenced directly in expressions.
            if (wrap_procedure_in_decl_ref and lhs and isa<ProcDecl>(*lhs)) {
                auto dr = new (mod) DeclRefExpr(cast<ProcDecl>(*lhs)->name, curr_scope, lhs->location);
                dr->decl = *lhs;
                lhs = dr;
            }
        } break;

        case Tk::If:
            lhs = ParseIf();
            break;

        case Tk::While:
            lhs = ParseWhile();
            break;

        case Tk::With:
            lhs = ParseWith();
            break;

        case Tk::For:
        case Tk::ForReverse:
            lhs = ParseFor();
            break;

        /// <expr-export> ::= EXPORT <expr>
        case Tk::Export: {
            tempset full_expr_or_export = true;
            auto start = Next();
            lhs = ParseExpr();
            if (IsError(lhs)) return lhs.diag;
            lhs = new (mod) ExportExpr(*lhs, {start, lhs->location});
        } break;

        /// <expr-return> ::= RETURN [ <expr> ]
        case Tk::Return: {
            auto start = Next();
            auto value = Result<Expr*>::Null();
            if (MayStartAnExpression(tok.type)) {
                value = ParseExpr();
                if (IsError(value)) return value.diag;
            }

            lhs = new (mod) ReturnExpr(*value, {start, *value ? value->location : start});
        } break;

        /// <expr-jump> ::= GOTO IDENTIFIER
        case Tk::Goto: {
            auto start = Next();
            if (not At(Tk::Identifier)) return Error("Expected identifier");
            lhs = new (mod) GotoExpr(tok.text, {start, curr_loc});
            Next();
        } break;

        /// <expr-defer> ::= DEFER <implicit-block>
        case Tk::Defer: {
            auto start = Next();
            auto expr = ParseImplicitBlock();
            if (IsError(expr)) return expr.diag;
            lhs = new (mod) DeferExpr(*expr, {start, expr->location});
        } break;

        case Tk::Star:
        case Tk::StarStar:
        case Tk::Not: {
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

        /// Parenthesised expression or tuple.
        case Tk::LParen: {
            auto start = Next();
            auto expr = ParseExpr();
            if (IsError(expr)) return expr.diag;

            /// At least one comma means this is a tuple.
            if (Consume(Tk::Comma)) {
                SmallVector<Expr*> exprs{*expr};
                while (not At(Tk::RParen, Tk::Eof)) {
                    expr = ParseExpr();
                    if (IsError(expr)) return expr.diag;
                    exprs.push_back(*expr);
                    if (not Consume(Tk::Comma)) break;
                }

                lhs = new (mod) TupleExpr(std::move(exprs), {start, curr_loc});
            } else {
                lhs = new (mod) ParenExpr(*expr, {start, curr_loc});
            }

            if (not At(Tk::RParen)) return Error("Expected ')'");
            Next();
        } break;

        /// Array literal.
        case Tk::LBrack: {
            auto start = Next();
            SmallVector<Expr*> exprs;
            while (not At(Tk::RBrack, Tk::Eof)) {
                auto expr = ParseExpr();
                if (IsError(expr)) return expr.diag;
                exprs.push_back(*expr);
                if (not Consume(Tk::Comma)) break;
            }

            lhs = new (mod) ArrayLitExpr(std::move(exprs), {start, curr_loc});
            if (not Consume(Tk::RBrack)) Error("Expected ']'");
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

    /// Whether an expression is invokable w/o parens.
    static const auto Invokable = [](Expr* e) { // clang-format off
        /// TODO: Allow invoking invoke expressions so `deque int a;` works.
        return isa<
            TypeBase,
            DeclRefExpr,
            MemberAccessExpr,
            ScopeAccessExpr,
            ParenExpr,
            TupleExpr,
            SubscriptExpr
        >(e);
    }; // clang-format on

    /// Parse anything that looks like a binary operator or
    /// postfix operator.
    while (not IsError(lhs)) {
        /// Some operators need special parsing.
        switch (tok.type) {
            /// If the token could be the start of an expression, but not a block,
            /// then this is an invoke expression, which means that this is either
            /// a declaration, a function call, a struct literal, or a template
            /// instantiation.
            default: {
                /// The next thing is not an expression, so it can’t be an invoke.
                if (not MayStartAnExpression(tok.type)) break;

                /// Invoke is left-associative.
                if (NakedInvokePrecedence <= curr_prec) break;

                /// Only allow invoking certain kinds of expressions.
                if (not Invokable(*lhs)) break;

                /// We specifically disallow blocks in this position so that, e.g.
                /// in `if a {`, `a {` does not get parsed as an invoke expression.
                if (At(Tk::LBrace)) break;

                /// Don’t parse `a if` etc. as invoke expr. This is so we can chain
                /// constructs easier, such as `if a while ...`.
                if (
                    At(
                        Tk::Continue,
                        Tk::Break,
                        Tk::For,
                        Tk::ForReverse,
                        Tk::Goto,
                        Tk::If,
                        Tk::Match,
                        Tk::Return,
                        Tk::While,
                        Tk::With
                    )
                ) break;

                /// Prefer binary '*' over unary '*' parse.
                if (tok.type == Tk::Star or tok.type == Tk::StarStar) break;

                /// If we don’t check for postfix operators here, then postfix
                /// operators that are syntactically equivalent to prefix operators
                /// will be impossible to parse in this position.
                if (IsPostfix(tok.type)) break;

                /// Parse the arguments of the invoke expression.
                lhs = ParseNakedInvokeExpr(*lhs);
                continue;
            }

            /// Delimited invoke.
            case Tk::LParen: {
                /// Disallow iife for named procedure decls.
                if (isa<ProcDecl>(*lhs) and not cast<ProcDecl>(*lhs)->name.empty())
                    return lhs;

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

            /// Subscripting.
            case Tk::LBrack: {
                Next();
                auto e = ParseExpr();
                if (IsError(e)) return e.diag;
                lhs = new (mod) SubscriptExpr(*lhs, *e, {lhs->location, curr_loc});
                if (not Consume(Tk::RBrack)) Error("Expected ']'");
                continue;
            }

            /// Member access.
            case Tk::Dot: {
                /// If the token after the identifier is `","`, and we’re not already parsing an
                /// invocation, then we have the situation `<expr> "." IDENT ","`, which is likely
                /// supposed to be a naked invoke, i.e.`<expr> ("." b) "," `, instead as the former
                /// is nonsense.
                if (
                    Is(LookAhead(1), Tk::Identifier, Tk::Init) and
                    Is(LookAhead(2), Tk::Comma) and
                    curr_prec != NakedInvokePrecedence and
                    Invokable(*lhs)
                ) {
                    lhs = ParseNakedInvokeExpr(*lhs);
                    continue;
                }

                /// Regular member access.
                Next();
                if (not At(Tk::Identifier, Tk::Init)) Error("Expected identifier");
                lhs = new (mod) MemberAccessExpr(*lhs, tok.text, {lhs->location, curr_loc});
                Next();
                continue;
            }

            /// Scope access.
            case Tk::ColonColon: {
                Next();
                if (not At(Tk::Identifier)) Error("Expected identifier");
                lhs = new (mod) ScopeAccessExpr(*lhs, tok.text, {lhs->location, curr_loc});
                Next();
                continue;
            }

            /// Explicit cast.
            case Tk::As:
            case Tk::AsBang: {
                auto kind = tok.type == Tk::As ? CastKind::Soft : CastKind::Hard;
                Next();
                auto ty = ParseType();
                if (IsError(ty)) return ty.diag;
                lhs = new (mod) CastExpr(kind, *lhs, *ty, {lhs->location, ty.value()->location});
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

/// <exprs> ::= { <expr> | <pragma> | ";" }
auto src::Parser::ParseExprs(Tk until, SmallVector<Expr*>& into) -> Result<void> {
    while (not At(Tk::Eof, until)) {
        /// Yeet excess semicolons.
        while (Consume(Tk::Semicolon)) continue;

        /// Handle pragmas.
        /// FIXME: Include these in the AST or somewhere else for Source fidelity?
        if (At(Tk::Pragma)) {
            ParsePragma();
            continue;
        }

        /// Parse an expression.
        tempset full_expr_or_export = true;
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

/// <file> ::= <module-part> <exprs>
/// <module-part> ::= [ MODULE IDENTIFIER ";" ] { IMPORT <import-name> ";" }
/// <import-name> ::= IDENTIFIER [ "." "*" ] | <header-name>
/// <header-name> ::= "<" TOKENS ">"
void src::Parser::ParseFile() {
    /// Parse preamble; this also creates the module.
    if (At(Tk::Identifier) and tok.text == "module") {
        auto loc = Next();
        if (not At(Tk::Identifier)) Error("Expected identifier");
        auto module_name = tok.text;
        Next();
        if (not Consume(Tk::Semicolon)) Error("Expected ';'");

        mod = Module::Create(ctx, std::move(module_name), false, loc);
    } else {
        mod = Module::Create(ctx, "");
    }

    /// Parse imports.
    while (At(Tk::Identifier) and tok.text == "import") {
        auto start = Next();
        std::string name;
        bool is_header = false;

        /// C++ header.
        if (At(Tk::Lt)) {
            /// Need to lex manually here.
            raw_mode = true;
            is_header = true;
            while (lastc != '>' and lastc != 0) {
                name.push_back(lastc);
                NextChar();
            }

            if (lastc == 0) {
                Error("Expected '>'");
                Next();
                return;
            }

            /// Bring the lexer back into sync.
            raw_mode = false;
            NextChar();
        }

        /// Regular module name.
        else if (At(Tk::Identifier))
            name = tok.text;

        /// Nonsense.
        else {
            Error("Expected identifier");
            Synchronise();
            continue;
        }

        /// Skip duplicate imports.
        if (utils::contains(mod->imports, name, &ImportedModuleRef::linkage_name)) {
            Diag::Warning(ctx, start, "Duplicate import '{}' ignored", name);
            Synchronise();
            continue;
        }

        /// Check whether this is an open import.
        bool is_open = false;
        Next();
        if (At(Tk::Dot) and Is(LookAhead(1), Tk::Star)) {
            is_open = true;
            Next();
            Next();
        }

        /// Check if there is a logical name.
        auto logical_name = name;
        if (Consume(Tk::As)) {
            if (not At(Tk::Identifier)) Error("Expected identifier");
            logical_name = tok.text;
            Next();
        }

        /// Headers must either be open or renamed.
        Location loc = {start, tok.location};
        if (is_header and not is_open and logical_name == name) Error(
            loc,
            "Header imports must either specify '.*' or be renamed using 'as'"
        );

        if (not Consume(Tk::Semicolon)) Error("Expected ';'");
        mod->imports.emplace_back(
            std::move(name),
            std::move(logical_name),
            loc,
            is_open,
            is_header
        );
    }

    curr_func = mod->top_level_func;

    /// Set up scopes.
    scope_stack.push_back(mod->global_scope);

    /// Parse expressions.
    std::ignore = ParseExprs(Tk::Eof, mod->top_level_func->body->exprs);
}

/// <expr-for-in> ::= FOR <type> IDENTIFIER IN <expr> <do>
auto src::Parser::ParseFor() -> Result<Expr*> {
    auto start = curr_loc;
    bool reverse = At(Tk::ForReverse);
    Assert(Consume(Tk::For, Tk::ForReverse));

    /// Iteration variable.
    ScopeRAII sc{this};
    if (not At(Tk::Identifier)) return Error("Expected identifier");
    auto decl = new (mod) LocalDecl(
        curr_func,
        tok.text,
        Type::Unknown,
        {},
        LocalKind::Synthesised,
        tok.location
    );

    /// Parse range.
    Next();
    if (not Consume(Tk::In)) Error("Expected 'in'");
    auto range = ParseExpr();
    if (IsError(range)) return range.diag;

    /// Parse body.
    Consume(Tk::Do);
    auto body = ParseImplicitBlock();
    if (IsError(body)) return body.diag;
    return new (mod) ForInExpr(decl, *range, *body, reverse, start);
}

/// <expr-decl-ref> ::= IDENTIFIER
/// <expr-labelled> ::= IDENTIFIER ":" <expr>
auto src::Parser::ParseIdentExpr() -> Result<Expr*> {
    auto text = tok.text;
    auto start = Next();

    /// If the next token is `:`, then this is a label.
    if (Consume(Tk::Colon)) {
        const auto Label = [&](Expr* e) {
            return new (mod) LabelExpr(curr_func, std::move(text), e, start);
        };

        return ParseExpr() >> Label;
    }

    /// Otherwise, this is a regular name.
    return new (mod) DeclRefExpr(std::move(text), curr_scope, start);
}

/// <expr-if> ::= IF <expr> <then> { ELIF <expr> <then> } [ ELSE <implicit-block> ]
/// <then> ::= [ THEN ] <implicit-block>
auto src::Parser::ParseIf() -> Result<Expr*> {
    auto start = curr_loc;
    Assert(Consume(Tk::If, Tk::Elif));

    /// Parse condition, elif clauses, and else clause.
    auto cond = ParseExpr();
    Consume(Tk::Then);

    auto then = ParseImplicitBlock();
    if (Is(LookAhead(1), Tk::Elif, Tk::Else)) Consume(Tk::Semicolon);
    auto else_ = At(Tk::Elif)      ? ParseIf()
               : Consume(Tk::Else) ? ParseImplicitBlock()
                                   : Result<Expr*>::Null();

    /// Create the expression.
    if (IsError(cond, then, else_)) return Diag();
    return new (mod) IfExpr(*cond, *then, *else_, {start, curr_loc});
}

/// Syntactically any expression, but wrapped in an implicit
/// block if it isn’t already one.
///
/// If we are at a block, then we only parse that block here,
/// and nothing else, meaning that if the user writes sth like
/// `defer { ... } + 4`, then it will be parsed as `(defer {...})
/// + 4`.
///
/// This is to avoid having to deal with highly degenerate cases
/// such as `defer { ... } + int a`, which, if we parsed the entire
/// `+` as the body of the defer, would cause us to emit the variable
/// in the defer but declare it outside it. By only parsing at most
/// a block here if we’re already at a block, we avoid creating two
/// scopes and the variable will both be emitted and declared outside
/// the defer (of course, in this case, sema is actually going to
/// complain that we can’t add `void` and `int`, but that’s beside
/// the point here).
///
/// <implicit-block> ::= <expr>
auto src::Parser::ParseImplicitBlock() -> Result<BlockExpr*> {
    if (At(Tk::LBrace)) return ParseBlock();

    /// Not already a block.
    ScopeRAII sc{this};
    if (auto res = ParseExpr(); res.is_diag) return Diag();
    else {
        sc.scope->location = res->location;
        sc.scope->exprs.push_back(*res);
        sc.scope->implicit = true;
        return sc.scope;
    }
}

/// Parse the arguments of a naked invoke expression.
///
/// This does no checking as to whether this should syntactically be parsed
/// as a naked invoke expression in the first place, so make sure to do that
/// first before calling this.
auto src::Parser::ParseNakedInvokeExpr(Expr* callee) -> Result<InvokeExpr*> {
    SmallVector<Expr*> args;
    do {
        auto arg = ParseExpr(NakedInvokePrecedence);
        if (not IsError(arg)) args.push_back(*arg);
    } while (Consume(Tk::Comma));

    /// An assignment expression after a naked invocation is bound to it
    /// in case it turns out to be a declaration.
    SmallVector<Expr*> init_args;
    if (Consume(Tk::Assign)) {
        do {
            auto expr = ParseExpr();
            if (IsError(expr)) return expr.diag;
            init_args.push_back(*expr);
        } while (Consume(Tk::Comma));
    }

    Location loc = {callee->location, not args.empty() ? args.back()->location : Location{}};
    return new (mod) InvokeExpr(
        callee,
        std::move(args),
        true,
        std::move(init_args),
        loc
    );
}

/// <proc-params> ::= "(" <parameter> { "," <parameter> } ")"
/// <parameter>   ::= [ WITH ] <param-decl>
/// <param-decl>  ::= <type> [ IDENTIFIER ] | PROC [ IDENTIFIER ] <proc-signature>
auto src::Parser::ParseParamDeclList(
    SVI<ParamDecl*>& param_decls,
    SVI<Type>& param_types
) -> Location {
    auto loc = curr_loc;
    Assert(Consume(Tk::LParen));

    /// List may be empty.
    if (At(Tk::RParen)) return {loc, Next()};

    /// Parse param decls.
    do {
        /// Modifiers.
        const bool with = Consume(Tk::With);

        /// Procedure.
        if (At(Tk::Proc)) {
            auto sig = ParseSignature();

            /// Return type defaults to void.
            if (sig.type->ret_type == Type::Unknown) sig.type->ret_type = Type::Void;

            param_types.push_back(sig.type);
            param_decls.push_back(new (mod) ParamDecl(
                nullptr,
                std::move(sig.name),
                sig.type,
                with,
                sig.loc
            ));
        }

        /// Regular type.
        else {
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
            param_types.push_back(Type(*param_type));
            param_decls.push_back(new (mod) ParamDecl(
                nullptr,
                std::move(name),
                *param_type,
                with,
                tok.location
            ));
        }
    } while (Consume(Tk::Comma));

    /// Parse closing paren.
    loc = {loc, curr_loc};
    if (not Consume(Tk::RParen)) Error("Expected ')'");
    return loc;
}

/// <pragma> ::= PRAGMA "nomangle" ";"
void src::Parser::ParsePragma() {
    auto loc = curr_loc;
    Assert(Consume(Tk::Pragma));

    if (At(Tk::Identifier)) {
        if (tok.text == "nomangle") {
            default_mangling = Mangling::None;
            Next();
            if (not Consume(Tk::Semicolon)) Error("Expected ';'");
            return;
        }
    }

    Diag::Warning(ctx, loc, "Unknown pragma ignored");
    Synchronise();
}

/// <proc-extern>    ::= PROC IDENTIFIER <proc-signature>
/// <proc-named>     ::= PROC IDENTIFIER <proc-signature> <proc-body>
/// <init-decl>      ::= INIT <proc-signature> <proc-body>
/// <proc-body>      ::= <expr-block> | "=" <implicit-block>
auto src::Parser::ParseProc() -> Result<ProcDecl*> {
    /// Procedure signatures are rather complicated, so
    /// we’ll parse them separately.
    auto smf = At(Tk::Init, Tk::Delete);
    auto sig = ParseSignature();

    /// Create the procedure early so we can set it as
    /// the current procedure.
    tempset curr_func = new (mod) ProcDecl(
        mod,
        curr_func,
        std::move(sig.name),
        sig.type,
        std::move(sig.param_decls),
        sig.is_extern ? Linkage::Imported : Linkage::Internal,
        sig.is_nomangle ? Mangling::None : default_mangling,
        sig.loc
    );

    /// Parse the body if the procedure is not external.
    bool infer_return_type = false;
    if (not sig.is_extern) {
        auto body = Result<BlockExpr*>::Null();
        if (Consume(Tk::Assign)) {
            infer_return_type = true;
            body = ParseImplicitBlock();
        } else if (At(Tk::LBrace)) {
            body = ParseBlock();
        } else {
            body = Error("Expected '=' or '{{' at start of procedure body");
        }

        /// Set the body if there is one.
        if (IsError(body)) return body.diag;
        curr_func->body = *body;
        curr_func->body->set_function_scope();
    }

    /// If this is not an initialiser declaration and the return type is not
    /// inferred and not provided, then default to 'void' rather than 'unknown'.
    if (not smf and not infer_return_type and sig.type->ret_type == Type::Unknown)
        sig.type->ret_type = Type::Void;

    /// Add it to the current scope if it is named.
    if (not curr_func->name.empty()) curr_scope->declare(curr_func->name, curr_func);
    return curr_func;
}

/// Parse a procedure signature. The return type is 'unknown' by
/// default, and not void, to allow distinguishing `-> void` from
/// no return type at all.
///
/// <proc-signature> ::= [ <proc-args> ] [ <proc-ret> ] { <proc-attrs> }
/// <proc-ret>       ::= "->" <type>
/// <proc-attrs>     ::= EXTERN | NOMANGLE
auto src::Parser::ParseSignature() -> Signature {
    Signature sig;
    sig.loc = curr_loc;
    const auto init = At(Tk::Init);
    const auto del = At(Tk::Delete);
    Assert(Consume(Tk::Proc, Tk::Init, Tk::Delete));

    /// Parse name, if there is one.
    if (init) {
        sig.name = "init";
    } else if (del) {
        sig.name = "delete";
    } else if (At(Tk::Identifier)) {
        sig.name = tok.text;
        Next();
    }

    /// Parse the arguments if there are any. The argument
    /// list may be omitted altogether.
    SmallVector<Type> param_types;
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
    bool variadic = false;
    while (At(Tk::Identifier) and ( // clang-format off
        ParseAttr("nomangle", sig.is_nomangle) or
        ParseAttr("extern", sig.is_extern) or
        ParseAttr("variadic", variadic)
    )); // clang-format on

    /// Finally, parse the return type.
    Type ret_type = init or del ? Type::Void : Type::Unknown;
    if (Consume(Tk::RArrow)) {
        if (init or del) Error("{} may not specify a return type", init ? "Initialiser" : "Deleter");
        auto res = ParseType();
        if (not IsError(res)) ret_type = *res;
    }

    /// Create the procedure type.
    sig.type = new (mod) ProcType(
        std::move(param_types),
        ret_type,
        variadic,
        sig.loc
    );
    return sig;
}

/// <type-struct>  ::= STRUCT <name> <struct-rest>
/// <struct-anon>  ::= STRUCT <struct-rest>
/// <struct-rest>  ::= "{" { <struct-field> | <init-decl> } "}"
/// <struct-field> ::= <var-decl>
/// <init-decl>    ::= INIT <proc-signature> <proc-body>
auto src::Parser::ParseStruct() -> Result<StructType*> {
    Assert(At(Tk::Struct));
    auto start = Next();

    /// Parse name if there is one.
    std::string name;
    if (At(Tk::Identifier)) {
        name = tok.text;
        Next();
    }

    /// Parse fields and member functions.
    if (not Consume(Tk::LBrace)) return Error("Expected '{{' in struct type");
    ScopeRAII sc{this};
    SmallVector<FieldDecl*> fields;
    SmallVector<ProcDecl*> initialisers;
    StructType::MemberProcedures member_procs;
    ProcDecl* deleter{};
    sc.scope->set_struct_scope();
    while (not At(Tk::RBrace, Tk::Eof)) {
        if (Consume(Tk::Semicolon)) continue;

        /// Initialiser / Deleter.
        if (At(Tk::Init, Tk::Delete)) {
            auto init = At(Tk::Init);
            auto smf = ParseProc();
            if (IsError(smf)) {
                Synchronise(Tk::Semicolon, Tk::RBrace);
                continue;
            }

            /// Add it to the struct type.
            smf->parent = mod->top_level_func;
            if (init) initialisers.push_back(*smf);
            else {
                if (deleter) Error(smf->location, "Type already has a deleter");
                deleter = *smf;
            }
            continue;
        }

        /// Field.
        auto decl = ParseDecl();
        if (IsError(decl)) {
            Synchronise(Tk::Semicolon, Tk::RBrace);
            continue;
        }

        /// If the decl is a var decl, add it as a field.
        if (auto var = dyn_cast<LocalDecl>(*decl)) {
            if (not Consume(Tk::Semicolon)) Error("Expected ';'");
            if (
                auto it = rgs::find_if(fields, [&](auto f) { return f->name == var->name; });
                it != fields.end()
            ) {
                Error(var->location, "Cannot redeclare field '{}'", var->name);
                Diag::Note(ctx, (*it)->location, "Previous declaration was here");
                continue;
            }

            if (auto it = member_procs.find(var->name); it != member_procs.end()) {
                Error(
                    var->location,
                    "Cannot declare field '{}' with the same name as a procedure",
                    var->name
                );

                Diag::Note(
                    ctx,
                    it->second.front()->location,
                    "Member procedure declaration was here"
                );
                continue;
            }

            fields.push_back(new (mod) FieldDecl(
                var->name,
                var->type,
                var->location
            ));

            continue;
        }

        /// If it is a procedure, then this is a member function.
        if (auto proc = dyn_cast<ProcDecl>(*decl)) {
            if (proc->name.empty()) Error(decl->location, "Member procedures must have a name");
            if (
                auto it = rgs::find_if(fields, [&](auto f) { return f->name == proc->name; });
                it != fields.end()
            ) {
                Error(
                    proc->location,
                    "Cannot declare member procedure '{}' with the same name as a field",
                    proc->name
                );

                Diag::Note(ctx, (*it)->location, "Field declaration was here");
                continue;
            }

            member_procs[proc->name].push_back(proc);
            continue;
        }

        /// Other decls are currently illegal in this position.
        Error(decl->location, "This declaration is not allowed here");
        Synchronise();
    }

    /// Parse closing brace.
    if (not Consume(Tk::RBrace)) Error("Expected '}}'");

    /// Create the struct type.
    auto s = new (mod) StructType(
        mod,
        std::move(name),
        std::move(fields),
        std::move(initialisers),
        std::move(member_procs),
        deleter,
        sc.scope,
        Mangling::Source,
        {start, curr_loc}
    );

    /// Named structs go in the symbol table.
    if (not s->name.empty()) sc.scope->parent->declare(s->name, s);
    return s;
}

/// <type-prim>      ::= INTEGER_TYPE | INT | BOOL | NIL | VOID | NORETURN | VAR
/// <type-named>     ::= IDENTIFIER
/// <type-tuple>     ::= "(" { <type> "," } [ <type> ] ")"
/// <type-qualified> ::= <type> { <type-qual> }
/// <type-qual>      ::= "&" | "^" | "?" | "[" [ <expr> ] "]"
auto src::Parser::ParseType() -> Result<Type> {
    /// Parse base type.
    Expr* base_type = nullptr;
    switch (tok.type) {
        default: return Error("Expected type");

        case Tk::Struct: {
            auto ty = ParseStruct();
            if (IsError(ty)) return ty.diag;
            base_type = *ty;
        } break;

        /// Tuple type.
        case Tk::LParen: {
            auto start = Next();
            SmallVector<FieldDecl*> types;
            while (not At(Tk::RParen, Tk::Eof)) {
                auto ty = ParseType();
                if (IsError(ty)) return ty;
                types.push_back(new (mod) FieldDecl("", *ty, ty->location));
                if (not Consume(Tk::Comma)) break;
            }

            base_type = new (mod) TupleType(std::move(types), {start, curr_loc});
            if (not Consume(Tk::RParen)) Error("Expected ')'");
        } break;

        case Tk::Identifier:
            base_type = new (mod) DeclRefExpr(tok.text, curr_scope, tok.location);
            Next();
            break;

        case Tk::IntegerType:
            base_type = new (mod) IntType(Size::Bits(tok.integer.getZExtValue()), tok.location);
            Next();
            break;

        case Tk::Int:
            base_type = BuiltinType::Int(mod, tok.location);
            Next();
            break;

        case Tk::Bool:
            base_type = BuiltinType::Bool(mod, tok.location);
            Next();
            break;

        case Tk::Var:
            base_type = BuiltinType::Unknown(mod, tok.location);
            Next();
            break;

        case Tk::Void:
            base_type = BuiltinType::Void(mod, tok.location);
            Next();
            break;

        case Tk::NoReturn:
            base_type = BuiltinType::NoReturn(mod, tok.location);
            Next();
            break;

        case Tk::Nil:
            base_type = new (mod) Nil(curr_loc);
            Next();
            break;

        /// Builtins. Not officially part of the language.
        case Tk::CChar:
            base_type = ctx->ffi_char;
            Next();
            break;

        case Tk::CShort:
            base_type = ctx->ffi_short;
            Next();
            break;

        case Tk::CInt:
            base_type = ctx->ffi_int;
            Next();
            break;

        case Tk::CLong:
            base_type = ctx->ffi_long;
            Next();
            break;

        case Tk::CLongLong:
            base_type = ctx->ffi_long_long;
            Next();
            break;

        case Tk::CSizeT:
            base_type = ctx->ffi_size_t;
            Next();
            break;
    }

    /// Parse qualifiers.
    Type type{base_type};
    for (;;) {
        switch (tok.type) {
            default: return type;

            case Tk::Ampersand:
                type = new (mod) ReferenceType(type, {type->location, curr_loc});
                Next();
                break;

            case Tk::Caret:
                type = new (mod) ScopedPointerType(type, {type->location, curr_loc});
                Next();
                break;

            case Tk::Question:
                type = new (mod) OptionalType(type, {type->location, curr_loc});
                Next();
                break;

            case Tk::LBrack:
                Next();

                if (tok.type == Tk::RBrack) type = new (mod) SliceType(type, type->location);
                else {
                    auto dim = ParseExpr();
                    if (IsError(dim)) return dim.diag;
                    type = new (mod) ArrayType(type, *dim, type->location);
                }

                type->location = {type->location, curr_loc};
                if (not Consume(Tk::RBrack)) Error("Expected ']'");
                break;
        }
    }
}

/// <expr-while> ::= WHILE <expr> <do>
auto src::Parser::ParseWhile() -> Result<Expr*> {
    auto start = curr_loc;
    Assert(Consume(Tk::While));

    /// Parse condition and body.
    auto cond = ParseExpr();
    auto body = Result<BlockExpr*>::Null();
    if (IsError(cond)) return Diag();
    Consume(Tk::Do);
    body = ParseImplicitBlock();

    /// Create the expression.
    if (IsError(body)) return body;
    return new (mod) WhileExpr(*cond, *body, {start, curr_loc});
}

/// <expr-with> ::= WITH <expr> [ <do> ]
auto src::Parser::ParseWith() -> Result<WithExpr*> {
    auto start = curr_loc;
    Assert(Consume(Tk::With));

    /// Parse object.
    auto object = ParseExpr();

    /// Parse optional body.
    auto body = Result<BlockExpr*>::Null();
    if (Consume(Tk::Do) or MayStartAnExpression(tok.type)) body = ParseImplicitBlock();
    else if (not At(Tk::Semicolon)) Error("Expected body or ';' in with expression");
    if (IsError(object, body)) return Diag();
    return new (mod) WithExpr(*object, *body, {start, curr_loc});
}
