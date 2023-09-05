#include <memory>
#include <source/Frontend/AST.hh>
#include <source/Frontend/Parser.hh>

/// ===========================================================================
///  API
/// ===========================================================================
src::Parser::Parser(Context* ctx, File& f) : Lexer(ctx, f) {}

auto src::Parser::Parse(Context* ctx, File& f) -> std::unique_ptr<Module> {
    Parser p{ctx, f};
    p.ParseFile();
    return ctx->has_error() ? nullptr : std::move(p.mod_ptr);
}

/// ===========================================================================
///  Helpers.
/// ===========================================================================
namespace src {
namespace {
constexpr int BinaryOrPostfixPrecedence(Tk t) {
    switch (t) {
        default: return 0;
    }
}

inline constexpr int UnaryOperatorPrecedence = 1'000;
inline constexpr int InvokePrecedence = BinaryOrPostfixPrecedence(Tk::LParen);

constexpr bool RightAssociative(Tk t) {
    switch (t) {
        default: return false;
    }
}

constexpr bool MayStartAnExpression(Tk k) {
    switch (k) {
        case Tk::Identifier:
        case Tk::Integer:
        case Tk::StringLiteral:
        case Tk::Assert:
        case Tk::Asm:
        case Tk::If:
        case Tk::Match:
        case Tk::While:
        case Tk::For:
        case Tk::With:
        case Tk::Try:
        case Tk::Return:
        case Tk::Defer:
        case Tk::Break:
        case Tk::Continue:
        case Tk::Unreachable:
        case Tk::Extern:
        case Tk::Static:
        case Tk::Not:
        case Tk::True:
        case Tk::False:
        case Tk::Null:
        case Tk::Proc:
        case Tk::Var:
        case Tk::Val:
        case Tk::Enum:
        case Tk::Struct:
        case Tk::Type:
        case Tk::Typeof:
        case Tk::NoReturn:
        case Tk::Bool:
        case Tk::Void:
        case Tk::I8:
        case Tk::I16:
        case Tk::I32:
        case Tk::I64:
        case Tk::Int:
        case Tk::F32:
        case Tk::F64:
        case Tk::CChar:
        case Tk::CChar8T:
        case Tk::CChar16T:
        case Tk::CChar32T:
        case Tk::CWCharT:
        case Tk::CShort:
        case Tk::CInt:
        case Tk::CLong:
        case Tk::CLongLong:
        case Tk::CLongDouble:
        case Tk::CSizeT:
        case Tk::StringKw:
        case Tk::LParen:
        case Tk::LBrack:
        case Tk::LBrace:
        case Tk::Dot:
        case Tk::LArrow:
        case Tk::Plus:
        case Tk::Minus:
        case Tk::Caret:
        case Tk::Ampersand:
        case Tk::Tilde:
        case Tk::MinusMinus:
        case Tk::PlusPlus:
            return true;

        default:
            return false;
    }
}
} // namespace
} // namespace src

bool src::Parser::AtStartOfExpression() { return MayStartAnExpression(tok.type); }

/// ===========================================================================
///  Main Parser Implementation
/// ===========================================================================
/// <expr-assert> ::= ASSERT <expr> [ "," <expr> ]
auto src::Parser::ParseAssertExpr() -> Result<AssertExpr*> {
    Assert(Consume(Tk::Assert), "ParseAssertExpr() called without 'assert'");

    /// Parse condition and message, if any.
    auto cond = ParseExpr();
    auto mess = Consume(Tk::Comma) ? ParseExpr() : Result<Expr*>::Null();
    if (IsError(cond, mess)) return Diag();
    return new (mod) AssertExpr(*cond, *mess);
}

/// <expr-block> ::= "{" <exprs> "}"
auto src::Parser::ParseBlockExpr() -> Result<BlockExpr*> {
    auto brace_loc = curr_loc;
    Assert(Consume(Tk::LBrace), "ParseBlockExpr() called without '{{'");

    /// Parse expressions.
    ScopeRAII sc{this};
    ExprList exprs;
    while (not At(Tk::Eof, Tk::RBrace)) ParseExpressions(exprs);
    if (not Consume(Tk::RBrace)) return Error("Expected '}}' at end of block");
    return new (mod) BlockExpr(std::move(exprs), false, brace_loc);
}

/// <exprs> ::= { <expr-block> | <expr> ";" }
void src::Parser::ParseExpressions(ExprList& into) {
    const auto Add = [&](Expr* e) { into.push_back(e); };

    /// Block expression.
    if (At(Tk::LBrace)) {
        std::ignore = ParseBlockExpr() >>= Add;
        Consume(Tk::Semicolon);
    }

    /// Expression + semicolon.
    else {
        std::ignore = ParseExpr() >>= Add;
        if (not Consume(Tk::Semicolon)) {
            Error("Expected ';' after expression");
            Synchronise();
        }
    }
}

/// <source> ::= <preamble> { <exprs> | <pragma> }
void src::Parser::ParseFile() {
    /// Set up scopes.
    scope_stack.emplace_back(new Scope{nullptr, mod});
    scope_stack.emplace_back(new Scope{global_scope, mod});

    /// Parse preamble; this also creates the module.
    ParsePreamble();

    /// Get the static assertion context for top-level declarations.
    static_assertions = &mod->static_assertions;

    /// Parse expressions.
    while (not At(Tk::Eof)) ParseExpressions(mod->top_level_func->body->exprs);
}

/// <preamble>    ::= [ <module-decl> ] { <import> }
/// <module-decl> ::= MODULE <identifier> ";"
/// <module-name> ::= <identifier> | "<" TOKENS ">"
/// <import>      ::= IMPORT <module-name> [ "." "*" ] [ AS <identifier> ] ";"
void src::Parser::ParsePreamble() {
    /// If the first token is 'module', then this is a module.
    auto mod_loc = curr_loc;
    if (Consume(Tk::Module)) {
        if (not At(Tk::Identifier)) Error("Expected module name after 'module'");
        mod_ptr = std::make_unique<Module>(ctx, tok.text, mod_loc);
        Next();
        if (not Consume(Tk::Semicolon)) Error("Expected ';' after module name");
        while (At(Tk::Semicolon)) Next();
    } else {
        mod_ptr = std::make_unique<Module>(ctx, "");
    }

    /// Set the current function.
    curr_func = mod->top_level_func;

    /// Parse imports.
    while (At(Tk::Import)) {
        auto import_loc = curr_loc;
        bool is_header = false;
        bool is_open = false;
        std::string name;
        std::string alias;
        Next();

        /// '<' indicates that this is a header. Don’t consume
        /// it as we need to read the header name in raw mode.
        if (At(Tk::Lt)) {
            tempset raw_mode = true;
            is_header = true;

            /// Read the entire header name.
            while (lastc and lastc != '>') {
                name += lastc;
                Next();
            }

            /// Lex the next token.
            if (not lastc) Error("Expected '>' at end of header name");
            Next();
        }

        /// Otherwise, this is a regular module import.
        else {
            if (not At(Tk::Identifier)) Error("Expected module name after 'import'");
            name = tok.text;
            Next();
        }

        /// Parse the wildcard import if there is one.
        if (Consume(Tk::Dot)) {
            if (not Consume(Tk::Star)) Error("Expected '*' after '.'");
            is_open = true;
        }

        /// Parse the module alias if there is one. Header imports
        /// are just paths, so they should always have an alias.
        if (Consume(Tk::As)) {
            if (not At(Tk::Identifier)) Error("Expected identifier after 'as'");
            alias = tok.text;
            Next();
        } else {
            if (is_header) Error("Expected 'as' after header name");
            alias = name;
        }

        /// Create the import.
        if (not Consume(Tk::Semicolon)) Error("Expected ';' after module name");
        mod->imports.emplace_back(
            std::move(name),
            std::move(alias),
            import_loc,
            is_open,
            is_header
        );

        /// Consume excess semicolons.
        while (At(Tk::Semicolon)) Next();
    }
}

auto src::Parser::ParseExpr(int current_precedence) -> Result<Expr*> {
    return ParseExprImpl<false>(current_precedence);
}

template <bool parsing_control_expr_of_match>
auto src::Parser::ParseExprImpl(int curr_prec) -> Result<Expr*> {
    /// See below.
    const auto start_token = tok.type;

    /// Synchronise on the next closing parenthesis.
    const auto SyncRParen = [this] { while (not At(Tk::Eof, Tk::RBrack)) Next(); };

    /// Parse the LHS of the expression.
    auto lhs = Result<Expr*>::Null();
    switch (tok.type) {
        default:
            return Error("Expected expression");

        case Tk::Pragma:
        case Tk::MacroParameter:
            Unreachable();

        case Tk::Module:
        case Tk::Import:
            return Error("'{}' is not allowed here", tok.spelling);

        case Tk::Then:
        case Tk::Else:
        case Tk::Elif:
            return Error("'{}' without 'if'", tok.spelling);

        /// Currently, all rules that start with an identifier then
        /// follow up with something that is parsed like a binary or
        /// postfix operator, so we’ll handle all of that in the big
        /// operator parsing loop below.
        case Tk::Identifier:
            lhs = new (mod) NameRefExpr(tok.text, curr_scope, false, curr_loc);
            Next();
            break;

        case Tk::Integer:
            lhs = new (mod) IntegerLiteralExpr(tok.integer, curr_loc);
            Next();
            break;

        case Tk::StringLiteral:
            lhs = new (mod) StringLiteralExpr(mod->strtab.intern(tok.text), curr_loc);
            Next();
            break;

        case Tk::True:
        case Tk::False:
        case Tk::Null:
            lhs = new (mod) BuiltinLiteralExpr(tok.type, curr_loc);
            Next();
            break;

        case Tk::Unreachable:
            lhs = new (mod) UnreachableExpr(curr_loc);
            Next();
            break;

        case Tk::Assert: lhs = ParseAssertExpr(); break;
        case Tk::Asm: lhs = ParseInlineAsm(); break;
        case Tk::If: lhs = ParseIfExpr(); break;
        case Tk::Match: lhs = ParseMatchExpr(); break;
        case Tk::While: lhs = ParseWhileExpr(); break;
        case Tk::For: lhs = ParseForExpr(); break;
        case Tk::With: lhs = ParseWithExpr(); break;

        /// <expr-try> ::= TRY <expr>
        case Tk::Try: {
            auto loc = curr_loc;
            Next();
            auto op = ParseExprImpl<parsing_control_expr_of_match>(0);
            if (IsError(op)) return Diag();
            lhs = new (mod) TryExpr(*op, {loc, op->location});
        } break;

        /// RETURN [ <expr> ]
        case Tk::Return: {
            auto loc = curr_loc;
            Next();
            lhs = AtStartOfExpression() ? ParseExpr() : Result<Expr*>::Null();
            if (IsError(lhs)) return Diag();
            lhs = new (mod) ReturnExpr(*lhs, *lhs ? Location{loc, lhs->location} : loc);
        } break;

        /// DEFER <expr>
        case Tk::Defer: {
            auto loc = curr_loc;
            Next();
            lhs = ParseExpr();
            if (IsError(lhs)) return Diag();
            lhs = new (mod) DeferExpr(*lhs, {loc, lhs->location});
        } break;

        case Tk::Break:
        case Tk::Continue: {
            const auto is_break = At(Tk::Break);
            std::string label;
            auto loc = curr_loc;
            Next();
            if (At(Tk::Identifier)) {
                label = tok.text;
                loc = {loc, curr_loc};
                Next();
            }

            lhs = new (mod) BreakContinueExpr(std::move(label), is_break, loc);
        } break;

        case Tk::Export:
        case Tk::Extern: {
            const auto location = curr_loc;
            const bool is_exported = Consume(Tk::Export);
            const bool is_external = Consume(Tk::Extern);
            auto type = ParseType();
            if (IsError(type)) return Diag();
            lhs = ParseDecl(is_exported, is_external, location, *type);
        } break;

        case Tk::Static: {
            auto loc = curr_loc;
            Next();
            auto expr = ParseExpr();
            if (IsError(expr)) return Diag();
            lhs = new (mod) StaticExpr(*expr, {loc, expr->location});
        } break;

        case Tk::Var:
        case Tk::Val:
        case Tk::NoReturn:
        case Tk::Type:
        case Tk::Typeof:
        case Tk::Bool:
        case Tk::Void:
        case Tk::I8:
        case Tk::I16:
        case Tk::I32:
        case Tk::I64:
        case Tk::Int:
        case Tk::F32:
        case Tk::F64:
        case Tk::CChar:
        case Tk::CChar8T:
        case Tk::CChar16T:
        case Tk::CChar32T:
        case Tk::CWCharT:
        case Tk::CShort:
        case Tk::CInt:
        case Tk::CLong:
        case Tk::CLongLong:
        case Tk::CLongDouble:
        case Tk::CSizeT:
        case Tk::StringKw: {
            auto ty = ParseType();
            if (IsError(ty)) return Diag();
            lhs = ParseTypeExpr(*ty);
        } break;

        case Tk::Not:
        case Tk::Caret:
        case Tk::Ampersand:
        case Tk::Plus:
        case Tk::Minus:
        case Tk::Tilde:
        case Tk::MinusMinus:
        case Tk::PlusPlus: {
            auto loc = curr_loc;
            auto op = tok.type;
            Next();
            auto operand = ParseExprImpl<parsing_control_expr_of_match>(UnaryOperatorPrecedence);
            if (IsError(operand)) return Diag();
            lhs = new (mod) UnaryExpr(op, *operand, false, {loc, operand->location});
        } break;

        /// Paren expr or tuple.
        case Tk::LParen: {
            auto start = curr_loc;
            auto expr = (Next(), ParseExpr());
            if (IsError(expr)) {
                SyncRParen();
                return Diag();
            }

            /// If there is a comma, then this is a tuple.
            if (Consume(Tk::Comma)) {
                ExprList exprs{*expr};

                while (not At(Tk::Eof, Tk::RParen)) {
                    auto e = ParseExpr();
                    if (not IsError(e)) exprs.push_back(*e);
                    if (not Consume(Tk::Comma)) break;
                }

                if (not Consume(Tk::RParen)) {
                    Error("Expected ')' at end of tuple");
                    SyncRParen();
                }

                lhs = new (mod) TupleLiteralExpr(std::move(exprs), {start, curr_loc});
                break;
            }

            /// Otherwise, this is a paren expression.
            if (not Consume(Tk::RParen)) {
                Error("Expected ')' ");
                SyncRParen();
            }

            lhs = new (mod) ParenExpr(*expr, {start, curr_loc});
        } break;

        /// Array literal.
        case Tk::LBrack: {
            auto start = curr_loc;

            /// Parse the literal elements.
            ExprList exprs;
            while (not At(Tk::Eof, Tk::RBrack)) {
                auto e = ParseExpr();
                if (not IsError(e)) exprs.push_back(*e);
                if (not Consume(Tk::Comma)) break;
            }

            /// Consume the closing bracket.
            if (not Consume(Tk::RBrack)) {
                Error("Expected ']' at end of array literal");
                SyncRParen();
            }

            lhs = new (mod) ArrayLiteralExpr(std::move(exprs), {start, curr_loc});
        } break;

        case Tk::Dot: {
            auto loc = curr_loc;
            Next();

            std::string name;
            if (At(Tk::Identifier)) {
                name = tok.text;
                loc = {loc, curr_loc};
                Next();
            } else {
                Error("Expected identifier after '.'");
            }

            lhs = new (mod) NameRefExpr(std::move(name), curr_scope, true, loc);
        } break;

        case Tk::Enum:
            lhs = ParseEnumDecl();
            break;

        case Tk::Struct:
            lhs = ParseStructDecl();
            break;

        case Tk::Proc:
            lhs = ParseProcExpr();
            break;

        case Tk::LArrow:
            lhs = ParseTerseProcExpr({});
            break;

        case Tk::LBrace:
            lhs = ParseBlockExpr();
            break;
    }

    /// Stop here if there was an error.
    if (lhs.is_diag) return lhs.diag;

    /// Make sure that the rest of the parser knows that this
    /// token can start an expression.
    Assert(
        MayStartAnExpression(start_token),
        "Add {} to MayStartAnExpression()",
        Spelling(start_token)
    );

    /// Parse anything that looks like a binary operator or
    /// postfix operator.
    for (;;) {
        /// Parse an invoke expression without parens.
        const auto ParseNakedInvokeExpr = [&] {
            ExprList args;
            do {
                auto arg = ParseExpr(InvokePrecedence);
                if (not IsError(arg)) args.push_back(*arg);
            } while (Consume(Tk::Comma));

            /// An assignment expression after a naked invocation is bound to it
            /// in case it turns out to be a declaration.
            auto init = Result<Expr*>::Null();
            if (Consume(Tk::Assign)) {
                auto expr = ParseExpr();
                if (IsError(init)) return Diag();
            }

            lhs = new (mod) InvokeExpr(*lhs, std::move(args), false, {lhs->location, curr_loc}, *init);
        };

        /// Some operators need special parsing.
        switch (tok.type) {
            default: break;

            /// If the token is an identifier, then this is an invoke expression,
            /// which means that this is either a declaration, a function call, a
            /// struct literal, or a template instantiation.
            case Tk::Identifier: {
                lhs = ParseNakedInvokeExpr();
                continue;
            }

            /// An invoke expression that uses parens cannot be a declaration.
            case Tk::LParen: {
                Next();
                ExprList args;
                while (not At(Tk::Eof, Tk::RParen)) {
                    auto arg = ParseExpr(InvokePrecedence);
                    if (not IsError(arg)) args.push_back(*arg);
                    if (not Consume(Tk::Comma)) break;
                }

                if (not Consume(Tk::RParen)) {
                    Error("Expected ')' at end of function call");
                    SyncRParen();
                }

                lhs = new (mod) InvokeExpr(*lhs, std::move(args), true, {lhs->location, curr_loc});
                continue;
            }

            /// Member or metaproperty access.
            case Tk::Dot:
            case Tk::ColonColon: {
                auto op = tok.type;
                auto loc = curr_loc;
                Next();

                std::string name;
                if (At(Tk::Identifier)) {
                    name = tok.text;
                    loc = {loc, curr_loc};
                    Next();
                } else {
                    Error("Expected identifier after '{}'", Spelling(op));
                }

                lhs = op == Tk::Dot
                        ? cast<Expr>(new (mod) MemberAccessExpr(*lhs, std::move(name), loc))
                        : cast<Expr>(new (mod) MetapropAccessExpr(*lhs, std::move(name), loc));
                continue;
            }

            /// Cast.
            case Tk::As:
            case Tk::AsBang: {
                auto kind = tok.type == Tk::As ? CastKind::Soft : CastKind::Hard;
                auto loc = curr_loc;
                Next();

                auto type = ParseType();
                if (IsError(type)) return Diag();
                lhs = new (mod) CastExpr(*lhs, *type, kind, {loc, type->location});
                continue;
            }

            /// An identifier followed by a comma can only be part of
            /// the argument list of a terse proc.
            case Tk::Comma: {
                auto loc = curr_loc;
                auto nr = dyn_cast<NameRefExpr>(*lhs);
                if (not nr or nr->is_local) {
                    Error("Unexpected ','");
                    Next();
                    continue;
                }

                /// Print a help note explaining why we’re even parsing this this way.
                auto HelpNote = [&] {
                    Diag::Note(
                        ctx,
                        loc,
                        "Parsing terse proc because of the ',' here. Did you accidentally put a comma here?"
                    );
                };

                /// Parse a comma separated list of identifiers.
                SmallVector<std::string> args;
                args.push_back(nr->name);
                while (Consume(Tk::Comma)) {
                    if (not At(Tk::Identifier)) {
                        Error("Expected identifier after ',' while parsing terse proc arguments.");
                        HelpNote();

                        /// Stop parsing this to avoid more confusing errors.
                        return Diag();
                    }

                    args.push_back(tok.text);
                    Next();
                }

                /// The next token must be an arrow.
                if (not At(Tk::RArrow)) {
                    Error("Expected '->' after argument list of terse proc");
                    HelpNote();
                    return Diag();
                }

                /// Parse the rest of the terse proc.
                lhs = ParseTerseProcExpr(std::move(args), lhs->location);
                continue;
            }

            /// Terse proc with one argument.
            case Tk::RArrow: {
                auto nr = dyn_cast<NameRefExpr>(*lhs);
                if (not nr or nr->is_local) {
                    Error("Unexpected '->'");
                    Next();
                    continue;
                }

                lhs = ParseTerseProcExpr({nr->name}, lhs->location);
            }

            /// TODO: Handling of type quals.
        }

        /// Check the precedence of the current token.
        const int prec = BinaryOrPostfixPrecedence(tok.type);
        if (not(prec > curr_prec or (prec == curr_prec and RightAssociative(tok.type)))) break;

        /// TODO: Parse binary operators.
    }
}

auto src::Parser::ParseExprInNewScope() -> Result<Expr*> {
    ScopeRAII sc{this};
    return ParseExpr();
}

/// <expr-for>     ::= FOR ( <for-infinite> | <for-each> | <for-cstyle> | <for-in> | <for-enum-in> )
/// <for-infinite> ::= DO <expr> | <expr-block>
/// <for-each>     ::= <expr> [ DO ] <expr>
/// <for-cstyle>   ::= [ <expr> ] ";" [ <expr> ] ";" [ <expr> ] [ DO ] <expr>
/// <for-in>       ::= <decl-base> IN <expr> [ DO ] <expr>
/// <for-enum-in>  ::= ENUM <identifier> [ "," <decl-base> ] IN <expr> [ DO ] <expr>
auto src::Parser::ParseForExpr() -> Result<Expr*> {
    auto for_loc = curr_loc;
    Assert(Consume(Tk::For), "ParseForExpr() called without 'for'");
    ScopeRAII sc{this};

    /// Parse optional 'do', followed by an expression.
    const auto ParseDoExpr = [&] -> Result<Expr*> {
        Consume(Tk::Do);
        return ParseExpr();
    };

    /// Parse a C-style for loop.
    const auto ParseForCStyle = [&](Result<Expr*> init) -> Result<Expr*> {
        if (not Consume(Tk::Semicolon)) Error("Expected ';' after first clause of c-style for loop");
        auto cond = At(Tk::Semicolon) ? Result<Expr*>::Null() : ParseExpr();
        if (not Consume(Tk::Semicolon)) Error("Expected ';' after condition of c-style for loop");
        auto step = AtStartOfExpression() ? ParseExpr() : Result<Expr*>::Null();
        auto body = ParseDoExpr();
        if (IsError(init, cond, step, body)) return Diag();
        return new (mod) ForCStyleExpr(*init, *cond, *step, *body, for_loc);
    };

    /// Parse a for-in loop.
    const auto ParseForIn = [&](std::string enum_name, Result<Expr*> decl) -> Result<Expr*> {
        /// Create a variable for the enumerator.
        bool has_enum = not enum_name.empty();
        auto enum_decl = Result<Expr*>::Null();
        if (has_enum) {
            enum_decl = new (mod) VarDecl(
                std::move(enum_name),
                curr_func,
                nullptr,
                Linkage::Local,
                Mangling::Source
            );
        }

        /// Parse 'in' keyword, range, and body.
        if (not Consume(Tk::In)) Error("Expected 'in' after enumerator in for-in loop");
        auto range = ParseExpr();
        auto body = ParseDoExpr();

        /// Create the expression.
        if (IsError(decl, enum_decl, range, body)) return Diag();
        return new (mod) ForInExpr(
            *decl,
            *enum_decl,
            *range,
            *body,
            for_loc
        );
    };

    /// Case 1: for-enum-in.
    if (Consume(Tk::Enum)) {
        std::string enum_name;
        if (not At(Tk::Identifier)) Error("Expected identifier after 'enum'");
        else {
            enum_name = tok.text;
            Next();
        }

        /// Decl is optional.
        auto control = Result<Expr*>::Null();
        if (Consume(Tk::Comma)) control = ParseDeclBase();
        return ParseForIn(std::move(enum_name), std::move(control));
    }

    /// Case 2: for-infinite.
    if (Consume(Tk::Do) or At(Tk::LBrace)) {
        auto body = ParseExpr();
        if (IsError(body)) return Diag();
        return new (mod) ForInfiniteExpr(*body, for_loc);
    }

    /// Maybe case 3: for-cstyle.
    if (At(Tk::Semicolon)) return ParseForCStyle(Result<Expr*>::Null());

    /// All remaining cases require at least one expression.
    auto control = ParseExpr();

    /// Case 3: for-cstyle.
    if (At(Tk::Semicolon)) return ParseForCStyle(std::move(control));

    /// Case 4: for-in.
    if (At(Tk::In)) return ParseForIn("", std::move(control));

    /// Case 5: for-each.
    auto body = ParseDoExpr();
    if (IsError(control, body)) return Diag();
    return new (mod) ForInExpr(nullptr, nullptr, *control, *body, for_loc);
}

/// <expr-if> ::= IF <expr> [ THEN ] <expr> { ELIF <expr> [ THEN ] <expr> } [ ELSE <expr> ]
auto src::Parser::ParseIfExpr() -> Result<IfExpr*> {
    auto if_loc = curr_loc;
    Assert(
        Consume(Tk::If) or Consume(Tk::Elif),
        "ParseIfExpr() called without 'if' or 'elif'"
    );

    /// Parse condition, body, and else clause.
    ScopeRAII sc{this};
    auto cond = ParseExpr();
    auto body = (Consume(Tk::Then), ParseExpr());

    /// Else clause is not in the same scope.
    sc.pop();
    auto elif = At(Tk::Elif)      ? ParseIfExpr()
              : Consume(Tk::Else) ? ParseExprInNewScope()
                                  : Result<Expr*>::Null();

    /// Create the expression.
    if (IsError(cond, body, elif)) return Diag();
    return new (mod) IfExpr(*cond, *body, *elif, if_loc);
}

/// <expr-asm> ::= ASM "{" { <asm-instruction> } "}" | ASM <asm-instruction>
/// <asm-instruction> ::= [ <identifier> ":" ] <identifier> <asm-operands> NEWLINE
/// <asm-operands> ::= [ <expr> ] { "," <expr> }
auto src::Parser::ParseInlineAsm() -> Result<Expr*> {
    Diag::ICE("TODO: Implement parsing of inline assembly");
}

/// <expr-match>    ::= MATCH <match-control> "{" { <match-case> } [ ELSE [ ":" ] <expr> ] "}"
/// <match-control> ::= [ <expr> [ <binary> ] ]
/// <match-case>    ::= <expr> [ ":" ] <expr>
auto src::Parser::ParseMatchExpr() -> Result<MatchExpr*> {
    auto loc = curr_loc;
    Assert(Consume(Tk::Match), "ParseMatchExpr() called without 'match'");

    /// Parse controlling expression and match operator. Make sure
    /// to tell the expression parser not to parse a binary expression
    /// here.
    auto control = Result<Expr*>::Null();
    auto op = Tk::Invalid;
    if (not At(Tk::LBrace)) control = ParseExprImpl<true>(NullPrecedence);
    if (BinaryOrPostfixPrecedence(tok.type) != -1) {
        op = tok.type;
        Next();
    }

    /// Parse match cases.
    SmallVector<CaseExpr*> cases;
    if (not Consume(Tk::LBrace)) Error("Expected '{{' at start of match body");
    while (not At(Tk::LBrace, Tk::Eof, Tk::Else)) {
        ScopeRAII sc{this};
        auto expr = ParseExpr();
        auto body = (Consume(Tk::Colon), ParseExpr());
        auto fallthrough = Consume(Tk::Fallthrough);
        if (IsError(expr, body)) return Diag(); /// Stop to avoid infinite loop.
        cases.push_back(new (mod) CaseExpr(*expr, *body, fallthrough, expr->location));
    }

    /// Parse else clause, if any.
    auto else_body = Consume(Tk::Else) ? (Consume(Tk::Colon), ParseExprInNewScope()) : Result<Expr*>::Null();
    if (not Consume(Tk::RBrace)) Error("Expected '}}' at end of match body");

    /// Create the expression.
    if (IsError(control, else_body)) return Diag();
    return new (mod) MatchExpr(
        *control,
        op,
        std::move(cases),
        else_body ? new (mod) CaseExpr(nullptr, *else_body, false, else_body->location) : nullptr,
        loc
    );
}

/// <expr-while> ::= WHILE <expr> [ DO ] <expr>
auto src::Parser::ParseWhileExpr() -> Result<WhileExpr*> {
    auto loc = curr_loc;
    Assert(Consume(Tk::While), "ParseWhileExpr() called without 'while'");

    /// Parse condition and body.
    ScopeRAII sc{this};
    auto cond = ParseExpr();
    auto body = (Consume(Tk::Do), ParseExpr());
    if (IsError(cond, body)) return Diag();
    return new (mod) WhileExpr(*cond, *body, loc);
}

/// <expr-with> ::= "with" <expr> [ DO ] [ <expr> ]
auto src::Parser::ParseWithExpr() -> Result<WithExpr*> {
    auto loc = curr_loc;
    Assert(Consume(Tk::With), "ParseWithExpr() called without 'with'");

    /// Parse the controlling expression.
    auto expr = ParseExpr();

    /// A with expression may have a body.
    Scope* sc = curr_scope;
    auto body = Result<Expr*>::Null();
    if (Consume(Tk::Do) or AtStartOfExpression()) {
        ScopeRAII ra{this};
        sc = ra.scope;
        body = ParseExpr();
    }

    /// Create the expression.
    if (IsError(expr, body)) return Diag();
    return new (mod) WithExpr(*expr, *body, sc, loc);
}
