#include <memory>
#include <source/Frontend/AST.hh>
#include <source/Frontend/Parser.hh>

#define nop()
#define bind SRC_BIND Parser::

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
/// Note: since a bunch of these aren’t really parsed
/// like operators, there precedences are only listed
/// here for conceptual reasons.
constexpr int BinaryOrPostfixPrecedence(Tk t) { // clang-format off
    switch (t) {
        case Tk::Dot:
        case Tk::ColonColon:
            return std::numeric_limits<int>::max();

        case Tk::MinusMinus:
        case Tk::PlusPlus:
        case Tk::Identifier:
        case Tk::LParen:
        case Tk::LBrack:
    /// case [VectorReduction]:
            return 10'000;

    /// case [PrefixOperator}:
    ///     return 1'000

        case Tk::As:
        case Tk::AsBang:
            return 100;

        case Tk::StarStar:
            return 90;

        case Tk::Star:
        case Tk::Slash:
        case Tk::Percent:
            return 80;

        case Tk::Plus:
        case Tk::Minus:
            return 70;

        case Tk::ShiftLeft:
        case Tk::ShiftRight:
        case Tk::ShiftRightLogical:
            return 60;

        case Tk::Xor:
            return 50;

        case Tk::DotDotLess:
        case Tk::DotDotEq:
            return 40;

        case Tk::Is:
        case Tk::Lt:
        case Tk::Le:
        case Tk::Gt:
        case Tk::Ge:
        case Tk::EqEq:
        case Tk::Neq:
            return 30;

        case Tk::And:
            return 20;

        case Tk::Or:
            return 10;

        case Tk::Assign:
        case Tk::LArrow:
        case Tk::PlusEq:
        case Tk::MinusEq:
        case Tk::StarEq:
        case Tk::SlashEq:
        case Tk::PercentEq:
        case Tk::ShiftLeftEq:
        case Tk::ShiftRightEq:
        case Tk::ShiftRightLogicalEq:
        case Tk::StarStarEq:
            return 1;

        default:
            return -1;
    }
} // clang-format on

/// Note: unary prefix operators have a precedence higher than any
/// other operator, except special operators (e.g. `as`) as well as
/// postfix operators. Vector reduction operators are postfix, so
/// they get the same precedence as postfix operators.
inline constexpr int VectorReductionPrecedence = BinaryOrPostfixPrecedence(Tk::PlusPlus);
inline constexpr int PrefixOperatorPrecedence = 1'000;
inline constexpr int InvokePrecedence = BinaryOrPostfixPrecedence(Tk::LParen);

constexpr bool IsPostfix(Tk t) {
    switch (t) {
        case Tk::MinusMinus:
        case Tk::PlusPlus:
            return true;

        default:
            return false;
    }
}

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

/// Used to represent type qualifiers of a proc signature.
struct TypeQualifier {
    Tk kind{};
    Expr* array_dimension{};
};

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

/// <decl>             ::= [ EXPORT ] [ EXTERN ] <decl-unqualified>
/// <decl-unqualified> ::= <decl-multi> | <decl-base> | <proc-unqual-sig> | <proc-named-sig>
/// <decl-multi>       ::= <decl-base> { "," <identifier> } [ "=" <expr> ]
/// <decl-base>        ::= <type> <identifier> [ <var-attrs> ]
/// <var-attrs>        ::= "nomangle"
auto src::Parser::ParseDecl(bool is_extern, Location loc) -> Result<Expr*> {
    auto type = Result<Expr*>::Null();

    /// The declaration type may contain a signature.
    if (At(Tk::Proc)) {
        auto sig = ParseProcSignature();
        if (sig.is_diag) return sig.diag;
        if (isa<FunctionType>((*sig).sig_type)) return ParseProcBody(is_extern, std::move(*sig));
        type = (*sig).sig_type;
    }

    /// Otherwise, parse the type.
    else { type = ParseType(); }

    /// Parse the name.
    if (not At(Tk::Identifier)) return Error("Expected identifier after type");
    std::string name = tok.text;
    loc = {loc, Next()};

    /// Helper to parse attributes.
    const auto ParseAttr = [&](std::string_view name, bool& flag) {
        if (tok.text == name) {
            if (flag) Diag::Warning(ctx, curr_loc, "Duplicated '{}' attribute ignored", name);
            loc = {loc, Next()};
            return flag = true;
        }

        return false;
    };

    /// Parse the attributes.
    bool nomangle{};
    while (At(Tk::Identifier) and ( // clang-format off
        ParseAttr("nomangle", nomangle)
    )); // clang-format on

    /// Parse the initialiser if there is one.
    auto init = Result<Expr*>::Null();
    if (Consume(Tk::Assign)) init = ParseExpr();

    /// Determine linkage.
    Linkage linkage;
    if (curr_func == mod->top_level_func) linkage = is_extern ? Linkage::Imported : Linkage::Internal;
    else linkage = Linkage::Local;

    /// TODO: `static` on a decl to make it a static variable. Top-level
    ///       variables should not be static by default.
    /// Create the decl.
    if (IsError(type, init)) return Diag();
    return new (mod) VarDecl(
        std::move(name),
        curr_func,
        *type,
        *init,
        linkage,
        nomangle ? Mangling::None : Mangling::Source,
        loc
    );
}

/// <enum-named> ::= [ EXPORT ] ENUM <identifier> [ ":" <type> ] <enum-rest>
/// <enum-anon>  ::= ENUM [ ":" <type> ] <enum-rest>
/// <enum-rest>  ::= "{" [ <enumerator> { "," <enumerator>  } ] "}"
/// <enumerator> ::= <identifier> [ "=" <expr> ]
auto src::Parser::ParseEnumDecl() -> Result<EnumType*> {
    auto loc = curr_loc;

    /// Name is optional.
    std::string name;
    if (At(Tk::Identifier)) {
        name = tok.text;
        Next();
    }

    /// Parse the type.
    Expr* type = Type::Unknown;
    if (Consume(Tk::Colon)) {
        auto res = ParseType();
        if (res.is_value) type = *res;
    }

    /// Parse enumerators.
    if (not Consume(Tk::LBrace)) return Error("Expected '{{' after enum");
    SmallVector<EnumeratorDecl*> enumerators;
    while (not At(Tk::RBrace, Tk::Eof)) {
        /// Name is required.
        if (not At(Tk::Identifier)) return Error("Expected identifier");
        auto enumerator_name = tok.text;
        auto enumerator_loc = Next();

        /// Value is optional.
        Expr* value = nullptr;
        if (Consume(Tk::Assign)) {
            auto val = ParseExpr();
            if (val.is_value) value = *val;
        }

        /// Add the numerator.
        enumerators.push_back(new (mod) EnumeratorDecl(std::move(enumerator_name), value, enumerator_loc));
        if (not Consume(Tk::Comma)) break;
    }

    /// Create the enum.
    if (not Consume(Tk::RBrace)) return Error("Expected '}}' at end of enum");
    return new (mod) EnumType(mod, std::move(name), std::move(enumerators), type, loc);
}

auto src::Parser::ParseExpr(int curr_prec) -> Result<Expr*> {
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
            auto loc = Next();
            auto op = ParseExpr();
            if (IsError(op)) return Diag();
            lhs = new (mod) TryExpr(*op, {loc, op->location});
        } break;

        /// RETURN [ <expr> ]
        case Tk::Return: {
            auto loc = Next();
            lhs = AtStartOfExpression() ? ParseExpr() : Result<Expr*>::Null();
            if (IsError(lhs)) return Diag();
            lhs = new (mod) ReturnExpr(*lhs, *lhs ? Location{loc, lhs->location} : loc);
        } break;

        /// DEFER <expr>
        case Tk::Defer: {
            auto loc = Next();
            lhs = ParseExpr();
            if (IsError(lhs)) return Diag();
            lhs = new (mod) DeferExpr(*lhs, {loc, lhs->location});
        } break;

        case Tk::Break:
        case Tk::Continue: {
            const auto is_break = At(Tk::Break);
            std::string label;
            auto loc = Next();
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
            auto Export = [&](Expr* e) {
                if (is_exported) {
                    if (auto decl = dyn_cast<ObjectDecl>(e)) {
                        decl->linkage = decl->linkage == Linkage::Imported
                                          ? Linkage::Reexported
                                          : Linkage::Exported;
                    } else if (isa<StructType, EnumType>(e)) {
                        auto& ex = mod->exports[cast<NamedType>(e)->name];
                        ex.push_back(e);
                    } else {
                        Error(e->location, "Cannot export this");
                    }
                }
                return e;
            };
            lhs = ParseDecl(is_external, location) >> Export;
        } break;

        case Tk::Static: {
            auto loc = Next();
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
        case Tk::StringKw:
            lhs = ParseType();
            break;

        case Tk::Not:
        case Tk::Caret:
        case Tk::Ampersand:
        case Tk::Plus:
        case Tk::Minus:
        case Tk::Tilde:
        case Tk::MinusMinus:
        case Tk::PlusPlus: {
            auto op = tok.type;
            auto loc = Next();
            auto operand = ParseExpr(PrefixOperatorPrecedence);
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
            auto loc = Next();

            std::string name;
            if (At(Tk::Identifier)) {
                name = tok.text;
                loc = {loc, Next()};
            } else {
                Error("Expected identifier after '.'");
            }

            lhs = new (mod) NameRefExpr(std::move(name), curr_scope, true, loc);
        } break;

        case Tk::Enum:
            lhs = ParseEnumDecl();
            break;

        case Tk::Struct:
        case Tk::Dynamic:
            lhs = ParseStructDecl();
            break;

        case Tk::Proc:
            lhs = ParseProcExpr(false);
            break;

        case Tk::LArrow:
            lhs = ParseTerseProcExpr({}, curr_loc);
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
                    not AtStartOfExpression() or
                    InvokePrecedence < curr_prec or
                    At(Tk::LBrace) or
                    IsPostfix(tok.type)
                ) break;

                /// Parse the arguments of the invoke expression.
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
            case Tk::Dot: {
                auto loc = Next();

                std::string name;
                if (At(Tk::Identifier)) {
                    name = tok.text;
                    loc = {loc, Next()};
                } else {
                    Error("Expected identifier after '.'");
                }

                lhs = cast<Expr>(new (mod) MemberAccessExpr(*lhs, std::move(name), loc));
                continue;
            }

            /// Cast.
            case Tk::As:
            case Tk::AsBang: {
                auto kind = tok.type == Tk::As ? CastKind::Soft : CastKind::Hard;
                auto loc = Next();

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
                continue;
            }

            /// '[' can be a slice type, array type, subscript expression,
            /// or vector reduction.
            case Tk::LBrack: {
                switch (const auto tok = LookAhead(1).type) {
                    /// Vector reduction operator.
                    case Tk::Plus:
                    case Tk::Star:
                    case Tk::And:
                    case Tk::Or:
                    case Tk::Xor:
                    case Tk::Lt:
                    case Tk::Gt: {
                        /// If there is another token other than ']' after the
                        /// operator, and the operator could also be a unary
                        /// prefix operator, then this is not a vector reduction.
                        if (not Is(LookAhead(2), Tk::RBrack) and tok == Tk::Plus) break;

                        /// Respect operator precedence.
                        if (VectorReductionPrecedence < curr_prec) return lhs;

                        /// Yeet '[' and operator.
                        Location op_loc = {curr_loc, LookAhead(1).location};
                        Next(), Next();

                        /// Create the expression and yeet ']'. We keep parsing if
                        /// if there is no ']' because the user probably just forgot
                        /// one.
                        lhs = new (mod) VectorReduceExpr(tok, *lhs, {lhs->location, curr_loc});
                        if (not Consume(Tk::RBrack)) Error(op_loc, "Missing ']' after '[{}'", Spelling(tok));
                        continue;
                    }

                    /// Slice type.
                    case Tk::RBrack: goto parse_type;

                    /// Maybe an array type, but we’ll parse it as a subscript
                    /// expression for now.
                    default: {
                        if (BinaryOrPostfixPrecedence(Tk::LBrack) < curr_prec) return lhs;
                        Next();
                        auto rhs = ParseExpr();
                        if (IsError(rhs)) return Diag();
                        lhs = new (mod) BinaryExpr(Tk::LBrack, *lhs, *rhs, {lhs->location, curr_loc});
                        if (not Consume(Tk::RBrack)) Error("Expected ']' after subscript expression");
                        continue;
                    }
                }
            } break;

            /// Since we have proper bools, we can use both 'and' and 'or'
            /// for bools and integers and can thus use '|' for something
            /// else. '&' is also not a binary operator because of this, so
            /// if we encounter one after an expression, it can only be
            /// a type. Lastly, '?', '..', and '^' are not operators in this
            /// language, so if we encounter one of them, we know that we
            /// are definitely dealing w/ a type.
            ///
            /// '&' can also be a unary operator, but we always take the
            /// longest possible match, so if an '&' can be a type qualifier,
            /// then it is a type qualifier. Besides, address-of is not that
            /// common of an operation in a language that also has references.
            case Tk::VBar:
            case Tk::Ampersand:
            case Tk::Question:
            case Tk::DotDot:
            case Tk::Caret:
            parse_type:
                lhs = ParseType(*lhs);
                continue;
        }

        /// Check the precedence of the current token.
        const int prec = BinaryOrPostfixPrecedence(tok.type);
        if (not(prec > curr_prec or (prec == curr_prec and RightAssociative(tok.type)))) break;

        /// Postfix expressions only have one operand.
        if (IsPostfix(tok.type)) {
            lhs = new (mod) UnaryExpr(tok.type, *lhs, true, {lhs->location, curr_loc});
            Next();
            continue;
        }

        /// Parse the RHS of the expression.
        Next();
        auto rhs = ParseExpr(prec);
        if (IsError(rhs)) return Diag();
        lhs = new (mod) BinaryExpr(tok.type, *lhs, *rhs, {lhs->location, rhs->location});
    }

    /// Nothing left to parse.
    return lhs;
}

auto src::Parser::ParseExprInNewScope() -> Result<Expr*> {
    ScopeRAII sc{this};
    return ParseExpr();
}

/// <exprs> ::= { <expr-block> | <expr> ";" }
void src::Parser::ParseExpressions(ExprList& into) {
    const auto Add = [&](Expr* e) { into.push_back(e); };

    /// Block expression.
    if (At(Tk::LBrace)) {
        std::ignore = ParseBlockExpr() >> Add;
        Consume(Tk::Semicolon);
    }

    /// Expression + semicolon.
    else {
        std::ignore = ParseExpr() >> Add;
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
    auto& exprs = cast<BlockExpr>(mod->top_level_func->body)->exprs;
    while (not At(Tk::Eof)) ParseExpressions(exprs);
}

/// <expr-for>     ::= FOR ( <for-infinite> | <for-each> | <for-cstyle> | <for-in> | <for-enum-in> )
/// <for-infinite> ::= DO <expr> | <expr-block>
/// <for-each>     ::= <expr> [ DO ] <expr>
/// <for-cstyle>   ::= [ <expr> ] ";" [ <expr> ] ";" [ <expr> ] [ DO ] <expr>
/// <for-in>       ::= <decl> IN <expr> [ DO ] <expr>
/// <for-enum-in>  ::= ENUM <identifier> [ "," <decl> ] IN <expr> [ DO ] <expr>
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
        if (Consume(Tk::Comma)) control = ParseDecl(false, curr_loc);
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

auto src::Parser::ParseMatchExpr() -> Result<MatchExpr*> {
    Assert(false, "TODO: figure out the syntax for this");
}

auto src::Parser::ParseParamDeclList(
    SmallVectorImpl<src::ParamDecl*>& params,
    SmallVectorImpl<src::Expr*>* param_types,
    bool in_struct_template
) -> Location {
    auto loc = curr_loc;
    Assert(Consume(Tk::LParen));

    /// '()' is valid, if redundant.
    if (At(Tk::RParen)) {
        loc = {loc, Next()};
        Diag::Warning(ctx, loc, "Empty parameter list '()' is redundant");
        return loc;
    }

    /// Parse parameter list.
    do {
        Expr* type{};
        std::string param_name{};
        Location param_loc = curr_loc;

        /// Parse optional static and with keywords.
        bool is_static = Consume(Tk::Static);
        bool is_with = Consume(Tk::With);
        if (in_struct_template) {
            if (is_static) Diag::Warning(
                ctx,
                curr_loc,
                "'static' has no effect here as struct template parameters are already implicitly static"
            );
        }

        /// Parse nested signature.
        if (At(Tk::Proc)) {
            auto sig = ParseProcSignature();
            if (IsError(sig)) continue;
            type = (*sig).sig_type;

            /// If the signature does not contain a name, then we can
            /// also put the name after it.
            if (At(Tk::Identifier)) {
                if ((*sig).name.empty()) param_name = tok.text;
                else Error("Parameter cannot have two names; did you forget a comma here?");
                Next();
            } else {
                param_name = (*sig).name;
            }

            param_loc = {param_loc, (*sig).loc};
        }

        /// Parse type and name.
        else {
            auto ty = ParseType();
            if (IsError(ty)) continue;
            type = *ty;
            if (At(Tk::Identifier)) {
                param_name = tok.text;
                param_loc = {param_loc, Next()};
            } else {
                param_loc = {param_loc, type->location};
            }
        }

        /// TODO: Parse default parameter value.

        /// Create the param decl.
        loc = {loc, param_loc};
        if (param_types) param_types->push_back(type);
        params.push_back(new (mod) ParamDecl(
            std::move(param_name),
            type,
            is_static or in_struct_template,
            is_with,
            param_loc
        ));
    } while (Consume(Tk::Comma));

    /// Yeet RParen.
    loc = {loc, curr_loc};
    if (not Consume(Tk::RParen)) Error("Expected ')' after parameter list");
    return loc;
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

auto src::Parser::ParseProcBody(bool is_extern, src::Parser::Signature sig) -> Result<Expr*> {
    /// Create the function declaration.
    const auto scope_index = mod->scopes.size();
    tempset curr_func = new (mod) FunctionDecl(
        std::move(sig.name),
        sig.sig_type,
        mod,
        curr_func,
        std::move(sig.param_decls),
        is_extern ? Linkage::Imported : Linkage::Internal,
        sig.nomangle ? Mangling::None : Mangling::Source,
        sig.loc
    );

    /// Parse the body, if any.
    if (is_extern) {
        if (At(Tk::Assign, Tk::LBrace)) return Error("External function cannot have a body");
    } else {
        auto body = Result<Expr*>::Null();
        if (Consume(Tk::Assign)) body = ParseExprInNewScope();
        else if (At(Tk::LBrace)) body = ParseBlockExpr();
        else return Error("Expected '=' or '{{' after function signature");
        if (IsError(body)) return Diag();
        curr_func->body = *body;
    }

    curr_func->scope = mod->scopes[scope_index];
    return curr_func;
}

/// This is only invoked by the expression parser and thus need
/// not parse a full expression.
///
/// <expr-proc-def> ::= <proc-unqual-sig> <proc-body>
/// <expr-lambda>   ::= <proc-anon-sig> <proc-body>
/// <proc-body>     ::= <expr-block> | "=" <expr>
auto src::Parser::ParseProcExpr(bool is_extern) -> Result<Expr*> {
    auto sig_result = ParseProcSignature();
    if (IsError(sig_result)) return Diag();
    Signature& sig = *sig_result;

    /// If the type of the signature is a function type,
    /// then this is a function declaration.
    if (isa<FunctionType>(sig.sig_type)) return ParseProcBody(is_extern, std::move(sig));

    /// Nomangle has no effect if this is not a function decl.
    if (sig.nomangle) Diag::Warning(
        ctx,
        sig.loc,
        "Attribute 'nomangle' of function type ignored outside of function declaration"
    );

    /// Otherwise, return just the function type.
    return sig.sig_type;
}

/// <proc-unqual-sig> ::= PROC <identifier> <proc-sig-rest>
/// <proc-named-sig>  ::= PROC { <type-qual> } <identifier> <proc-sig-rest>
/// <proc-anon-sig>   ::= PROC { <type-qual> } <proc-sig-rest>
/// <proc-sig-rest>   ::=  [ <param-list> ] [ <proc-attrs> ] [ <proc-return> ]
/// <param-list>      ::= "(" <parameter> { "," <parameter> } [ "," ] ")"
/// <parameter>       ::= <param-decl> | <proc-named-sig> | <proc-anon-sig>
/// <param-decl>      ::= [ STATIC ] [ WITH ] <type> [ <identifier> ] [ "=" <expr> ]
/// <proc-attrs>      ::= "variadic" | "nomangle"
/// <proc-return>     ::= "->" <type>
auto src::Parser::ParseProcSignature() -> Result<Signature> {
    auto loc = curr_loc;
    Assert(Consume(Tk::Proc), "ParseProcExpr() called without 'proc'");

    /// Parse optional type qualifiers.
    SmallVector<TypeQualifier> quals;
    while (At(Tk::Caret, Tk::Ampersand, Tk::Question, Tk::LBrack)) {
        quals.emplace_back(tok.type, nullptr);

        /// Arrays also have a dimension.
        if (At(Tk::LBrack)) {
            Next();
            if (Consume(Tk::RBrack)) continue;
            auto dim = ParseExpr();
            if (IsError(dim)) return Diag();
            loc = {loc, dim->location};
            if (not Consume(Tk::RBrack)) Error("Expected ']' after array dimension");
            quals.back().array_dimension = *dim;
        } else {
            loc = {loc, Next()};
        }
    }

    /// Name is optional.
    std::string name;
    if (At(Tk::Identifier)) {
        name = tok.text;
        loc = {loc, Next()};
    }

    /// Parse parameter list.
    SmallVector<ParamDecl*> params;
    SmallVector<Expr*> param_types;
    if (At(Tk::LParen)) loc = {loc, ParseParamDeclList(params, &param_types, false)};

    /// Helper to handle an attribute.
    const auto ParseAttr = [&](std::string_view name, bool& flag) {
        if (tok.text == name) {
            if (flag) Diag::Warning(ctx, curr_loc, "Duplicated '{}' attribute ignored", name);
            loc = {loc, Next()};
            return flag = true;
        }

        return false;
    };

    /// Parse attributes.
    bool variadic{}, nomangle{};
    while (At(Tk::Identifier) and ( // clang-format off
        ParseAttr("variadic", variadic) or
        ParseAttr("nomangle", nomangle)
    )); // clang-format on

    /// Parse return type.
    auto ret = Result<Expr*>::Null();
    if (Consume(Tk::RArrow)) {
        ret = ParseType();
        if (IsError(ret)) return Diag();
        loc = {loc, ret->location};
    }

    /// Create the function type.
    Type* sig_type = new (mod) FunctionType(
        std::move(param_types),
        *ret ?: Type::Void,
        CallingConv::C,
        variadic,
        curr_func != mod->top_level_func
    );

    /// Apply type qualifiers.
    for (const auto& qual : quals) {
        switch (qual.kind) {
            default: Unreachable();
            case Tk::Caret: sig_type = new (mod) ScopedPointerType(sig_type); break;
            case Tk::Ampersand: sig_type = new (mod) ReferenceType(sig_type); break;
            case Tk::Question: sig_type = new (mod) OptionalType(sig_type); break;
            case Tk::LBrack:
                if (qual.array_dimension) sig_type = new (mod) ArrayType(sig_type, qual.array_dimension);
                else sig_type = new (mod) SliceType(sig_type);
                break;
        }
    }

    /// Create the signature.
    return Signature{
        sig_type,
        std::move(name),
        std::move(params),
        loc,
        nomangle,
    };
}

/// <struct-body>     ::= "{" { ( <decl> | <variant-clause> ) ";" } "}" | ";"
/// <variant-clause>  ::= <variant-inline> | <variant-named> | <variant-void>
/// <variant-inline>  ::= [ DYNAMIC ] VARIANT <identifier> <struct-body>
/// <variant-named>   ::= [ DYNAMIC ] VARIANT <identifier> <type> ";"
/// <variant-void>    ::= [ DYNAMIC ] VARIANT VOID
void src::Parser::ParseStructBody(
    bool dynamic,
    SmallVectorImpl<src::MemberDecl*>& members,
    SmallVectorImpl<FunctionDecl*>& member_functions,
    SmallVectorImpl<src::VariantClauseDecl*>& variants
) {
    Assert(Consume(Tk::LBrace));
    while (not At(Tk::RBrace, Tk::Eof)) {
        if (Consume(Tk::Semicolon)) continue;
        auto member_loc = curr_loc;

        /// Variant clause.
        if (const auto dyn_clause = Consume(Tk::Dynamic); dyn_clause or At(Tk::Variant)) {
            std::string clause_name;
            Expr* type = Type::Void;

            /// Dynamic is redundant if the entire variant is redundant.
            if (dyn_clause and dynamic) Diag::Warning(
                ctx,
                curr_loc,
                "'dynamic' is redundant here as the struct itself is already dynamic."
            );

            /// Make sure we don’t have 'dynamic' w/o 'variant'
            if (not Consume(Tk::Variant)) Error(member_loc, "Expected 'variant' after 'dynamic'");

            /// Parse clause name and type.
            if (At(Tk::Void)) {
                member_loc = {member_loc, Next()};
            } else if (At(Tk::Identifier)) {
                clause_name = tok.text;
                member_loc = {member_loc, Next()};

                /// Type may be inline struct body or explicit type.
                if (At(Tk::LBrace)) {
                    ScopeRAII sc{this};
                    RecordType::MemberDecls clause_members;
                    SmallVector<FunctionDecl*> clause_member_functions;
                    SmallVector<src::VariantClauseDecl*> clause_variants;
                    ParseStructBody(dynamic, clause_members, clause_member_functions, clause_variants);
                    type = new (mod) StructType(
                        mod,
                        "",
                        sc.scope,
                        std::move(clause_members),
                        std::move(clause_member_functions),
                        std::move(clause_variants),
                        false,
                        member_loc
                    );
                } else {
                    auto res = ParseType();
                    if (IsError(res)) {
                        Synchronise();
                        continue;
                    }
                    type = *res;
                }
            } else {
                Error("Expected 'void' or identifier after 'variant'");
                Synchronise();
                continue;
            }

            /// Create the variant clause.
            variants.push_back(new (mod) VariantClauseDecl(
                std::move(clause_name),
                type,
                dyn_clause or dynamic,
                member_loc
            ));
        }

        /// Regular decl.
        else {
            auto decl = ParseDecl(false, member_loc);
            if (IsError(decl)) {
                Synchronise();
                continue;
            }

            /// Keep functions as they are. We’ll fix them later in Sema.
            if (auto func = dyn_cast<FunctionDecl>(*decl)) member_functions.push_back(func);

            /// Rewrite variable declarations to member decls.
            else if (auto var = dyn_cast<VarDecl>(*decl)) {
                members.push_back(new (mod) MemberDecl(
                    std::move(var->name),
                    var->type
                ));
            }
        }
    }
}
/// <struct-named>    ::= [ EXPORT ] [ DYNAMIC ] STRUCT <identifier> <struct-rest>
/// <struct-anon>     ::= [ DYNAMIC ] STRUCT <struct-rest>
/// <struct-opaque>   ::= [ EXPORT ] STRUCT <identifier> OPAQUE
/// <struct-rest>     ::= [ <param-list> /* template */ ] <struct-body>
auto src::Parser::ParseStructDecl() -> Result<Expr*> {
    /// Structs may be dynamic; this means that every variant
    /// is a dynamic variant.
    const auto dynamic_loc = curr_loc;
    const bool dynamic = Consume(Tk::Dynamic);
    const auto struct_loc = curr_loc;
    Assert(Consume(Tk::Struct), "ParseStructDecl() called without 'struct'");

    /// Parse name, if any.
    std::string name;
    if (At(Tk::Identifier)) {
        name = tok.text;
        Next();
    }

    /// Struct may be an opaque type.
    if (At(Tk::Identifier) and tok.text == "opaque") {
        if (dynamic) Error(dynamic_loc, "Opaque struct cannot be marked 'dynamic'");
        Next();
        return new (mod) OpaqueType(mod, std::move(name), struct_loc);
    }

    /// Parse template parameter list.
    ScopeRAII sc{this};
    RecordType::MemberDecls members;
    SmallVector<FunctionDecl*> member_functions;
    SmallVector<VariantClauseDecl*> variants;
    SmallVector<ParamDecl*> template_params;
    ParseParamDeclList(template_params, nullptr, true);

    /// Parse the body if there is one. ';' means this is an empty struct;
    /// any other token is an error.
    if (At(Tk::Semicolon)) nop();
    else if (At(Tk::LBrace)) ParseStructBody(dynamic, members, member_functions, variants);
    else Error("Expected 'opaque', ';', or '{{' in struct declaration");

    /// If we have template parameters, create a template.
    if (not template_params.empty()) return new (mod) StructTemplate(
        mod,
        std::move(name),
        sc.scope,
        std::move(members),
        std::move(member_functions),
        std::move(variants),
        std::move(template_params),
        false,
        struct_loc
    );

    /// Otherwise, create a struct.
    else return new (mod) StructType(
        mod,
        std::move(name),
        sc.scope,
        std::move(members),
        std::move(member_functions),
        std::move(variants),
        false,
        struct_loc
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
