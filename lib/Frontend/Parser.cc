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
        case Tk::CBool:
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
/// <expr-assert> ::= [ STATIC ] ASSERT <expr> [ "," <expr> ]
auto src::Parser::ParseAssertExpr(bool is_static) -> Result<AssertExpr*> {
    Assert(Consume(Tk::Assert), "ParseAssertExpr() called without 'assert'");

    /// Parse condition and message, if any.
    auto cond = ParseExpr();
    auto mess = Consume(Tk::Comma) ? ParseExpr() : Result<Expr*>::Null();
    if (IsError(cond, mess)) return Diag();
    return new (mod) AssertExpr(*cond, *mess, is_static);
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
        ParseBlockExpr() >>= Add;
        Consume(Tk::Semicolon);
    }

    /// Expression + semicolon.
    else {
        ParseExpr() >>= Add;
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
auto src::Parser::ParseExprImpl(int operator_precedence) -> Result<Expr*> {
    auto start_token = tok.type;
    auto lhs = Result<Expr*>::Null();
    switch (tok.type) {
        default:
            return Error("Expected expression");

        case Tk::Pragma:
        case Tk::MacroParameter:
            Unreachable();

        case Tk::Module:
        case Tk::Export:
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
            lhs = new (mod) NameRefExpr(tok.text, curr_scope, curr_loc);
            Next();
            break;

        case Tk::Integer:
            lhs = new (mod) IntegerLiteralExpr(tok.integer, curr_loc);
            Next();
            break;

        case Tk::StringLiteral:
            lhs = new (mod) StringLiteralExpr(mod->strtab.intern(tok.text), curr_loc);
            break;

        case Tk::Assert: lhs = ParseAssertExpr(false); break;
        case Tk::Asm: lhs = ParseInlineAsm(); break;
        case Tk::If: lhs = ParseIfExpr(false); break;
        case Tk::Match: lhs = ParseMatchExpr(false); break;
        case Tk::While: lhs = ParseWhileExpr(); break;
        case Tk::For: lhs = ParseForExpr(false); break;
        case Tk::With: break;
        case Tk::Try: break;
        case Tk::Return: break;
        case Tk::Defer: break;
        case Tk::Break: break;
        case Tk::Continue: break;
        case Tk::Unreachable: break;
        case Tk::Extern: break;
        case Tk::Static: break;
        case Tk::Not: break;
        case Tk::True: break;
        case Tk::False: break;
        case Tk::Null: break;
        case Tk::Proc: break;
        case Tk::Var: break;
        case Tk::Val: break;
        case Tk::Enum: break;
        case Tk::Struct: break;
        case Tk::Type: break;
        case Tk::Typeof: break;
        case Tk::NoReturn: break;
        case Tk::Bool: break;
        case Tk::Void: break;
        case Tk::I8: break;
        case Tk::I16: break;
        case Tk::I32: break;
        case Tk::I64: break;
        case Tk::Int: break;
        case Tk::F32: break;
        case Tk::F64: break;
        case Tk::CChar: break;
        case Tk::CChar8T: break;
        case Tk::CChar16T: break;
        case Tk::CChar32T: break;
        case Tk::CWCharT: break;
        case Tk::CShort: break;
        case Tk::CInt: break;
        case Tk::CLong: break;
        case Tk::CLongLong: break;
        case Tk::CLongDouble: break;
        case Tk::CBool: break;
        case Tk::CSizeT: break;
        case Tk::StringKw: break;
        case Tk::LParen: break;
        case Tk::LBrack: break;
        case Tk::LBrace: break;
        case Tk::Dot: break;
        case Tk::LArrow: break;
        case Tk::Plus: break;
        case Tk::Minus: break;
        case Tk::Caret: break;
        case Tk::Ampersand: break;
        case Tk::Tilde: break;
        case Tk::MinusMinus: break;
        case Tk::PlusPlus: break;
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
}

/// <expr-for>     ::= [ STATIC ] FOR ( <for-infinite> | <for-each> | <for-cstyle> | <for-in> | <for-enum-in> )
/// <for-infinite> ::= DO <expr> | <expr-block>
/// <for-each>     ::= <expr> <delim-expr>
/// <for-cstyle>   ::= [ <expr> ] ";" [ <expr> ] ";" [ <expr> ] <delim-expr>
/// <for-in>       ::= <decl-base> IN <expr> <delim-expr>
/// <for-enum-in>  ::= ENUM <identifier> [ "," <decl-base> ] IN <expr> <delim-expr>
auto src::Parser::ParseForExpr(bool is_static) -> Result<Expr*> {
    auto for_loc = curr_loc;
    Assert(Consume(Tk::For), "ParseForExpr() called without 'for'");
    ScopeRAII sc{this};

    /// Parse a C-style for loop.
    const auto ParseForCStyle = [&] (Result<Expr*> init) -> Result<Expr*> {
        if (not Consume(Tk::Semicolon)) Error("Expected ';' after first clause of c-style for loop");
        auto cond = At(Tk::Semicolon) ? Result<Expr*>::Null() : ParseExpr();
        if (not Consume(Tk::Semicolon)) Error("Expected ';' after condition of c-style for loop");
        auto step = AtStartOfExpression() ? ParseExpr() : Result<Expr*>::Null();
        auto body = ParseDelimExpr();
        if (IsError(init, cond, step, body)) return Diag();
        return new (mod) ForCStyleExpr(*init, *cond, *step, *body, for_loc);
    };

    /// Parse a for-in loop.
    const auto ParseForIn = [&](std::string enum_name, Result<Expr*> decl) -> Result<Expr*> {
        /// Create a variable for the enumerator.
        bool has_enum = not enum_name.empty();
        auto enum_decl = Result<Expr*>::Null();
        if (has_enum) {
            enum_decl = VarDecl::Create(
                std::move(enum_name),
                curr_scope,
                curr_func,
                nullptr,
                Linkage::Local,
                Mangling::Source
            );
        }

        /// Parse 'in' keyword, range, and body.
        if (not Consume(Tk::In)) Error("Expected 'in' after enumerator in for-in loop");
        auto range = ParseExpr();
        auto body = ParseDelimExpr();

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
    auto body = ParseDelimExpr();
    if (IsError(control, body)) return Diag();
    return new (mod) ForInExpr(nullptr, nullptr, *control, *body, for_loc);
}

/// <expr-if> ::= [ STATIC ] IF <expr> <delim-expr> { ELIF <expr> <delim-expr> } [ ELSE <expr> ]
auto src::Parser::ParseIfExpr(bool is_static) -> Result<IfExpr*> {
    auto if_loc = curr_loc;
    Assert(
        Consume(Tk::If) or Consume(Tk::Elif),
        "ParseIfExpr() called without 'if' or 'elif'"
    );

    /// Parse condition, body, and else clause.
    auto cond = ParseExpr();
    auto body = ParseDelimExpr();
    auto elif = At(Tk::Elif) ? ParseIfExpr(false) : Consume(Tk::Else) ? ParseExpr()
                                                                      : Result<Expr*>::Null();
    if (IsError(cond, body, elif)) return Diag();
    return new (mod) IfExpr(*cond, *body, *elif, is_static, if_loc);
}

/// <expr-asm> ::= ASM "{" { <asm-instruction> } "}" | ASM <asm-instruction>
/// <asm-instruction> ::= [ <identifier> ":" ] <identifier> <asm-operands> NEWLINE
/// <asm-operands> ::= [ <expr> ] { "," <expr> }
auto src::Parser::ParseInlineAsm() -> Result<Expr*> {
    Diag::ICE("TODO: Implement parsing of inline assembly");
}

/// <expr-match>    ::= [ STATIC ] MATCH <match-control> "{" { <match-case> } [ ELSE <delim-expr> ] "}"
/// <match-control> ::= [ <expr> [ <binary> ] ]
/// <match-case>    ::= <expr> <delim-expr>
auto src::Parser::ParseMatchExpr(bool is_static) -> Result<MatchExpr*> {
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
        auto expr = ParseExpr();
        auto body = ParseDelimExpr();
        auto fallthrough = Consume(Tk::Fallthrough);
        if (IsError(expr, body)) return Diag(); /// Stop to avoid infinite loop.
        cases.push_back(new (mod) CaseExpr(*expr, *body, fallthrough, expr->location));
    }

    /// Parse else clause, if any.
    auto else_body = Consume(Tk::Else) ? ParseDelimExpr() : Result<Expr*>::Null();
    if (not Consume(Tk::RBrace)) Error("Expected '}}' at end of match body");

    /// Create the expression.
    if (IsError(control, else_body)) return Diag();
    return new (mod) MatchExpr(
        *control,
        op,
        std::move(cases),
        else_body ? new (mod) CaseExpr(nullptr, *else_body, false, else_body->location) : nullptr,
        is_static,
        loc
    );
}

/// <expr-while> ::= WHILE <expr> <delim-expr>
auto src::Parser::ParseWhileExpr() -> Result<WhileExpr*> {
    auto loc = curr_loc;
    Assert(Consume(Tk::While), "ParseWhileExpr() called without 'while'");

    /// Parse condition and body.
    auto cond = ParseExpr();
    auto body = ParseDelimExpr();
    if (IsError(cond, body)) return Diag();
    return new (mod) WhileExpr(*cond, *body, loc);
}
