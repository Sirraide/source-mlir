#include <frontend/ast.hh>
#include <frontend/parser.hh>
#include <memory>

Parser::Parser(Context* ctx, File& f) : Lexer(ctx, f) {}

auto Parser::Parse(Context* ctx, File& f) -> std::unique_ptr<Module> {
    Parser p{ctx, f};
    p.ParseFile();
    return ctx->has_error() ? nullptr : std::move(p.mod_ptr);
}

bool Parser::MayStartAnExpression(Tk k) {
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

/// <exprs> ::= { <expr-block> | <expr> ";" }
void Parser::ParseExpressions(ExprList& into) {
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
void Parser::ParseFile() {
    /// Set up scopes.
    scope_stack.emplace_back(new Scope{nullptr, mod});
    scope_stack.emplace_back(new Scope{global_scope, mod});

    /// Parse preamble; this also creates the module.
    ParsePreamble();

    /// Get the decl context for top-level declarations.
    decl_context = &mod->top_level_func->body->exprs;

    /// Parse expressions.
    while (not At(Tk::Eof)) ParseExpressions(*decl_context);
}

/// <preamble>    ::= [ <module-decl> ] { <import> }
/// <module-decl> ::= MODULE <identifier> ";"
/// <module-name> ::= <identifier> | "<" TOKENS ">"
/// <import>      ::= IMPORT <module-name> [ "." "*" ] [ AS <identifier> ] ";"
void Parser::ParsePreamble() {
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

auto Parser::ParseExpr(int current_precedence) -> Result<Expr*> {
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

        case Tk::Identifier:
            lhs = new (mod) NameRefExpr(tok.text, curr_scope);
            Next();
            break;

        case Tk::Integer: break;
        case Tk::StringLiteral: break;
        case Tk::Assert: break;
        case Tk::Asm: break;
        case Tk::If: break;
        case Tk::Match: break;
        case Tk::While: break;
        case Tk::For: break;
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

    /// Make sure that the rest of the parser knows that this
    /// token can start an expression.
    Assert(
        MayStartAnExpression(start_token),
        "Add {} to MayStartAnExpression()",
        Spelling(start_token)
    );
}
