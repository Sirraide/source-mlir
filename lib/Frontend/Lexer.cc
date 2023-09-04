#include <source/Frontend/Lexer.hh>

/// ===========================================================================
///  Lexer — Helpers and Data.
/// ===========================================================================
namespace src {
namespace {
/// Check if a character is allowed at the start of an identifier.
constexpr bool IsStart(char c) {
    return std::isalpha(c) or c == '_' or c == '$';
}

/// Check if a character is allowed in an identifier.
constexpr bool IsContinue(char c) {
    return IsStart(c) or isdigit(c);
}

constexpr bool IsBinary(char c) { return c == '0' or c == '1'; }
constexpr bool IsDecimal(char c) { return c >= '0' and c <= '9'; }
constexpr bool IsOctal(char c) { return c >= '0' and c <= '7'; }
constexpr bool IsHex(char c) { return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F'); }

/// All keywords.
const StringMap<Tk> keywords = {
    {"module", Tk::Module},
    {"export", Tk::Export},
    {"import", Tk::Import},
    {"pragma", Tk::Pragma},
    {"assert", Tk::Assert},
    {"asm", Tk::Asm},
    {"if", Tk::If},
    {"then", Tk::Then},
    {"elif", Tk::Elif},
    {"else", Tk::Else},
    {"match", Tk::Match},
    {"while", Tk::While},
    {"do", Tk::Do},
    {"for", Tk::For},
    {"in", Tk::In},
    {"with", Tk::With},
    {"try", Tk::Try},
    {"return", Tk::Return},
    {"defer", Tk::Defer},
    {"break", Tk::Break},
    {"continue", Tk::Continue},
    {"fallthrough", Tk::Fallthrough},
    {"unreachable", Tk::Unreachable},
    {"variant", Tk::Variant},
    {"extern", Tk::Extern},
    {"static", Tk::Static},
    {"is", Tk::Is},
    {"as", Tk::As},
    {"not", Tk::Not},
    {"and", Tk::And},
    {"or", Tk::Or},
    {"xor", Tk::Xor},
    {"true", Tk::True},
    {"false", Tk::False},
    {"null", Tk::Null},
    {"proc", Tk::Proc},
    {"var", Tk::Var},
    {"val", Tk::Val},
    {"enum", Tk::Enum},
    {"struct", Tk::Struct},
    {"init", Tk::Init},
    {"type", Tk::Type},
    {"typeof", Tk::Typeof},
    {"noreturn", Tk::NoReturn},
    {"bool", Tk::Bool},
    {"void", Tk::Void},
    {"i8", Tk::I8},
    {"i16", Tk::I16},
    {"i32", Tk::I32},
    {"i64", Tk::I64},
    {"int", Tk::Int},
    {"f32", Tk::F32},
    {"f64", Tk::F64},
    {"string", Tk::StringKw},
    {"__c_char", Tk::CChar},
    {"__c_short", Tk::CShort},
    {"__c_int", Tk::CInt},
    {"__c_long", Tk::CLong},
    {"__c_longlong", Tk::CLongLong},
    {"__c_longdouble", Tk::CLongDouble},
    {"__c_wchar", Tk::CWCharT},
    {"__c_char16", Tk::CChar16T},
    {"__c_char32", Tk::CChar32T},
    {"__c_size_t", Tk::CSizeT},
    {"__c_bool", Tk::CBool},
};
} // namespace
} // namespace src
/// ========================================================================
///  Main lexer implementation.
/// ========================================================================
src::Lexer::Lexer(Context* ctx, File& f)
    : ctx{ctx}, f{f} {
    /// Init state.
    tok.location.file_id = u16(f.file_id());
    curr = f.data();
    end = curr + f.size();

    /// Define predefined macros.
    AllocateMacroDefinition(
        "__filename__",
        {},
        {Token::Make(Tk::StringLiteral, f.path().filename().string())}
    );

    AllocateMacroDefinition(
        "__directory__",
        {},
        {Token::Make(Tk::StringLiteral, f.path().parent_path().string())}
    );

    /// Init first token.
    NextChar();
    Next();
}

auto src::Lexer::CurrLoc() const -> Location { return {CurrOffs(), 1, u16(f.file_id())}; }
auto src::Lexer::CurrOffs() const -> u32 { return u32(curr - f.data()) - 1; }

auto src::Lexer::LookAhead(usz n) -> Token& {
    if (n == 0) return tok;

    /// If we already have enough tokens, just return the nth token.
    const auto idx = n - 1;
    if (idx < lookahead_tokens.size()) return lookahead_tokens[idx];

    /// Otherwise, lex enough tokens.
    auto current = std::move(tok);
    for (usz i = lookahead_tokens.size(); i < n; i++) {
        tok = {};
        Next();
        lookahead_tokens.push_back(std::move(tok));
    }
    tok = std::move(current);

    /// Return the nth token.
    return lookahead_tokens[idx];
}

void src::Lexer::Next() {
    /// Tokens are not artificial by default.
    tok.artificial = false;

    /// Pop lookahead tokens if we’re not looking ahead.
    if (not looking_ahead and not lookahead_tokens.empty()) {
        tok = std::move(lookahead_tokens.front());
        lookahead_tokens.pop_front();
        return;
    }

    /// Pop empty macro expansions off the expansion stack.
    while (not macro_expansion_stack.empty())
        if (macro_expansion_stack.back().done())
            macro_expansion_stack.pop_back();

    /// Insert tokens from macro expansion.
    if (not macro_expansion_stack.empty()) {
        auto& expansion = macro_expansion_stack.back();
        tok = ++expansion;

        /// If this token is another macro definition, handle it.
        if (tok.type == Tk::Identifier and tok.text == "macro") LexMacroDefinition();
        return;
    }

    /// Skip whitespace.
    SkipWhitespace();

    /// Keep returning EOF if we're at EOF.
    if (lastc == 0) {
        tok.type = Tk::Eof;
        return;
    }

    /// Reset the token. We set the token type to 'invalid' here so that,
    /// if we encounter an error, we can just issue a diagnostic and return
    /// without setting the token type. The parser will then stop because
    /// it encounters an invalid token.
    tok.artificial = false;
    tok.type = Tk::Invalid;
    tok.location.pos = CurrOffs();

    /// Lex the token.
    switch (lastc) {
        case '\\':
            LexEscapedId();
            return;

        case ';':
            NextChar();
            tok.type = Tk::Semicolon;
            break;

        case ':':
            NextChar();
            if (lastc == ':') {
                NextChar();
                tok.type = Tk::ColonColon;
            } else {
                tok.type = Tk::Colon;
            }
            break;

        case ',':
            NextChar();
            tok.type = Tk::Comma;
            break;

        case '?':
            NextChar();
            tok.type = Tk::Question;
            break;

        case '(':
            NextChar();
            tok.type = Tk::LParen;
            break;

        case ')':
            NextChar();
            tok.type = Tk::RParen;
            break;

        case '[':
            NextChar();
            tok.type = Tk::LBrack;
            break;

        case ']':
            NextChar();
            tok.type = Tk::RBrack;
            break;

        case '{':
            NextChar();
            tok.type = Tk::LBrace;
            break;

        case '}':
            NextChar();
            tok.type = Tk::RBrace;
            break;

        /// Syntax of this token is `#<identifier>` or `#<number>`.
        case '#': {
            auto l = CurrLoc();
            NextChar();
            Next();

            /// Validate name.
            if (tok.type == Tk::Identifier or tok.type == Tk::Integer) {
                if (tok.type == Tk::Integer) tok.text = std::to_string(tok.integer);
                tok.type = Tk::MacroParameter;
                tok.location = {l, tok.location};
            } else {
                Error(CurrLoc(), "Expected identifier or integer after '#'");
            }

            /// Check if this token is even allowed here.
            if (not in_macro_definition) Error(tok.location, "Unexpected macro parameter outside of macro definition");
        } break;

        case '.':
            NextChar();
            if (lastc == '.') {
                NextChar();
                if (lastc == '.') {
                    NextChar();
                    tok.type = Tk::Ellipsis;
                } else if (lastc == '<') {
                    NextChar();
                    tok.type = Tk::DotDotLess;
                } else if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::DotDotEq;
                } else {
                    tok.type = Tk::DotDot;
                }
            } else {
                tok.type = Tk::Dot;
            }
            break;

        case '-':
            NextChar();
            if (lastc == '>') {
                NextChar();
                tok.type = Tk::RArrow;
            } else if (lastc == '-') {
                NextChar();
                tok.type = Tk::MinusMinus;
            } else if (lastc == '=') {
                NextChar();
                tok.type = Tk::MinusEq;
            } else {
                tok.type = Tk::Minus;
            }
            break;

        case '+':
            NextChar();
            if (lastc == '+') {
                NextChar();
                tok.type = Tk::PlusPlus;
            } else if (lastc == '=') {
                NextChar();
                tok.type = Tk::PlusEq;
            } else {
                tok.type = Tk::Plus;
            }
            break;

        case '*':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::StarEq;
            } else if (lastc == '*') {
                NextChar();
                if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::StarStarEq;
                } else {
                    tok.type = Tk::StarStar;
                }
            } else {
                tok.type = Tk::Star;
            }
            break;

        case '/':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::SlashEq;
            } else if (lastc == '/') {
                SkipLine();
                return Next();
            } else {
                tok.type = Tk::Slash;
            }
            break;

        case '%':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::PercentEq;
            } else {
                tok.type = Tk::Percent;
            }
            break;

        case '^':
            NextChar();
            tok.type = Tk::Caret;
            break;

        case '&':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::AmpersandEq;
            } else {
                tok.type = Tk::Ampersand;
            }
            break;

        case '|':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::VBarEq;
            } else {
                tok.type = Tk::VBar;
            }
            break;

        case '~':
            NextChar();
            tok.type = Tk::Tilde;
            break;

        case '!':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::Neq;
            } else {
                tok.type = Tk::Bang;
            }
            break;

        case '=':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::EqEq;
            } else {
                tok.type = Tk::Assign;
            }
            break;

        case '<':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::Le;
            } else if (lastc == '<') {
                NextChar();
                if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::ShiftLeftEq;
                } else {
                    tok.type = Tk::ShiftLeft;
                }
            } else if (lastc == '-') {
                NextChar();
                tok.type = Tk::LArrow;
            } else {
                tok.type = Tk::Lt;
            }
            break;

        case '>':
            NextChar();
            if (lastc == '=') {
                NextChar();
                tok.type = Tk::Ge;
            } else if (lastc == '>') {
                NextChar();
                if (lastc == '>') {
                    NextChar();
                    if (lastc == '=') {
                        NextChar();
                        tok.type = Tk::ShiftRightLogicalEq;
                    } else {
                        tok.type = Tk::ShiftRightLogical;
                    }
                } else if (lastc == '=') {
                    NextChar();
                    tok.type = Tk::ShiftRightEq;
                } else {
                    tok.type = Tk::ShiftRight;
                }
            } else {
                tok.type = Tk::Gt;
            }
            break;

        case '"':
        case '\'':
            LexString(lastc);
            break;

        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            LexNumber();
            break;

        default:
            if (IsStart(lastc)) LexIdentifier();
            else {
                Error(CurrLoc() << 1, "Unexpected <U+{:X}> character in program", lastc);
                return;
            }
    }

    /// Set the end of the token.
    tok.location.len = u16(u64(curr - f.data()) - tok.location.pos - 1);
    if (curr == end and not lastc) tok.location.len++;
}

void src::Lexer::NextChar() {
    if (curr == end) {
        lastc = 0;
        return;
    }

    lastc = *curr++;

    /// Collapse CR LF and LF CR to a single newline,
    /// but keep CR CR and LF LF as two newlines.
    if (lastc == '\r' || lastc == '\n') {
        /// Two newlines in a row.
        if (curr != end && (*curr == '\r' || *curr == '\n')) {
            bool same = lastc == *curr;
            lastc = '\n';

            /// CR CR or LF LF.
            if (same) return;

            /// CR LF or LF CR.
            curr++;
        }

        /// Either CR or LF followed by something else.
        lastc = '\n';
    }
}

void src::Lexer::SkipLine() {
    while (lastc != '\n' && lastc != 0) NextChar();
}

void src::Lexer::SkipWhitespace() {
    while (std::isspace(lastc)) NextChar();
}

void src::Lexer::LexIdentifier() {
    tok.type = Tk::Identifier;
    tok.text.clear();
    do {
        tok.text += lastc;
        NextChar();
    } while (IsContinue(lastc));

    /// In raw mode, special processing is disabled. This is used for
    /// parsing the argument and expansion lists of macros, as well as
    /// for handling __id.
    if (raw_mode) {
        if (auto k = keywords.find(tok.text); k != keywords.end()) tok.type = k->second;
        return;
    }

    /// Handle macro expansions.
    if (auto m = macro_definitions_by_name.find(tok.text); m != macro_definitions_by_name.end())
        return LexMacroExpansion(m->second);

    /// Handle macro definitions.
    if (tok.text == "macro") return LexMacroDefinition();

    /// Check for keywords.
    if (auto k = keywords.find(tok.text); k != keywords.end()) tok.type = k->second;

    /// Handle pragmas.
    if (tok.type == Tk::Pragma and not pragma_handler()) tok.type = Tk::Invalid;
}

void src::Lexer::LexNumber() {
    /// Helper function that actually parses a number.
    auto lex_number_impl = [this](bool pred(char), usz conv(char), usz base) {
        /// Need at least one digit.
        if (not pred(lastc)) {
            Error(CurrLoc() << 1 <<= 1, "Invalid integer literal");
            return;
        }

        /// Parse the literal.
        usz value{};
        do {
            usz old_value = value;
            value *= base;

            /// Check for overflow.
            if (value < old_value) {
            overflow:
                /// Consume the remaining digits so we can highlight the entire thing in the error.
                while (pred(lastc)) NextChar();
                Error(Location{tok.location, CurrLoc()} >>= -1, "Integer literal overflow");
                return;
            }

            /// Add the next digit.
            old_value = value;
            value += conv(lastc);
            if (value < old_value) goto overflow;

            /// Yeet it.
            NextChar();
        } while (pred(lastc));

        /// The next character must not be a start character.
        if (IsStart(lastc)) {
            Error(Location{tok.location, CurrLoc()}, "Invalid character in integer literal: '{}'", lastc);
            return;
        }

        /// We have a valid integer literal!
        tok.type = Tk::Integer;
        tok.integer = isz(value);
    };

    /// If the first character is a 0, then this might be a non-decimal constant.
    if (lastc == 0) {
        NextChar();

        /// Hexadecimal literal.
        if (lastc == 'x' or lastc == 'X') {
            NextChar();
            static const auto xctoi = [](char c) -> usz {
                switch (c) {
                    case '0' ... '9': return static_cast<usz>(c - '0');
                    case 'a' ... 'f': return static_cast<usz>(c - 'a');
                    case 'A' ... 'F': return static_cast<usz>(c - 'A');
                    default: Unreachable();
                }
            };
            return lex_number_impl(IsHex, xctoi, 16);
        }

        /// Octal literal.
        if (lastc == 'o' or lastc == 'O') {
            NextChar();
            return lex_number_impl(
                IsOctal,
                [](char c) { return static_cast<usz>(c - '0'); },
                8
            );
        }

        /// Binary literal.
        if (lastc == 'b' or lastc == 'B') {
            NextChar();
            return lex_number_impl(
                IsBinary,
                [](char c) { return static_cast<usz>(c - '0'); },
                2
            );
        }

        /// Multiple leading 0’s are not permitted.
        if (std::isdigit(lastc)) {
            Error(CurrLoc() << 1, "Leading 0 in integer literal. (Hint: Use 0o/0O for octal literals)");
            return;
        }

        /// Integer literal must be a literal 0.
        if (IsStart(lastc)) {
            Error(CurrLoc() <<= 1, "Invalid character in integer literal: '{}'", lastc);
            return;
        }

        /// Integer literal is 0.
        tok.type = Tk::Integer;
        tok.integer = 0;
        return;
    }

    /// If the first character is not 0, then we have a decimal literal.
    return lex_number_impl(
        IsDecimal,
        [](char c) { return static_cast<usz>(c - '0'); },
        10
    );
}

void src::Lexer::LexString(char delim) {
    /// Yeet the delimiter.
    tok.text.clear();
    NextChar();

    /// Lex the string. If it’s a raw string, we don’t need to
    /// do any escaping.
    if (delim == '\'') {
        while (lastc != delim && lastc != 0) {
            tok.text += lastc;
            NextChar();
        }
    }

    /// Otherwise, we also need to replace escape sequences.
    else if (delim == '"') {
        while (lastc != delim && lastc != 0) {
            if (lastc == '\\') {
                NextChar();
                switch (lastc) {
                    case 'a': tok.text += '\a'; break;
                    case 'b': tok.text += '\b'; break;
                    case 'f': tok.text += '\f'; break;
                    case 'n': tok.text += '\n'; break;
                    case 'r': tok.text += '\r'; break;
                    case 't': tok.text += '\t'; break;
                    case 'v': tok.text += '\v'; break;
                    case '\\': tok.text += '\\'; break;
                    case '\'': tok.text += '\''; break;
                    case '"': tok.text += '"'; break;
                    case '0': tok.text += '\0'; break;
                    default:
                        Error({tok.location, CurrLoc()}, "Invalid escape sequence");
                        return;
                }
            } else {
                tok.text += lastc;
            }
            NextChar();
        }
    }

    /// Other string delimiters are invalid.
    else {
        Error(CurrLoc() << 1, "Invalid delimiter: {}", delim);
        return;
    }

    /// Make sure we actually have a delimiter.
    if (lastc != delim) {
        Error(CurrLoc() << 1, "Unterminated string literal");
        return;
    }
    NextChar();

    /// This is a valid string.
    tok.type = Tk::StringLiteral;
}

void src::Lexer::LexEscapedId() {
    /// Yeet backslash.
    tempset raw_mode = true;
    auto start = tok.location;
    Next();

    /// Mark this token as ‘artificial’. This is so we can e.g. nest
    /// macro definitions using `\expands` and `\endmacro`.
    tok.artificial = true;

    /// If the next token is anything other than "(", then it becomes the name.
    if (tok.type != Tk::LParen) {
        tok.type = Tk::Identifier;
        tok.text = Spelling(tok.type);
        tok.location = {start, tok.location};
        return;
    }

    /// If the token is "(", then everything up to the next ")" is the name.
    tok.type = Tk::Identifier;
    tok.text.clear();
    while (lastc != ')' and lastc != 0) {
        tok.text += lastc;
        NextChar();
    }

    /// EOF.
    if (lastc == 0) {
        Error(start, "EOF reached while lexing \\(...");
        tok.type = Tk::Invalid;
        return;
    }

    /// Skip the ")".
    tok.location = {start, tok.location};
    NextChar();
}

/// ========================================================================
///  Macros.
/// ========================================================================
auto src::Lexer::AllocateMacroDefinition(
    std::string name,
    Location location,
    SmallVector<Token>&& expansion
) -> Macro& {
    auto& m = macro_definitions.emplace_back();
    macro_definitions_by_name[m.name = std::move(name)] = &m;
    m.location = location;
    m.expansion = std::move(expansion);
    return m;
}

/// <macro-definition> ::= MACRO <identifier> <tokens> EXPANDS <tokens> ENDMACRO
void src::Lexer::LexMacroDefinition() {
    tempset raw_mode = true;
    tempset in_macro_definition = true;
    Next(); /// Yeet 'macro'.

    /// Make sure we can define this macro.
    if (tok.type != Tk::Identifier) {
        Error(tok.location, "Expected identifier");

        /// Synchronise on 'endmacro'.
        while (tok.type != Tk::Eof and (tok.type != Tk::Identifier or tok.text != "endmacro")) Next();
        return;
    }

    if (macro_definitions_by_name.find(tok.text) == macro_definitions_by_name.end())
        Error(tok.location, "Macro '{}' is already defined. Use `undef` to undefine it first.", tok.text);

    /// Allocate a new macro.
    auto& m = AllocateMacroDefinition(tok.text, tok.location);
    Next();

    /// Helper to issue an error and print the current macro definition.
    auto Err = [&]<typename... arguments>(fmt::format_string<arguments...> fmt, arguments&&... args) {
        Diag::Note(ctx, m.location, "In definition of macro '{}' here", m.name);
        Error(tok.location, fmt, std::forward<arguments>(args)...);
    };

    /// Collect macro parameters to make sure they exist and that there aren’t any duplicates.
    std::unordered_set<std::string> parameter_names;

    /// Read tokens until we encounter 'expands'.
    for (;;) {
        switch (tok.type) {
            case Tk::Eof: return Err("EOF reached while reading macro definition");

            /// Parameter must not exist.
            case Tk::MacroParameter:
                if (parameter_names.contains(tok.text)) return Err("Duplicate macro parameter '{}'", tok.text);
                parameter_names.insert(tok.text);
                goto param_list_default_case;

            /// Check for 'expands'.
            case Tk::Identifier:
                if (tok.text == "expands" and not tok.artificial) goto expands;
                [[fallthrough]];

            /// Add the token to the parameter list.
            default:
            param_list_default_case:
                m.parameter_list.push_back(tok);
                Next();
        }
    }

    /// Yeet 'expands'.
expands:
    Next();

    /// Read tokens until we encounter 'endmacro'.
    for (;;) {
        switch (tok.type) {
            case Tk::Eof: return Err("EOF reached while reading macro definition");

            /// Parameter must exist.
            case Tk::MacroParameter:
                if (not parameter_names.contains(tok.text)) return Err("Undefined macro parameter '{}'", tok.text);
                goto expansion_default_case;

            /// Check for 'endmacro'.
            case Tk::Identifier:
                if (tok.text == "endmacro" and not tok.artificial) goto endmacro;
                [[fallthrough]];

            /// Add the token to the expansion.
            default:
            expansion_default_case:
                m.expansion.push_back(tok);
                Next();
        }
    }

    /// Yeet 'endmacro' and lex the next token.
endmacro:
    raw_mode = false;
    in_macro_definition = false;
    Next();
}

void src::Lexer::LexMacroExpansion(Lexer::Macro* m) {
    StringMap<Token> args;
    auto loc = tok.location;

    /// Disable macro expansion etc.
    tempset raw_mode = true;

    /// Read tokens according to the definition of the macro parameter list
    /// and bind macro parameters to their values.
    for (auto& t : m->parameter_list) {
        /// Yeet the previous token. We do this here to make sure
        /// that we don’t discard past the last token of the param
        /// list, since the next token after that must come from the
        /// expansion itself.
        Next();

        /// EOF is not a token.
        if (tok.type == Tk::Eof) {
            Error(
                tok.location,
                "EOF reached while reading parameters of macro '{}'",
                m->name
            );
            return;
        }

        /// Handle macro parameters.
        if (t.type == Tk::MacroParameter) {
            args[t.text] = tok;
            continue;
        }

        /// Make sure the token matches the expected token.
        if (tok != t) Error(
            tok.location,
            "Expected {} token in expansion of macro '{}', got {}",
            Spelling(t.type),
            m->name,
            Spelling(tok.type)
        );
    }

    /// Push the expansion onto the stack.
    macro_expansion_stack.emplace_back(*this, *m, std::move(args), loc);

    /// Reënable macro expansion and read the next token.
    raw_mode = false;
    Next();
}

auto src::Lexer::MacroExpansion::operator++() -> Token {
    Token ret;
    Assert(not done());

    /// If the token is a macro arg, get the bound argument.
    if (it->type == Tk::MacroParameter) {
        auto arg = bound_parameters.find(it->text);
        Assert(arg != bound_parameters.end(), "Unbound macro argument '{}'", it->text);
        it++;
        ret = arg->second;
    }

    /// Otherwise, return a copy of the token.
    else { ret = *it++; }

    /// Mark the token as non-artificial, because, for example,
    /// if we are inserting "endmacro", then we want the inserted
    /// identifier to *not* be artificial so we actually end up
    /// closing a macro definition.
    ret.artificial = false;
    return ret;
}
