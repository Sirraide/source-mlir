#include <source/Frontend/AST.hh>
#include <source/Support/Utils.hh>

namespace {
src::BuiltinType IntTypeInstance{src::BuiltinTypeKind::Int, {}};
src::BuiltinType UnknownTypeInstance{src::BuiltinTypeKind::Unknown, {}};
src::BuiltinType VoidTypeInstance{src::BuiltinTypeKind::Void, {}};
src::BuiltinType BoolTypeInstance{src::BuiltinTypeKind::Bool, {}};
src::BuiltinType NoReturnTypeInstance{src::BuiltinTypeKind::NoReturn, {}};
} // namespace
src::Expr* const src::detail::UnknownType = &UnknownTypeInstance;
src::BuiltinType* const src::Type::Int = &IntTypeInstance;
src::BuiltinType* const src::Type::Void = &VoidTypeInstance;
src::BuiltinType* const src::Type::Unknown = &UnknownTypeInstance;
src::BuiltinType* const src::Type::Bool = &BoolTypeInstance;
src::BuiltinType* const src::Type::NoReturn = &NoReturnTypeInstance;

/// ===========================================================================
///  Expressions
/// ===========================================================================
auto src::Expr::_type() -> TypeHandle {
    switch (kind) {
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
            return Type::NoReturn;

        case Kind::AssertExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
            return Type::Void;

        /// Typed exprs.
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::MemberAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::CastExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::VarDecl:
            return cast<TypedExpr>(this)->stored_type;

        /// Already a type.
        case Kind::BuiltinType:
        case Kind::FFIType:
        case Kind::ReferenceType:
        case Kind::ScopedPointerType:
        case Kind::OptionalType:
        case Kind::ProcType:
        case Kind::IntType:
        case Kind::SliceType:
            return this;
    }
}

auto src::ProcDecl::_ret_type() -> TypeHandle {
    return cast<ProcType>(stored_type)->ret_type;
}

/// ===========================================================================
///  Types
/// ===========================================================================
auto src::Expr::TypeHandle::align([[maybe_unused]] src::Context* ctx) -> isz {
    switch (ptr->kind) {
        case Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::Unknown: return 8;
                case BuiltinTypeKind::Void: return 8;
                case BuiltinTypeKind::Int: return 64; /// FIXME: Use context.
                case BuiltinTypeKind::Bool: return 8;
                case BuiltinTypeKind::NoReturn: return 1; /// Alignment can’t be 0.
            }

            Unreachable();

        case Kind::FFIType:
            switch (cast<FFIType>(ptr)->ffi_kind) {
                case FFITypeKind::CChar: return 8;
                case FFITypeKind::CInt: return 32; /// FIXME: Use context.
            }

            Unreachable();

        case Kind::IntType:
            /// FIXME: Use context.
            return isz(std::min<usz>(64, std::bit_ceil(usz(cast<IntType>(ptr)->bits))));

        case Kind::ReferenceType:
            return 64; /// FIXME: Use context.

        case Kind::ScopedPointerType:
            return 64; /// FIXME: Use context.

        case Kind::SliceType:
            return 64; /// FIXME: Use context.

        case Kind::OptionalType:
            Todo();

        /// Invalid.
        case Kind::ProcType:
            Unreachable(".align accessed on function type");

        case Kind::DeclRefExpr:
            Todo();

        case Kind::AssertExpr:
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::CastExpr:
        case Kind::MemberAccessExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::VarDecl:
            Unreachable(".align accessed on non-type expression");
    }

    Unreachable();
}

bool src::Expr::TypeHandle::is_int(bool bool_is_int) {
    switch (ptr->kind) {
        default: return false;
        case Kind::IntType: return true;
        case Kind::BuiltinType: {
            auto k = cast<BuiltinType>(ptr)->builtin_kind;
            if (k == BuiltinTypeKind::Int) return true;
            return bool_is_int and k == BuiltinTypeKind::Bool;
        }
    }
}

bool src::Expr::TypeHandle::_is_noreturn() {
    return Type::Equal(ptr, Type::NoReturn);
}

auto src::Expr::TypeHandle::_ref_depth() -> isz {
    isz depth = 0;
    for (
        auto th = *this;
        isa<ReferenceType>(th);
        th = cast<ReferenceType>(th)->elem
    ) depth++;
    return depth;
}

auto src::Expr::TypeHandle::size([[maybe_unused]] src::Context* ctx) -> isz {
    switch (ptr->kind) {
        case Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::Unknown: return 0;
                case BuiltinTypeKind::Void: return 0;
                case BuiltinTypeKind::Int: return 64; /// FIXME: Use context.
                case BuiltinTypeKind::Bool: return 1;
                case BuiltinTypeKind::NoReturn: return 0;
            }

            Unreachable();

        case Kind::FFIType:
            switch (cast<FFIType>(ptr)->ffi_kind) {
                case FFITypeKind::CChar: return 1;
                case FFITypeKind::CInt: return 32; /// FIXME: Use context.
            }

            Unreachable();

        case Kind::IntType:
            return cast<IntType>(ptr)->bits;

        case Kind::ReferenceType:
            return 64; /// FIXME: Use context.

        case Kind::ScopedPointerType:
            return 64; /// FIXME: Use context.

        case Kind::SliceType:
            return 128; /// FIXME: Use context.

        case Kind::OptionalType:
            Todo();

        /// Invalid.
        case Kind::ProcType:
            Unreachable(".size accessed on function type");

        case Kind::DeclRefExpr:
            Todo();

        case Kind::AssertExpr:
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::CastExpr:
        case Kind::MemberAccessExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::VarDecl:
            Unreachable(".size accessed on non-type expression");
    }

    Unreachable();
}

auto src::Expr::TypeHandle::str(bool use_colour) const -> std::string {
    using enum utils::Colour;
    utils::Colours C{use_colour};
    std::string out{C(Cyan)};

    /// Helper to write a type that has an element type.
    const auto WriteSElem = [&](std::string_view suffix) {
        out += cast<SingleElementTypeBase>(ptr)->elem->type.str(use_colour);
        out += C(Red);
        out += suffix;
    };

    switch (ptr->kind) {
        case Kind::ReferenceType: WriteSElem("&"); break;
        case Kind::ScopedPointerType: WriteSElem("^"); break;
        case Kind::OptionalType: WriteSElem("?"); break;
        case Kind::SliceType: WriteSElem("[]"); break;
        case Kind::IntType: out += fmt::format("i{}", cast<IntType>(ptr)->bits); break;

        case Kind::BuiltinType: {
            auto bk = cast<BuiltinType>(ptr)->builtin_kind;
            switch (bk) {
                case BuiltinTypeKind::Unknown: out += "<unknown>"; goto done;
                case BuiltinTypeKind::Void: out += "void"; goto done;
                case BuiltinTypeKind::Int: out += "int"; goto done;
                case BuiltinTypeKind::Bool: out += "bool"; goto done;
                case BuiltinTypeKind::NoReturn: out += "noreturn"; goto done;
            }

            out += fmt::format("<invalid builtin type: {}>", int(bk));
        } break;

        case Kind::FFIType: {
            switch (cast<FFIType>(ptr)->ffi_kind) {
                case FFITypeKind::CChar: out += "__ffi_char"; goto done;
                case FFITypeKind::CInt: out += "__ffi_int"; goto done;
            }

            out += fmt::format("<invalid ffi type: {}>", int(cast<FFIType>(ptr)->ffi_kind));
        } break;

        case Kind::ProcType: {
            auto p = cast<ProcType>(ptr);
            out += fmt::format("{}proc", C(Red));

            if (not p->param_types.empty()) {
                out += " (";
                for (usz i = 0; i < p->param_types.size(); i++) {
                    out += p->param_types[i]->type.str(use_colour);
                    if (i != p->param_types.size() - 1) out += fmt::format("{}, ", C(Red));
                }
                out += C(Red);
                out += ")";
            }

            /// Avoid relying on operator==() for this.
            if (
                not isa<BuiltinType>(p->ret_type) or
                cast<BuiltinType>(p->ret_type)->builtin_kind != BuiltinTypeKind::Void
            ) {
                out += " -> ";
                out += p->ret_type->type.str(use_colour);
            }
        } break;

        case Kind::AssertExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
            return Type::Void->as_type.str(use_colour);

        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
            return Type::NoReturn->as_type.str(use_colour);

        /// Typed exprs.
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::MemberAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::CastExpr:
        case Kind::IfExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::BinaryExpr:
        case Kind::VarDecl:
            return cast<TypedExpr>(ptr)->stored_type->type.str(use_colour);
    }

done:
    out += C(Reset);
    return out;
}

auto src::Expr::TypeHandle::_strip_refs() -> TypeHandle {
    if (auto r = dyn_cast<ReferenceType>(ptr)) return r->elem->as_type.strip_refs;
    return *this;
}

bool src::Expr::TypeHandle::_yields_value() {
    return not Type::Equal(ptr, Type::Void) and not Type::Equal(ptr, Type::NoReturn);
}

bool src::Type::Equal(Expr* a, Expr* b) {
    /// Non-types are never equal.
    if (not isa<Type>(a) or not isa<Type>(b)) return false;

    /// Types of different kinds are never equal.
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case Kind::BuiltinType:
            return cast<BuiltinType>(a)->builtin_kind == cast<BuiltinType>(b)->builtin_kind;

        case Kind::FFIType:
            return cast<FFIType>(a)->ffi_kind == cast<FFIType>(b)->ffi_kind;

        case Kind::IntType:
            return cast<IntType>(a)->bits == cast<IntType>(b)->bits;

        case Kind::ReferenceType:
        case Kind::ScopedPointerType:
        case Kind::SliceType:
        case Kind::OptionalType: {
            return Type::Equal(
                cast<SingleElementTypeBase>(a)->elem,
                cast<SingleElementTypeBase>(b)->elem
            );
        }

        case Kind::ProcType: {
            auto pa = cast<ProcType>(a);
            auto pb = cast<ProcType>(b);

            if (pa->param_types.size() != pb->param_types.size()) return false;
            for (auto [p1, p2] : llvm::zip_equal(pa->param_types, pb->param_types))
                if (not Type::Equal(p1, p2))
                    return false;

            return Type::Equal(pa->ret_type, pb->ret_type);
        }

        case Kind::AssertExpr:
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::MemberAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::CastExpr:
        case Kind::IfExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::BinaryExpr:
        case Kind::VarDecl:
            Unreachable("Not a type");
    }

    Unreachable();
}

/// ===========================================================================
///  Scope
/// ===========================================================================
void* src::Scope::operator new(size_t sz, src::Module* mod) noexcept {
    return utils::AllocateAndRegister<Scope>(sz, mod->scopes);
}

/// ===========================================================================
///  AST Printing
/// ===========================================================================
namespace src {
namespace {
struct ASTPrinter {
    using enum utils::Colour;

    Module* mod;
    DenseSet<ProcDecl*> printed_functions{};
    bool print_children_of_children = true;
    std::string out;
    bool use_colour = true;
    utils::Colours C{use_colour};

    ASTPrinter(Module* mod, bool use_colour) : mod{mod}, use_colour{use_colour} {}
    ~ASTPrinter() {
        if (not out.empty()) fmt::print("{}{}", out, C(Reset));
    }

    /// Print basic information about an AST node.
    void PrintBasicNode(
        std::string_view node_name,
        Expr* node,
        Expr* type
    ) {
        PrintBasicHeader(node_name, node);

        /// Print the type if there is one.
        if (type) out += fmt::format(" {}", type->type.str(use_colour));
        if (node->is_lvalue) out += fmt::format(" {}lvalue", C(Blue));
        out += fmt::format("{}\n", C(Reset));
    }

    /// Print the start of the header of an AST node.
    /// Example: IfExpr 0xdeadbeef <0>
    void PrintBasicHeader(std::string_view node_name, const Expr* node) {
        out += fmt::format(
            "{}{} {}{} {}<{}>",
            C(Red),
            node_name,
            C(Blue),
            fmt::ptr(node),
            C(Magenta),
            node->location.pos
        );
    }

    /// Print the linkage of a node.
    void PrintLinkage(Linkage l) {
        out += C(Red);
        switch (l) {
            case Linkage::Local: out += "Local "; return;
            case Linkage::Internal: out += "Internal "; return;
            case Linkage::Exported: out += "Exported "; return;
            case Linkage::Imported: out += "Imported "; return;
            case Linkage::Reexported: out += "Reexported "; return;
        }
        Unreachable();
    }

    /// Print the children of a node.
    void PrintChildren(ArrayRef<Expr*> exprs, std::string leading_text) {
        for (usz i = 0; i < exprs.size(); i++) {
            const bool last = i == exprs.size() - 1;

            /// Print the leading text.
            out += fmt::format("{}{}{}", C(Red), leading_text, last ? "└─" : "├─");

            /// Print the child.
            operator()(exprs[i], leading_text + (last ? "  " : "│ "));
        }
    }

    /// Print the header (name + location + type) of a node.
    void PrintHeader(Expr* e) {
        using K = Expr::Kind;
        switch (e->kind) {
            case K::ProcDecl: {
                auto f = cast<ProcDecl>(e);
                PrintLinkage(f->linkage);
                PrintBasicHeader("FuncDecl", e);
                out += fmt::format(
                    " {}{} {}\n",
                    C(Green),
                    f->name,
                    f->type.str(use_colour)
                );
                return;
            }

            case K::VarDecl: {
                auto v = cast<VarDecl>(e);
                PrintLinkage(v->linkage);
                PrintBasicHeader("VarDecl", e);
                out += fmt::format(
                    " {}{} {}{} lvalue\n",
                    C(White),
                    v->name,
                    v->type.str(use_colour),
                    C(Blue)
                );
                return;
            }

            case K::BoolLiteralExpr: {
                auto i = cast<BoolLitExpr>(e);
                PrintBasicHeader("BoolLiteral", e);
                out += fmt::format(
                    " {}{} {}\n",
                    C(Red),
                    i->value ? "true" : "false",
                    i->type.str(use_colour)
                );
                return;
            }

            case K::IntegerLiteralExpr: {
                auto i = cast<IntLitExpr>(e);
                PrintBasicHeader("IntegerLiteral", e);
                out += fmt::format(
                    " {}{} {}\n",
                    C(Magenta),
                    i->value,
                    i->type.str(use_colour)
                );
                return;
            }

            case K::StringLiteralExpr: {
                auto i = cast<StrLitExpr>(e);
                PrintBasicHeader("StringLiteral", e);
                if (mod) {
                    out += fmt::format(
                        " {}\"{}\" {}\n",
                        C(Yellow),
                        mod->strtab[i->index],
                        i->type.str(use_colour)
                    );
                } else {
                    out += fmt::format(" {}\n", i->type.str(use_colour));
                }
                return;
            }

            case K::DeclRefExpr: {
                auto n = cast<DeclRefExpr>(e);
                PrintBasicHeader("NameRefExpr", e);
                out += fmt::format(
                    " {}{} {}{}{}\n",
                    C(White),
                    n->name,
                    n->type.str(use_colour),
                    C(Blue),
                    n->is_lvalue ? " lvalue" : ""
                );
                return;
            }

            case K::ParamDecl: {
                auto n = cast<ParamDecl>(e);
                PrintBasicHeader("ParamDecl", e);
                out += fmt::format(
                    " {}{} {} {}lvalue\n",
                    C(Blue),
                    n->name,
                    n->type.str(use_colour),
                    C(Blue)
                );
                return;
            }

            case K::CastExpr: {
                auto c = cast<CastExpr>(e);
                PrintBasicHeader("CastExpr", e);
                out += C(Red);
                switch (c->cast_kind) {
                    case CastKind::Implicit: out += " Implicit"; break;
                    case CastKind::ReferenceToLValue: out += " ReferenceToLValue"; break;
                    case CastKind::LValueRefToLValue: out += " LValueRefToLValue"; break;
                    case CastKind::LValueToRValue: out += " LValueToRValue"; break;
                    case CastKind::LValueToReference: out += " LValueToReference"; break;
                    case CastKind::Soft: out += " Soft"; break;
                    case CastKind::Hard: out += " Hard"; break;
                }
                out += fmt::format(
                    " {}{}{}\n",
                    c->type.str(use_colour),
                    C(Blue),
                    c->is_lvalue ? " lvalue" : ""
                );
                return;
            }

            case K::BlockExpr:
                if (cast<BlockExpr>(e)->implicit) out += fmt::format("{}Implicit ", C(Red));
                PrintBasicNode("BlockExpr", e, e->type);
                return;

            case K::InvokeExpr: PrintBasicNode("InvokeExpr", e, e->type); return;
            case K::MemberAccessExpr: {
                auto m = cast<MemberAccessExpr>(e);
                PrintBasicHeader("MemberAccessExpr", e);
                out += fmt::format(
                    " {} {}{}{}{}\n",
                    m->type.str(use_colour),
                    C(Magenta),
                    m->member,
                    C(Blue),
                    m->is_lvalue ? " lvalue" : ""
                );
                return;
            }

            case K::LoopControlExpr: {
                auto l = cast<LoopControlExpr>(e);
                PrintBasicHeader("LoopControlExpr", e);
                out += fmt::format(
                    " {}{}\n",
                    C(Yellow),
                    l->label.empty() ? "<parent>" : l->label
                );
                return;
            }

            case K::WhileExpr: {
                auto w = cast<WhileExpr>(e);
                PrintBasicHeader("WhileExpr", e);
                if (not w->label.empty()) {
                    out += fmt::format(
                        " {}{}\n",
                        C(Yellow),
                        w->label
                    );
                } else {
                    out += '\n';
                }
                return;
            }
            case K::ReturnExpr: PrintBasicNode("ReturnExpr", e, nullptr); return;
            case K::DeferExpr: PrintBasicNode("DeferExpr", e, nullptr); return;
            case K::AssertExpr: PrintBasicNode("AssertExpr", e, nullptr); return;
            case K::IfExpr: PrintBasicNode("IfExpr", e, e->type); return;

            case K::UnaryPrefixExpr: {
                auto u = cast<UnaryPrefixExpr>(e);
                PrintBasicHeader("UnaryPrefixExpr", e);
                out += fmt::format(
                    " {}{} {}{}{}\n",
                    C(Red),
                    Spelling(u->op),
                    u->type.str(use_colour),
                    C(Blue),
                    u->is_lvalue ? " lvalue" : ""
                );
                return;
            }

            case K::BinaryExpr: {
                auto b = cast<BinaryExpr>(e);
                PrintBasicHeader("BinaryExpr", e);
                out += fmt::format(
                    " {}{} {}{}{}\n",
                    C(Red),
                    Spelling(b->op),
                    b->type.str(use_colour),
                    C(Blue),
                    b->is_lvalue ? " lvalue" : ""
                );
                return;
            }

            /// We don’t print types here.
            case K::ProcType:
            case K::BuiltinType:
            case K::FFIType:
            case K::ReferenceType:
            case K::ScopedPointerType:
            case K::OptionalType:
            case K::IntType:
            case K::SliceType:
                PrintBasicNode("Type", e, e);
                return;
        }

        PrintBasicNode(R"(<???>)", e, e->type);
    }

    void PrintNodeChildren(const Expr* e, std::string leading_text = "") {
        if (not print_children_of_children) return;

        /// Print the children of a node.
        using K = Expr::Kind;
        switch (e->kind) {
            /// We only print function bodies at the top level.
            case K::ProcDecl: break;

            /// These don’t have children.
            case K::BoolLiteralExpr:
            case K::IntegerLiteralExpr:
            case K::StringLiteralExpr:
            case K::ParamDecl:
                break;

            /// We don’t print types here.
            case K::BuiltinType:
            case K::ProcType:
            case K::FFIType:
            case K::ReferenceType:
            case K::ScopedPointerType:
            case K::OptionalType:
            case K::IntType:
            case K::SliceType:
                break;

            case K::DeclRefExpr: {
                auto n = cast<DeclRefExpr>(e);
                if (n->decl) {
                    tempset print_children_of_children = false;
                    PrintChildren(n->decl, leading_text);
                }
            } break;

            case K::InvokeExpr: {
                auto c = cast<InvokeExpr>(e);
                SmallVector<Expr*, 12> children{c->callee};
                children.insert(children.end(), c->args.begin(), c->args.end());
                if (c->init) children.push_back(c->init);
                PrintChildren(children, leading_text);
            } break;

            case K::IfExpr: {
                auto i = cast<IfExpr>(e);
                SmallVector<Expr*, 3> children{{i->cond, i->then}};
                if (i->else_) children.push_back(i->else_);
                PrintChildren(children, leading_text);
            } break;

            case K::AssertExpr: {
                auto i = cast<AssertExpr>(e);
                SmallVector<Expr*, 2> children{i->cond};
                if (i->msg) children.push_back(i->msg);
                PrintChildren(children, leading_text);
            } break;

            case K::VarDecl: {
                auto v = cast<VarDecl>(e);
                if (v->init) PrintChildren(v->init, leading_text);
            } break;

            case K::MemberAccessExpr: {
                auto m = cast<MemberAccessExpr>(e);
                PrintChildren(m->object, leading_text);
            } break;

            case K::BinaryExpr: {
                auto b = cast<BinaryExpr>(e);
                PrintChildren({b->lhs, b->rhs}, leading_text);
            } break;

            case K::WhileExpr: {
                auto w = cast<WhileExpr>(e);
                PrintChildren({w->cond, w->body}, leading_text);
            } break;

            case K::ReturnExpr: {
                auto r = cast<ReturnExpr>(e);
                if (r->value) PrintChildren(r->value, leading_text);
            } break;

            case K::LoopControlExpr: {
                tempset print_children_of_children = false;
                auto r = cast<LoopControlExpr>(e);
                if (r->target) PrintChildren(r->target, leading_text);
            } break;

            case K::DeferExpr:
                PrintChildren(cast<DeferExpr>(e)->expr, leading_text);
                break;

            case K::UnaryPrefixExpr:
                PrintChildren(cast<UnaryPrefixExpr>(e)->operand, leading_text);
                break;

            case K::BlockExpr:
                PrintChildren(cast<BlockExpr>(e)->exprs, leading_text);
                break;

            case K::CastExpr:
                PrintChildren(cast<CastExpr>(e)->operand, leading_text);
                break;
        }
    }

    /// Print a top-level node.
    void PrintTopLevelNode(Expr* e) {
        PrintHeader(e);
        if (auto f = dyn_cast<ProcDecl>(e)) {
            printed_functions.insert(f);
            if (auto body = llvm::dyn_cast_if_present<BlockExpr>(f->body))
                PrintChildren(body->exprs, "");
        } else {
            PrintNodeChildren(e);
        }
    }

    /// Print a node.
    void operator()(Expr* e, std::string leading_text = "") {
        PrintHeader(e);
        PrintNodeChildren(e, std::move(leading_text));
    }

    void print() {
        if (mod) {
            printed_functions.insert(mod->top_level_func);
            for (auto* node : cast<BlockExpr>(mod->top_level_func->body)->exprs)
                PrintTopLevelNode(node);

            for (auto* f : mod->functions)
                if (not printed_functions.contains(f))
                    PrintTopLevelNode(f);
        }
    }
};
} // namespace
} // namespace src

void src::Expr::print() const {
    /// Ok because ASTPrinter does not attempt to mutate this.
    ASTPrinter{nullptr, true}(const_cast<Expr*>(this));
}

void src::Module::print_ast() const {
    /// Ok because ASTPrinter does not attempt to mutate this.
    ASTPrinter{const_cast<Module*>(this), true}.print();
}
