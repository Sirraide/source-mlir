#include <source/Frontend/AST.hh>
#include <source/Support/Utils.hh>

namespace {
src::BuiltinType IntTypeInstance{src::BuiltinTypeKind::Int, {}};
src::BuiltinType UnknownTypeInstance{src::BuiltinTypeKind::Unknown, {}};
src::BuiltinType VoidTypeInstance{src::BuiltinTypeKind::Void, {}};
} // namespace
src::Expr* const src::detail::UnknownType = &UnknownTypeInstance;
src::BuiltinType* const src::Type::Int = &IntTypeInstance;
src::BuiltinType* const src::Type::Void = &VoidTypeInstance;
src::BuiltinType* const src::Type::Unknown = &UnknownTypeInstance;

/// ===========================================================================
///  Expressions
/// ===========================================================================
auto src::Expr::_type() -> Expr* {
    switch (kind) {
        /// Typed exprs.
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::MemberAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::CastExpr:
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

/// ===========================================================================
///  Types
/// ===========================================================================
auto src::Expr::type_str(bool use_colour) const -> std::string {
    using enum utils::Colour;
    utils::Colours C{use_colour};
    std::string out{C(Cyan)};

    /// Helper to write a type that has an element type.
    const auto WriteSElem = [&](std::string_view suffix) {
        out += cast<SingleElementTypeBase>(this)->elem->type_str(use_colour);
        out += C(Red);
        out += suffix;
    };

    switch (kind) {
        case Kind::ReferenceType: WriteSElem("&"); break;
        case Kind::ScopedPointerType: WriteSElem("^"); break;
        case Kind::OptionalType: WriteSElem("?"); break;
        case Kind::SliceType: WriteSElem("[]"); break;
        case Kind::IntType: out += fmt::format("i{}", cast<IntType>(this)->bits); break;

        case Kind::BuiltinType: {
            switch (cast<BuiltinType>(this)->builtin_kind) {
                case BuiltinTypeKind::Unknown: out += "<unknown>"; break;
                case BuiltinTypeKind::Void: out += "void"; break;
                case BuiltinTypeKind::Int: out += "int"; break;
            }
        } break;

        case Kind::FFIType: {
            switch (cast<FFIType>(this)->ffi_kind) {
                case FFITypeKind::CChar: out += "__ffi_char"; break;
                case FFITypeKind::CInt: out += "__ffi_int"; break;
            }
        } break;

        case Kind::ProcType: {
            auto p = cast<ProcType>(this);
            out += fmt::format("{}proc", C(Red));

            if (not p->param_types.empty()) {
                out += " (";
                for (usz i = 0; i < p->param_types.size(); i++) {
                    out += p->param_types[i]->type_str(use_colour);
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
                out += p->ret_type->type_str(use_colour);
            }
        } break;

        /// Typed exprs.
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::MemberAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::CastExpr:
            return cast<TypedExpr>(this)->stored_type->type_str(use_colour);
    }

    out += C(Reset);
    return out;
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

        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::MemberAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ParamDecl:
        case Kind::ProcDecl:
        case Kind::CastExpr:
            Unreachable();
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
        const Expr* node,
        const Expr* type
    ) {
        PrintBasicHeader(node_name, node);

        /// Print the type if there is one.
        if (type) out += fmt::format(" {}", type->type_str(use_colour));
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
                    f->type_str(use_colour)
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
                    i->type_str(use_colour)
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
                        i->type_str(use_colour)
                    );
                } else {
                    out += fmt::format(" {}\n", i->type_str(use_colour));
                }
                return;
            }

            case K::DeclRefExpr: {
                auto n = cast<DeclRefExpr>(e);
                PrintBasicHeader("NameRefExpr", e);
                out += fmt::format(
                    " {}{} {}\n",
                    C(White),
                    n->name,
                    n->type_str(use_colour)
                );
                return;
            }

            case K::ParamDecl: {
                auto n = cast<ParamDecl>(e);
                PrintBasicHeader("ParamDecl", e);
                out += fmt::format(
                    " {}{} {}\n",
                    C(Blue),
                    n->name,
                    n->type_str(use_colour)
                );
                return;
            }

            case K::CastExpr: {
                auto c = cast<CastExpr>(e);
                PrintBasicHeader("CastExpr", e);
                out += C(Red);
                switch (c->cast_kind) {
                    case CastKind::LValueToRValue: out += " LValueToRValue"; break;
                }
                out += fmt::format(
                    " {}\n",
                    c->type_str(use_colour)
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
                    " {} {}{}\n",
                    m->type_str(use_colour),
                    C(Magenta),
                    m->member
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
                PrintBasicNode(R"(<type>)", e, e);
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
                PrintChildren(children, leading_text);
            } break;

            case K::BlockExpr:
                PrintChildren(cast<BlockExpr>(e)->exprs, leading_text);
                break;

            case K::MemberAccessExpr: {
                auto m = cast<MemberAccessExpr>(e);
                PrintChildren(m->object, leading_text);
            } break;

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
