#include <source/Frontend/AST.hh>
#include <source/Support/Utils.hh>

namespace src {
namespace {
BuiltinType IntTypeInstance{BuiltinTypeKind::Int, {}};
BuiltinType UnknownTypeInstance{BuiltinTypeKind::Unknown, {}};
BuiltinType VoidTypeInstance{BuiltinTypeKind::Void, {}};
BuiltinType BoolTypeInstance{BuiltinTypeKind::Bool, {}};
BuiltinType NoReturnTypeInstance{BuiltinTypeKind::NoReturn, {}};
BuiltinType OverloadSetTypeInstance{BuiltinTypeKind::OverloadSet, {}};
BuiltinType MemberProcTypeInstance{BuiltinTypeKind::MemberProc, {}};
BuiltinType ArrayLiteralTypeInstance{BuiltinTypeKind::ArrayLiteral, {}};
IntType IntType8Instance{Size::Bits(8), {}};
IntType IntType16Instance{Size::Bits(16), {}};
IntType IntType32Instance{Size::Bits(32), {}};
IntType IntType64Instance{Size::Bits(64), {}};
Nil NilInstance{{}};
ReferenceType VoidRefTypeInstance = [] {
    ReferenceType ty {&VoidTypeInstance, {}};
    ty.sema.set_done();
    return ty;
}();
ReferenceType VoidRefRefTypeInstance = [] {
    ReferenceType ty {&VoidRefTypeInstance, {}};
    ty.sema.set_done();
    return ty;
}();
} // namespace
constinit const Type Type::Int = &IntTypeInstance;
constinit const Type Type::Void = &VoidTypeInstance;
constinit const Type Type::Unknown = &UnknownTypeInstance;
constinit const Type Type::Bool = &BoolTypeInstance;
constinit const Type Type::NoReturn = &NoReturnTypeInstance;
constinit const Type Type::OverloadSet = &OverloadSetTypeInstance;
constinit const Type Type::MemberProc = &MemberProcTypeInstance;
constinit const Type Type::ArrayLiteral = &ArrayLiteralTypeInstance;
constinit const Type Type::VoidRef = &VoidRefTypeInstance;
constinit const Type Type::VoidRefRef = &VoidRefRefTypeInstance;
constinit const Type Type::I8 = &IntType8Instance;
constinit const Type Type::I16 = &IntType16Instance;
constinit const Type Type::I32 = &IntType32Instance;
constinit const Type Type::I64 = &IntType64Instance;
constinit const Type Type::Nil = &NilInstance;
} // namespace src

/// ===========================================================================
///  Expressions
/// ===========================================================================
src::ProcDecl::ProcDecl(
    Module* mod,
    ProcDecl* parent,
    std::string name,
    Type type,
    SmallVector<ParamDecl*> param_decls,
    Linkage linkage,
    Mangling mangling,
    Location loc
) : ObjectDecl(Kind::ProcDecl, mod, std::move(name), type, linkage, mangling, loc),
    parent(parent),
    params(std::move(param_decls)),
    body(nullptr) {
    module->add_function(this);
    for (auto param : params) param->parent = this;
}

src::DeclRefExpr::DeclRefExpr(Decl* referenced, Location loc)
    : TypedExpr(Kind::DeclRefExpr, referenced->type, loc),
      name(referenced->name),
      decl(referenced) {
    sema = referenced->sema;
    is_lvalue = referenced->is_lvalue;
}

src::LocalRefExpr::LocalRefExpr(ProcDecl* parent, LocalDecl* decl, Location loc)
    : TypedExpr(Kind::LocalRefExpr, decl->type, loc),
      parent(parent),
      decl(decl) {
    is_lvalue = true;
}

src::LabelExpr::LabelExpr(
    ProcDecl* in_procedure,
    std::string label,
    Expr* expr,
    Location loc
) : Expr(Kind::LabelExpr, loc),
    label(std::move(label)),
    expr(expr) {
    in_procedure->add_label(this->label, this);
}

void src::LocalDecl::set_captured() {
    if (is_captured) return;
    is_captured = true;
    parent->captured_locals.push_back(this);
}

auto src::Expr::_ignore_lv2rv() -> Expr* {
    if (auto c = dyn_cast<CastExpr>(this); c and c->cast_kind == CastKind::LValueToRValue)
        return c->operand;
    return this;
}

auto src::Expr::_ignore_parens() -> Expr* {
    if (auto p = dyn_cast<ParenExpr>(this)) return p->expr->ignore_parens;
    return this;
}

auto src::Expr::_ignore_paren_cast_refs() -> Expr* {
    if (auto p = dyn_cast<ParenExpr>(this)) return p->expr->ignore_paren_cast_refs;
    if (auto d = dyn_cast<DeclRefExpr>(this)) return d->decl->ignore_paren_cast_refs;
    if (auto d = dyn_cast<LocalRefExpr>(this)) return d->decl->ignore_paren_cast_refs;
    if (auto c = dyn_cast<CastExpr>(this)) return c->operand->ignore_paren_cast_refs;
    return this;
}

auto src::Expr::_ignore_paren_refs() -> Expr* {
    if (auto p = dyn_cast<ParenExpr>(this)) return p->expr->ignore_paren_refs;
    if (auto d = dyn_cast<DeclRefExpr>(this)) return d->decl->ignore_paren_refs;
    if (auto d = dyn_cast<LocalRefExpr>(this)) return d->decl->ignore_paren_refs;
    return this;
}

bool src::Expr::_is_nil() {
    return isa<Nil>(this);
}

auto src::Expr::_scope_name() -> std::string {
    switch (kind) {
        case Kind::AliasExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::ArrayType:
        case Kind::AssertExpr:
        case Kind::BinaryExpr:
        case Kind::BlockExpr:
        case Kind::BoolLiteralExpr:
        case Kind::BuiltinType:
        case Kind::CastExpr:
        case Kind::ClosureType:
        case Kind::ConstExpr:
        case Kind::ConstructExpr:
        case Kind::DeclRefExpr:
        case Kind::DeferExpr:
        case Kind::EmptyExpr:
        case Kind::FieldDecl:
        case Kind::ForInExpr:
        case Kind::GotoExpr:
        case Kind::IfExpr:
        case Kind::ImplicitThisExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::IntType:
        case Kind::InvokeBuiltinExpr:
        case Kind::InvokeExpr:
        case Kind::LabelExpr:
        case Kind::LocalDecl:
        case Kind::LocalRefExpr:
        case Kind::LoopControlExpr:
        case Kind::MemberAccessExpr:
        case Kind::Nil:
        case Kind::OpaqueType:
        case Kind::OptionalType:
        case Kind::OverloadSetExpr:
        case Kind::ParamDecl:
        case Kind::ParenExpr:
        case Kind::ProcType:
        case Kind::ReferenceType:
        case Kind::ReturnExpr:
        case Kind::ScopedPointerType:
        case Kind::SliceType:
        case Kind::StringLiteralExpr:
        case Kind::SubscriptExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::WhileExpr:
            return "<?>";

        case Kind::ModuleRefExpr:
            return cast<ModuleRefExpr>(this)->module->name;

        case Kind::ExportExpr:
            return cast<ExportExpr>(this)->expr->scope_name;

        case Kind::SugaredType:
            return Type(cast<SugaredType>(this)).desugared->scope_name;

        case Kind::ScopedType: {
            auto s = cast<ScopedType>(this);
            return fmt::format("{}::{}", s->object->scope_name, s->name);
        }

        case Kind::ScopeAccessExpr: {
            auto s = cast<ScopeAccessExpr>(this);
            return fmt::format("{}::{}", s->object->scope_name, s->element);
        }

        case Kind::StructType: {
            auto s = cast<StructType>(this);
            if (s->name.empty()) return "<anonymous>";
            if (s->module->is_logical_module) return fmt::format("{}::{}", s->module->name, s->name);
            return s->name;
        }

        case Kind::ProcDecl: {
            auto p = cast<ProcDecl>(this);
            if (p->name.empty()) return "<anonymous>";
            if (p->parent != p->module->top_level_func) return fmt::format("{}::{}", p->parent->scope_name, p->name);
            if (p->module->is_logical_module) return fmt::format("{}::{}", p->module->name, p->name);
            return p->name;
        }
    }

    Unreachable();
}

auto src::Expr::_type() -> Type {
    switch (kind) {
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
            return Type::NoReturn;

        case Kind::OverloadSetExpr:
            return Type::OverloadSet;

        case Kind::ArrayLiteralExpr:
            return Type::ArrayLiteral;

        case Kind::AliasExpr:
        case Kind::AssertExpr:
        case Kind::ConstructExpr:
        case Kind::DeferExpr:
        case Kind::EmptyExpr:
        case Kind::ExportExpr:
        case Kind::ForInExpr:
        case Kind::LabelExpr:
        case Kind::ModuleRefExpr:
        case Kind::WhileExpr:
            return Type::Void;

        /// Typed exprs.
        case Kind::BinaryExpr:
        case Kind::BlockExpr:
        case Kind::BoolLiteralExpr:
        case Kind::CastExpr:
        case Kind::ConstExpr:
        case Kind::DeclRefExpr:
        case Kind::FieldDecl:
        case Kind::IfExpr:
        case Kind::ImplicitThisExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::InvokeExpr:
        case Kind::LocalDecl:
        case Kind::LocalRefExpr:
        case Kind::MemberAccessExpr:
        case Kind::ParamDecl:
        case Kind::ParenExpr:
        case Kind::ProcDecl:
        case Kind::ScopeAccessExpr:
        case Kind::StringLiteralExpr:
        case Kind::SubscriptExpr:
        case Kind::UnaryPrefixExpr:
            return cast<TypedExpr>(this)->stored_type;

        /// Already a type.
        case Kind::ArrayType:
        case Kind::BuiltinType:
        case Kind::ClosureType:
        case Kind::IntType:
        case Kind::Nil:
        case Kind::OpaqueType:
        case Kind::OptionalType:
        case Kind::ProcType:
        case Kind::ReferenceType:
        case Kind::ScopedPointerType:
        case Kind::ScopedType:
        case Kind::SliceType:
        case Kind::StructType:
        case Kind::SugaredType:
            return Type(this);
    }
}

auto src::ProcDecl::_ret_type() -> Type {
    return cast<ProcType>(stored_type)->ret_type;
}

bool src::ProcDecl::_takes_static_chain() {
    return cast<ProcType>(stored_type)->static_chain_parent;
}

/// ===========================================================================
///  Types
/// ===========================================================================
src::StructType::StructType(
    Module* mod,
    std::string sname,
    SmallVector<FieldDecl*> fields,
    SmallVector<ProcDecl*> inits,
    MemberProcedures member_procs,
    ProcDecl* deleter,
    BlockExpr* scope,
    Mangling mangling,
    Location loc
) : NamedType(Kind::StructType, mod, std::move(sname), mangling, loc),
    all_fields(std::move(fields)),
    initialisers(std::move(inits)),
    member_procs(std::move(member_procs)),
    deleter(deleter),
    scope(scope) {
    if (not name.empty()) mod->named_structs.push_back(this);
}

auto src::Type::align(Context* ctx) const -> Align {
    switch (ptr->kind) {
        case Expr::Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::Bool:
                    return Align(1);

                case BuiltinTypeKind::Int:
                    return ctx->align_of_int;

                /// Alignment can’t be 0.
                case BuiltinTypeKind::ArrayLiteral:
                case BuiltinTypeKind::MemberProc:
                case BuiltinTypeKind::NoReturn:
                case BuiltinTypeKind::OverloadSet:
                case BuiltinTypeKind::Unknown:
                case BuiltinTypeKind::Void:
                    return Align(1);
            }

            Unreachable();

        case Expr::Kind::IntType:
            /// FIXME: Use context.
            return Align(std::max<usz>(1, std::bit_ceil(usz(cast<IntType>(ptr)->size.bytes()))));

        case Expr::Kind::Nil:
            return Align(1);

        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
            return ctx->align_of_pointer;

        case Expr::Kind::SliceType:
        case Expr::Kind::ClosureType:
            return std::max(Type::VoidRef.align(ctx), Type::Int.align(ctx));

        case Expr::Kind::ArrayType:
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem->type.align(ctx);

        case Expr::Kind::StructType:
            return cast<StructType>(ptr)->stored_alignment;

        case Expr::Kind::OptionalType: {
            auto opt = cast<OptionalType>(ptr);
            if (isa<ReferenceType>(opt->elem)) return opt->elem.align(ctx);
            Todo();
        }

        case Expr::Kind::OpaqueType:
            Diag::ICE("Cannot get alignment of opaque type '{}'", str(true));

        /// Invalid.
        case Expr::Kind::ProcType:
            Unreachable(".align accessed on function type");

        SOURCE_NON_TYPE_EXPRS:
            Unreachable(".align accessed on non-type expression");
    }

    Unreachable();
}

auto src::Type::_callable() -> ProcType* {
    if (auto p = dyn_cast<ProcType>(ptr)) return p;
    if (auto p = dyn_cast<ClosureType>(ptr)) return p->proc_type;
    Unreachable("Type '{}' is not callable", str(true));
}

auto src::Type::_default_constructor() -> ProcDecl* {
    auto d = desugared;
    if (auto s = dyn_cast<StructType>(d.ptr)) {
        auto c = rgs::find_if(
            s->initialisers,
            [](auto init) { return init->params.empty(); }
        );

        if (c == s->initialisers.end()) return nullptr;
        return *c;
    }

    return nullptr;
}

auto src::Type::_desugared() const -> Type {
    if (auto s = dyn_cast<SugaredType>(ptr)) return s->elem.desugared;
    if (auto s = dyn_cast<ScopedType>(ptr)) return s->elem.desugared;
    return *this;
}

bool src::Type::is_int(bool bool_is_int) {
    if (isa<SugaredType, ScopedType>(ptr)) return desugared.is_int(bool_is_int);
    switch (ptr->kind) {
        default: return false;
        case Expr::Kind::IntType: return true;
        case Expr::Kind::BuiltinType: {
            auto k = cast<BuiltinType>(ptr)->builtin_kind;
            if (k == BuiltinTypeKind::Int) return true;
            return bool_is_int and k == BuiltinTypeKind::Bool;
        }
    }
}

bool src::Type::_is_nil() {
    return isa<src::Nil>(ptr);
}

bool src::Type::_is_noreturn() {
    return *this == NoReturn;
}

auto src::Type::_ref_depth() -> isz {
    isz depth = 0;
    for (
        auto th = *this;
        isa<ReferenceType, ScopedPointerType>(th);
        th = cast<SingleElementTypeBase>(th)->elem
    ) depth++;
    return depth;
}

auto src::Type::size(Context* ctx) const -> Size {
    switch (ptr->kind) {
        case Expr::Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::ArrayLiteral: return {};
                case BuiltinTypeKind::Bool: return Size::Bits(1);
                case BuiltinTypeKind::Int: return ctx->size_of_int;
                case BuiltinTypeKind::MemberProc: return {};
                case BuiltinTypeKind::NoReturn: return {};
                case BuiltinTypeKind::OverloadSet: return {};
                case BuiltinTypeKind::Unknown: return {};
                case BuiltinTypeKind::Void: return {};
            }

            Unreachable();

        case Expr::Kind::IntType:
            return cast<IntType>(ptr)->size;

        case Expr::Kind::Nil:
            return Size::Bits(0);

        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
            return ctx->size_of_pointer;

        case Expr::Kind::SliceType:
        case Expr::Kind::ClosureType: {
            auto ptr_size = Type::VoidRef.size(ctx);
            auto int_size = Type::Int.size(ctx);
            auto ptr_align = Type::VoidRef.align(ctx);
            auto int_align = Type::Int.align(ctx);
            return (ptr_size.align_to(int_align) + int_size).align_to(std::max(ptr_align, int_align));
        }

        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem->type.size(ctx);

        case Expr::Kind::ArrayType: {
            if (not ptr->sema.ok) return {};
            auto a = cast<ArrayType>(ptr);
            return a->elem->type.size(ctx) * a->dimension().getZExtValue();
        }

        case Expr::Kind::StructType:
            return cast<StructType>(ptr)->stored_size;

        case Expr::Kind::OptionalType: {
            auto o = cast<OptionalType>(ptr);
            if (isa<ReferenceType>(o->elem.desugared)) return o->elem.size(ctx);
            Todo("Optional non-references");
        }

        case Expr::Kind::OpaqueType:
            Diag::ICE("Cannot get size of opaque type '{}'", str(true));

        /// Invalid.
        case Expr::Kind::ProcType:
            Unreachable(".size accessed on function type");

        SOURCE_NON_TYPE_EXPRS:
            Unreachable(".size accessed on non-type expression");
    }

    Unreachable();
}

auto src::Type::str(bool use_colour, bool include_desugared) const -> std::string {
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
        case Expr::Kind::IntType: out += fmt::format("i{}", cast<IntType>(ptr)->size); break;
        case Expr::Kind::Nil: out += "nil"; break;
        case Expr::Kind::OptionalType: WriteSElem("?"); break;
        case Expr::Kind::ReferenceType: WriteSElem("&"); break;
        case Expr::Kind::ScopedPointerType: WriteSElem("^"); break;
        case Expr::Kind::SliceType: WriteSElem("[]"); break;

        case Expr::Kind::SugaredType: {
            out += C(Yellow);
            out += cast<SugaredType>(ptr)->name;
        } break;

        case Expr::Kind::ScopedType: {
            auto sc = cast<ScopedType>(ptr);
            out += fmt::format("{}::{}", sc->object->scope_name, sc->name);
        } break;

        case Expr::Kind::BuiltinType: {
            auto bk = cast<BuiltinType>(ptr)->builtin_kind;
            switch (bk) {
                case BuiltinTypeKind::ArrayLiteral: out += "<empty array literal>"; goto done;
                case BuiltinTypeKind::Bool: out += "bool"; goto done;
                case BuiltinTypeKind::Int: out += "int"; goto done;
                case BuiltinTypeKind::MemberProc: out += "<member procedure>"; goto done;
                case BuiltinTypeKind::NoReturn: out += "noreturn"; goto done;
                case BuiltinTypeKind::OverloadSet: out += "<overload set>"; goto done;
                case BuiltinTypeKind::Unknown: out += "<unknown>"; goto done;
                case BuiltinTypeKind::Void: out += "void"; goto done;
            }

            out += fmt::format("<invalid builtin type: {}>", int(bk));
        } break;

        case Expr::Kind::ArrayType: {
            auto a = cast<ArrayType>(ptr);
            out += fmt::format(
                "{}{}[{}{}{}]",
                a->elem.str(use_colour),
                C(Red),
                C(Magenta),
                a->sema.ok ? std::to_string(a->dimension().getZExtValue()) : "???",
                C(Red)
            );
        } break;

        case Expr::Kind::ClosureType: {
            auto c = cast<ClosureType>(ptr);
            out += fmt::format("{}closure ", C(Red));
            out += c->proc_type->type.str(use_colour);
        } break;

        case Expr::Kind::ProcType: {
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

            if (p->variadic) out += " variadic";

            /// Avoid relying on operator==() for this.
            if (
                not isa<BuiltinType>(p->ret_type) or
                cast<BuiltinType>(p->ret_type)->builtin_kind != BuiltinTypeKind::Void
            ) {
                out += " -> ";
                out += p->ret_type->type.str(use_colour);
            }
        } break;

        case Expr::Kind::StructType: {
            auto s = cast<StructType>(ptr);
            out += fmt::format("{}struct ", C(Red));
            if (not s->name.empty()) {
                out += fmt::format("{}{}", C(Cyan), s->name);
            } else {
                out += fmt::format(
                    "<anonymous {}{}{} at {}{}{}:{}{}{}>",
                    C(Blue),
                    fmt::ptr(ptr),
                    C(Red),
                    C(Magenta),
                    s->location.file_id,
                    C(Red),
                    C(Magenta),
                    s->location.pos,
                    C(Red)
                );
            }
        } break;

        case Expr::Kind::OpaqueType:
            out += fmt::format("{}{}", C(Cyan), cast<OpaqueType>(ptr)->name);
            break;

        SOURCE_NON_TYPE_EXPRS:
            return ptr->type.str(use_colour);
    }

done: // clang-format off
    auto d = desugared;
    if (
        include_desugared and
        d.ptr != ptr and (
            /// Do not print e.g. ‘bar aka struct bar’.
            not isa<SugaredType>(ptr) or
            not isa<StructType>(d.ptr) or
            cast<SugaredType>(ptr)->name != cast<StructType>(d.ptr)->name
        )
    ) out += fmt::format(" {}aka {}", C(Reset), d.str(use_colour));
    out += C(Reset);
    return out;
} // clang-format on

auto src::Type::_strip_arrays() -> ArrayInfo {
    Assert(ptr->sema.ok);
    auto d = desugared;
    if (auto ref = dyn_cast<ArrayType>(d.ptr)) {
        auto [type, dim, depth] = ref->elem.strip_arrays;
        return {type, dim * ref->dimension().getZExtValue(), depth + 1};
    } else {
        return {d, 1, 0};
    }
}

auto src::Type::_strip_refs() -> Type {
    auto d = desugared;
    if (auto ref = dyn_cast<ReferenceType>(d.ptr)) return ref->elem.strip_refs;
    else return d;
}

auto src::Type::_strip_refs_and_pointers() -> Type {
    auto d = desugared;
    if (isa<ReferenceType, ScopedPointerType>(d.ptr))
        return cast<SingleElementTypeBase>(d.ptr)->elem.strip_refs_and_pointers;
    return d;
}

bool src::Type::_trivial() {
    Assert(ptr->sema.ok);
    switch (ptr->kind) {
        case Expr::Kind::BuiltinType:
        case Expr::Kind::IntType:
        case Expr::Kind::Nil:
        case Expr::Kind::OptionalType:
        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::SliceType:
            return true;

        case Expr::Kind::ClosureType:
        case Expr::Kind::OpaqueType:
        case Expr::Kind::ProcType:
        case Expr::Kind::ReferenceType:
            return false;

        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            return desugared.trivial;

        case Expr::Kind::ArrayType:
            return cast<ArrayType>(ptr)->elem.trivial;

        case Expr::Kind::StructType: {
            auto s = cast<StructType>(ptr);
            return s->initialisers.empty() and not s->deleter;
        }

        SOURCE_NON_TYPE_EXPRS:
            Unreachable("Not a type");
    }
}

bool src::Type::_yields_value() {
    return *this != Type::Void and *this != Type::NoReturn;
}

bool src::operator==(Type a, Type b) {
    /// Any instance of a type is equal to itself.
    if (a.ptr == b.ptr) return true;

    /// Non-types are never equal.
    if (not isa<TypeBase>(a) or not isa<TypeBase>(b)) return false;

    /// If either is a sugared type, look through the sugar.
    if (isa<SugaredType, ScopedType>(a)) return a.desugared == b;
    if (isa<SugaredType, ScopedType>(b)) return a == b.desugared;

    /// Types of different kinds are never equal.
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            Unreachable();

        case Expr::Kind::OpaqueType: {
            auto oa = cast<OpaqueType>(a);
            auto ob = cast<OpaqueType>(b);
            return oa->name == ob->name and oa->module == ob->module;
        }

        case Expr::Kind::Nil:
            return true;

        case Expr::Kind::BuiltinType:
            return cast<BuiltinType>(a)->builtin_kind == cast<BuiltinType>(b)->builtin_kind;

        case Expr::Kind::IntType:
            return cast<IntType>(a)->size == cast<IntType>(b)->size;

        case Expr::Kind::ArrayType: {
            auto arr_a = cast<ArrayType>(a);
            auto arr_b = cast<ArrayType>(b);
            if (not a->sema.ok or not b->sema.ok) return false;
            if (arr_a->dimension() != arr_b->dimension()) return false;
            [[fallthrough]];
        }

        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::SliceType:
        case Expr::Kind::OptionalType:
        case Expr::Kind::ClosureType: {
            return cast<SingleElementTypeBase>(a)->elem == cast<SingleElementTypeBase>(b)->elem;
        }

        case Expr::Kind::ProcType: {
            auto pa = cast<ProcType>(a);
            auto pb = cast<ProcType>(b);

            if (pa->param_types.size() != pb->param_types.size()) return false;
            if (pa->variadic != pb->variadic) return false;
            for (auto [p1, p2] : llvm::zip_equal(pa->param_types, pb->param_types))
                if (p1 != p2)
                    return false;

            return pa->ret_type == pb->ret_type;
        }

        case Expr::Kind::StructType: {
            auto sa = cast<StructType>(a);
            auto sb = cast<StructType>(b);

            /// If either type is named, they’re the same iff they’re the
            /// same instance. Since we’ve already tested for that above,
            /// we can just return here.
            ///
            /// Note that two different structs may have the same name if
            /// they are declared in different scopes, so checking for name
            /// equality makes no sense, ever.
            if (not sa->name.empty() or not sb->name.empty()) return false;

            /// Two anonymous structs are equal iff they are layout-compatible
            /// and neither has a user-defined constructor or destructor or any
            /// member functions.
            if (not sa->initialisers.empty() or not sb->initialisers.empty()) return false;
            if (not sa->member_procs.empty() or not sb->member_procs.empty()) return false;
            if (sa->deleter or sb->deleter) return false;
            return StructType::LayoutCompatible(sa, sb);
        }

        SOURCE_NON_TYPE_EXPRS:
            Unreachable("Not a type");
    }

    Unreachable();
}

auto src::TypeBase::DenseMapInfo::getHashValue(const Expr* t) -> usz {
    usz hash = 0;

    /// Hash names for structs.
    if (auto d = dyn_cast<StructType>(t)) hash = llvm::hash_combine(hash, d->name);

    /// Include element types for types that have them.
    else if (auto* s = dyn_cast<SingleElementTypeBase>(t)) {
        do {
            hash = llvm::hash_combine(hash, static_cast<Expr*>(s->elem)->kind);
            s = dyn_cast<SingleElementTypeBase>(s->elem);
        } while (s);
    }

    /// Always add at least our kind.
    return llvm::hash_combine(hash, t->kind);
}

/// This only checks the layout; whether this makes sense
/// at all is up to the caller.
bool src::StructType::LayoutCompatible(StructType* a, StructType* b) {
    using std::get;
    if (a->all_fields.size() != b->all_fields.size()) return false;

    /// Check if all fields’ types are the same. Note that,
    /// since padding is explicitly encoded in the form of
    /// extra fields, this also takes care of checking for
    /// alignment.
    return llvm::all_of(
        llvm::zip_equal(a->field_types(), b->field_types()),
        [](auto&& t) { return std::get<0>(t) == std::get<1>(t); }
    );
}

auto src::BlockExpr::NCAInFunction(BlockExpr* a, BlockExpr* b) -> BlockExpr* {
    llvm::SmallPtrSet<BlockExpr*, 8> scopes{};

    for (; a; a = a->parent) {
        scopes.insert(a);
        if (a->is_function) break;
    }

    for (; b; b = b->parent) {
        if (scopes.contains(b)) return b;
        if (b->is_function) break;
    }

    return nullptr;
}

/// ===========================================================================
///  AST Printing
/// ===========================================================================
namespace src {
namespace {
struct ASTPrinter {
    using enum utils::Colour;

    Module* mod;
    DenseSet<const ProcDecl*> printed_functions{};
    bool print_children_of_children = true;
    bool print_procedure_bodies = true;
    std::string out;
    bool use_colour = true;
    utils::Colours C{use_colour};

    ASTPrinter(Module* mod, bool use_colour, bool print_children)
        : mod{mod},
          print_children_of_children{print_children},
          use_colour{use_colour} {
    }

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
            case Linkage::Exported: out += "Exported "; return;
            case Linkage::Imported: out += "Imported "; return;
            case Linkage::Internal: out += "Internal "; return;
            case Linkage::LinkOnceODR: out += "LinkOnceODR "; return;
            case Linkage::Local: out += "Local "; return;
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
                PrintBasicHeader("ProcDecl", e);
                if (f->mangling != Mangling::None) {
                    out += fmt::format(
                        " {}{} {}[{}{}{}] {}{}{}\n",
                        C(Green),
                        f->name,
                        C(Red),
                        C(Green),
                        f->sema.ok ? f->mangled_name : "???",
                        C(Red),
                        f->type.str(use_colour),
                        f->nested ? fmt::format(" {}nested", C(Blue)) : "",
                        f->takes_static_chain ? " chain" : ""
                    );
                } else {
                    out += fmt::format(
                        " {}{} {}{}{}\n",
                        C(Green),
                        f->name,
                        f->type.str(use_colour),
                        f->nested ? fmt::format(" {}nested", C(Blue)) : "",
                        f->takes_static_chain ? " chain" : ""
                    );
                }
                return;
            }

            case K::LocalDecl:
            case K::ParamDecl: {
                auto v = cast<LocalDecl>(e);
                PrintBasicHeader("LocalDecl", e);
                out += fmt::format(
                    "{}{}{} {}{} lvalue{}",
                    C(White),
                    v->name.empty() ? "" : " ",
                    v->name,
                    v->type.str(use_colour),
                    C(Blue),
                    v->captured ? " captured" : ""
                );

                if (auto p = dyn_cast<ParamDecl>(v)) {
                    if (p->with) out += fmt::format(" with");
                }

                out += "\n";
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
                        utils::Escape(mod->strtab[i->index].drop_back(1)),
                        i->type.str(use_colour)
                    );
                } else {
                    out += fmt::format(" {}\n", i->type.str(use_colour));
                }
                return;
            }

            case K::DeclRefExpr: {
                auto n = cast<DeclRefExpr>(e);
                PrintBasicHeader("DeclRefExpr", e);
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

            case K::ModuleRefExpr: {
                auto n = cast<ModuleRefExpr>(e);
                PrintBasicHeader("ModuleRefExpr", e);
                out += fmt::format(
                    " {}{}\n",
                    C(White),
                    n->module->name
                );
                return;
            }

            case K::LocalRefExpr: {
                auto n = cast<LocalRefExpr>(e);
                PrintBasicHeader("LocalRefExpr", e);
                out += fmt::format(
                    " {} {}lvalue{}\n",
                    n->type.str(use_colour),
                    C(Blue),
                    n->parent != n->decl->parent
                        ? fmt::format(" chain{}:{}{}", C(Red), C(Green), n->decl->parent->name)
                        : ""
                );
                return;
            }

            case K::CastExpr: {
                auto c = cast<CastExpr>(e);
                PrintBasicHeader("CastExpr", e);
                out += C(Red);
                switch (c->cast_kind) {
                    case CastKind::ArrayToElemRef: out += " ArrayToElemRef"; break;
                    case CastKind::Hard: out += " Hard"; break;
                    case CastKind::Implicit: out += " Implicit"; break;
                    case CastKind::LValueRefToLValue: out += " LValueRefToLValue"; break;
                    case CastKind::LValueToReference: out += " LValueToReference"; break;
                    case CastKind::LValueToRValue: out += " LValueToRValue"; break;
                    case CastKind::OptionalNilTest: out += " OptionalNilTest"; break;
                    case CastKind::OptionalUnwrap: out += " OptionalUnwrap"; break;
                    case CastKind::OptionalWrap: out += " OptionalWrap"; break;
                    case CastKind::ReferenceToLValue: out += " ReferenceToLValue"; break;
                    case CastKind::Soft: out += " Soft"; break;
                }
                out += fmt::format(
                    " {}{}{}\n",
                    c->type.str(use_colour),
                    C(Blue),
                    c->is_lvalue ? " lvalue" : ""
                );
                return;
            }

            case K::BlockExpr: {
                auto b = cast<BlockExpr>(e);
                if (b->implicit) out += fmt::format("{}Implicit ", C(Red));
                out += fmt::format(
                    "{}BlockExpr {}{}{} {}<{}> {}\n",
                    C(Red),
                    C(Blue),
                    fmt::ptr(e),
                    C(Red),
                    C(Magenta),
                    e->location.pos,
                    b->type.str(use_colour)
                );
                return;
            }

            case K::InvokeExpr: PrintBasicNode("InvokeExpr", e, static_cast<Expr*>(e->type)); return;

            case K::InvokeBuiltinExpr: {
                static const auto String = [](Builtin b) -> std::string_view {
                    switch (b) {
                        case Builtin::Destroy: return "__srcc_destroy";
                        case Builtin::Memcpy: return "__srcc_memcpy";
                        case Builtin::New: return "new";
                    }

                    Unreachable();
                };

                auto i = cast<InvokeBuiltinExpr>(e);
                PrintBasicHeader("InvokeBuiltinExpr", e);
                out += fmt::format(
                    " {}{} {}{}{}\n",
                    C(Green),
                    String(i->builtin),
                    i->type.str(use_colour),
                    C(Blue),
                    i->is_lvalue ? " lvalue" : ""
                );

                return;
            }

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

            case K::ScopeAccessExpr: {
                auto m = cast<ScopeAccessExpr>(e);
                PrintBasicHeader("ScopeAccessExpr", e);
                out += fmt::format(
                    " {} {}{}{}{}\n",
                    m->type.str(use_colour),
                    C(Magenta),
                    m->element,
                    C(Blue),
                    m->is_lvalue ? " lvalue" : ""
                );
                return;
            }

            case K::LoopControlExpr: {
                auto l = cast<LoopControlExpr>(e);
                PrintBasicHeader("LoopControlExpr", e);
                out += fmt::format(
                    " {}{} {}{}\n",
                    C(Red),
                    l->is_break ? "break" : "continue",
                    C(Yellow),
                    l->label.empty() ? "<parent>" : l->label
                );
                return;
            }

            case K::GotoExpr: {
                auto l = cast<GotoExpr>(e);
                PrintBasicHeader("GotoExpr", e);
                out += fmt::format(
                    " {}{}\n",
                    C(Yellow),
                    l->label
                );
                return;
            }

            case K::LabelExpr: {
                auto l = cast<LabelExpr>(e);
                PrintBasicHeader("LabelExpr", e);
                out += fmt::format(
                    " {}{}\n",
                    C(Yellow),
                    l->label
                );
                return;
            }

            case K::ForInExpr: {
                auto f = cast<ForInExpr>(e);
                PrintBasicHeader("ForInExpr", e);
                if (f->reverse) out += fmt::format(" {}reverse", C(Yellow));
                out += "\n";
                return;
            }

            case K::ConstructExpr: {
                auto c = cast<ConstructExpr>(e);
                PrintBasicHeader("ConstructExpr", e);
                out += C(Blue);
                switch (c->ctor_kind) {
                    using enum ConstructKind;
                    case Uninitialised: out += " uninit"; break;
                    case Zeroinit: out += " zero"; break;
                    case MoveParameter: out += " move"; break;
                    case TrivialCopy: out += " trivial"; break;
                    case SliceFromParts: out += " slice"; break;
                    case InitialiserCall: out += " init"; break;
                    case ArrayInitialiserCall: out += " array-init"; break;
                    case ArrayBroadcast: out += " broadcast"; break;
                    case ArrayListInit: out += " list-init"; break;
                    case ArrayZeroinit: out += " array-zero"; break;
                }
                if (auto els = c->elems(); els != 1) out += fmt::format("{}:{}{}", C(Red), C(Yellow), els);
                out += '\n';
                return;
            }

            case K::ArrayLiteralExpr: PrintBasicNode("ArrayLiteralExpr", e, nullptr); return;
            case K::AssertExpr: PrintBasicNode("AssertExpr", e, nullptr); return;
            case K::EmptyExpr: PrintBasicNode("EmptyExpr", e, nullptr); return;
            case K::Nil: PrintBasicNode("Nil", e, nullptr); return;
            case K::OverloadSetExpr: PrintBasicNode("OverloadSetExpr", e, nullptr); return;
            case K::ReturnExpr: PrintBasicNode("ReturnExpr", e, nullptr); return;
            case K::WhileExpr: PrintBasicNode("WhileExpr", e, nullptr); return;

            case K::ConstExpr: PrintBasicNode("ConstExpr", e, static_cast<Expr*>(e->type)); return;
            case K::ExportExpr: PrintBasicNode("ExportExpr", e, static_cast<Expr*>(e->type)); return;
            case K::IfExpr: PrintBasicNode("IfExpr", e, static_cast<Expr*>(e->type)); return;
            case K::ImplicitThisExpr: PrintBasicNode("ImplicitThisExpr", e, static_cast<Expr*>(e->type)); return;
            case K::ParenExpr: PrintBasicNode("ParenExpr", e, static_cast<Expr*>(e->type)); return;
            case K::SubscriptExpr: PrintBasicNode("SubscriptExpr", e, static_cast<Expr*>(e->type)); return;

            case K::FieldDecl: {
                auto f = cast<FieldDecl>(e);
                PrintBasicHeader("FieldDecl", e);
                out += fmt::format(
                    "{} {}{} {}at {}{}\n",
                    f->type.str(use_colour),
                    C(Magenta),
                    f->name,
                    C(Red),
                    C(Yellow),
                    f->sema.ok ? std::to_string(f->offset.bits()) : "?"
                );
                return;
            }

            case K::AliasExpr: {
                PrintBasicHeader("AliasExpr", e);
                out += fmt::format(" {}{}\n", C(White), cast<AliasExpr>(e)->alias);
                return;
            }

            case K::DeferExpr: {
                PrintBasicHeader("DeferExpr", e);
                out += "\n";
                return;
            }

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

            /// Struct declaration.
            case K::StructType: {
                if (auto s = cast<StructType>(e); not s->name.empty()) {
                    PrintBasicHeader("StructDecl", e);

                    out += fmt::format(" {}{}", C(Cyan), s->name);
                    if (s->mangling != Mangling::None) {
                        out += fmt::format(
                            " {}[{}{}{}]",
                            C(Red),
                            C(Cyan),
                            s->sema.ok ? Type(s).mangled_name : "???",
                            C(Red)
                        );
                    }

                    out += fmt::format(
                        " {}{}{}/{}{}\n",
                        C(Yellow),
                        s->sema.ok ? std::to_string(s->stored_size.bits()) : "?",
                        C(Red),
                        C(Yellow),
                        s->sema.ok ? std::to_string(s->stored_alignment.value()) : "?"
                    );
                    return;
                }
                [[fallthrough]];
            }

            /// We don’t print any other types here.
            case K::ArrayType:
            case K::BuiltinType:
            case K::ClosureType:
            case K::IntType:
            case K::OpaqueType:
            case K::OptionalType:
            case K::ProcType:
            case K::ReferenceType:
            case K::ScopedPointerType:
            case K::ScopedType:
            case K::SliceType:
            case K::SugaredType:
                PrintBasicNode("Type", e, e);
                return;
        }

        PrintBasicNode(R"(<???>)", e, static_cast<Expr*>(e->type));
    }

    void PrintNodeChildren(const Expr* e, std::string leading_text = "") {
        if (not print_children_of_children) return;

        /// Print the children of a node.
        using K = Expr::Kind;
        switch (e->kind) {
            /// These don’t have children.
            case K::BoolLiteralExpr:
            case K::EmptyExpr:
            case K::FieldDecl:
            case K::IntegerLiteralExpr:
            case K::ModuleRefExpr:
            case K::StringLiteralExpr:
                break;

            /// We don’t print children of most types here.
            case K::ArrayType:
            case K::BuiltinType:
            case K::ClosureType:
            case K::IntType:
            case K::Nil:
            case K::OpaqueType:
            case K::OptionalType:
            case K::ProcType:
            case K::ReferenceType:
            case K::ScopedPointerType:
            case K::ScopedType:
            case K::SliceType:
            case K::SugaredType:
                break;

            case K::StructType: {
                auto s = cast<StructType>(e);
                SmallVector<Expr*, 36> children{};
                utils::append(children, s->all_fields);
                utils::append(children, s->initialisers);
                for (auto& [_, procs] : s->member_procs) utils::append(children, procs);
                if (s->deleter) children.push_back(s->deleter);
                PrintChildren(children, leading_text);
            } break;

            case K::ProcDecl: {
                auto f = cast<ProcDecl>(e);
                printed_functions.insert(f);
                if (not print_procedure_bodies) break;
                SmallVector<Expr*, 25> children{f->params.begin(), f->params.end()};
                if (auto body = llvm::dyn_cast_if_present<BlockExpr>(f->body))
                    children.insert(children.end(), body->exprs.begin(), body->exprs.end());
                PrintChildren(children, leading_text);
            } break;

            case K::DeclRefExpr: {
                auto n = cast<DeclRefExpr>(e);
                if (n->decl) {
                    tempset print_children_of_children = false;
                    PrintChildren(n->decl, leading_text);
                }
            } break;

            case K::LocalRefExpr: {
                auto n = cast<LocalRefExpr>(e);
                tempset print_children_of_children = false;
                PrintChildren(n->decl, leading_text);
            } break;

            case K::OverloadSetExpr: {
                auto o = cast<OverloadSetExpr>(e);
                tempset print_children_of_children = false;
                auto overloads = ArrayRef<Expr*>(reinterpret_cast<Expr* const*>(o->overloads.data()), o->overloads.size());
                PrintChildren(overloads, leading_text);
            } break;

            case K::InvokeExpr: {
                auto c = cast<InvokeExpr>(e);
                SmallVector<Expr*, 12> children{c->callee};
                children.insert(children.end(), c->args.begin(), c->args.end());
                children.insert(children.end(), c->init_args.begin(), c->init_args.end());
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

            case K::LocalDecl:
            case K::ParamDecl: {
                tempset print_procedure_bodies = false;
                auto v = cast<LocalDecl>(e);
                if (v->ctor) PrintChildren(v->ctor, leading_text);
            } break;

            case K::InvokeBuiltinExpr: {
                auto c = cast<InvokeBuiltinExpr>(e);
                PrintChildren(c->args, leading_text);
            } break;

            case K::MemberAccessExpr: {
                auto m = cast<MemberAccessExpr>(e);
                SmallVector<Expr*, 2> children{};
                if (m->object) children.push_back(m->object);
                if (m->field) children.push_back(m->field);
                PrintChildren(children, leading_text);
            } break;

            case K::ScopeAccessExpr: {
                auto m = cast<ScopeAccessExpr>(e);
                PrintChildren(m->object, leading_text);
            } break;

            case K::BinaryExpr: {
                auto b = cast<BinaryExpr>(e);
                PrintChildren({b->lhs, b->rhs}, leading_text);
            } break;

            case K::SubscriptExpr: {
                auto s = cast<SubscriptExpr>(e);
                PrintChildren({s->object, s->index}, leading_text);
            } break;

            case K::WhileExpr: {
                auto w = cast<WhileExpr>(e);
                PrintChildren({w->cond, w->body}, leading_text);
            } break;

            case K::ForInExpr: {
                auto w = cast<ForInExpr>(e);
                PrintChildren({w->iter, w->range, w->body}, leading_text);
            } break;

            case K::AliasExpr: {
                auto a = cast<AliasExpr>(e);
                PrintChildren(a->expr, leading_text);
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

            case K::GotoExpr: {
                tempset print_children_of_children = false;
                auto r = cast<GotoExpr>(e);
                SmallVector<Expr*, 2> children{};
                if (r->target) children.push_back(r->target);
                children.insert(children.end(), r->unwind.begin(), r->unwind.end());
                PrintChildren(children, leading_text);
            } break;

            case K::ImplicitThisExpr: {
                tempset print_children_of_children = false;
                auto r = cast<ImplicitThisExpr>(e);
                PrintChildren(r->init, leading_text);
            } break;

            case K::ConstExpr: {
                auto c = cast<ConstExpr>(e);
                if (c->expr) PrintChildren(c->expr, leading_text);
            } break;

            case K::ConstructExpr: {
                auto c = const_cast<ConstructExpr*>(cast<ConstructExpr>(e));
                PrintChildren(c->args_and_init(), leading_text);
            } break;

            case K::ArrayLiteralExpr:
                PrintChildren(cast<ArrayLitExpr>(e)->elements, leading_text);
                break;

            case K::LabelExpr:
                PrintChildren(cast<LabelExpr>(e)->expr, leading_text);
                break;

            case K::ExportExpr:
                PrintChildren(cast<ExportExpr>(e)->expr, leading_text);
                break;

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

            case K::ParenExpr:
                PrintChildren(cast<ParenExpr>(e)->expr, leading_text);
                break;
        }
    }

    /// Print a top-level node.
    void PrintTopLevelNode(Expr* e) {
        PrintHeader(e);
        PrintNodeChildren(e);
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

void src::Expr::print(bool print_children) const {
    /// Ok because ASTPrinter does not attempt to mutate this.
    ASTPrinter{nullptr, true, print_children}(const_cast<Expr*>(this));
}

void src::Module::print_ast() const {
    /// Ok because ASTPrinter does not attempt to mutate this.
    ASTPrinter{const_cast<Module*>(this), context->use_colours, true}.print();
}

void src::Module::print_exports() const {
    for (auto& exps : exports)
        for (auto e : exps.second)
            e->print(context->use_colours);
}
