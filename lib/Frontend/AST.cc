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
IntType IntType8Instance{Size::Bits(8), {}};
IntType IntType16Instance{Size::Bits(16), {}};
IntType IntType32Instance{Size::Bits(32), {}};
IntType IntType64Instance{Size::Bits(64), {}};
TupleType NilInstance{{}, {}};
ReferenceType VoidRefTypeInstance = [] {
    ReferenceType ty{&VoidTypeInstance, {}};
    ty.sema.set_done();
    return ty;
}();
ReferenceType VoidRefRefTypeInstance = [] {
    ReferenceType ty{&VoidRefTypeInstance, {}};
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
void src::Expr::operator delete(Expr* e, std::destroying_delete_t) {
    switch (e->kind) {
#define SOURCE_AST_EXPR(name)         \
    case Kind::name:                  \
        cast<name>(e)->~name();       \
        __builtin_operator_delete(e); \
        return;
#include <source/Frontend/AST.def>
    }
}

src::ProcDecl::ProcDecl(
    Module* mod,
    ProcDecl* parent,
    String name,
    Type type,
    SmallVector<ParamDecl*> param_decls,
    Linkage linkage,
    Mangling mangling,
    Location loc
) : ObjectDecl(Kind::ProcDecl, mod, name, type, linkage, mangling, loc),
    parent_ptr(parent),
    params(std::move(param_decls)),
    body(nullptr) {
    module->add_function(this);
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
    is_lvalue = decl->is_lvalue;
}

src::TupleIndexExpr::TupleIndexExpr(Expr* object, FieldDecl* field, Location loc)
    : TypedExpr(Kind::TupleIndexExpr, field->type, loc),
      object(object),
      field(field) {
    Assert(object->sema.ok, "Cannot index into unanalysed or dependent tuple");
    Assert(field->sema.ok, "Cannot index unanalysed or dependent field");
    sema.set_done();
    is_lvalue = object->is_lvalue;
}

auto src::LocalDecl::parent(Module* parent_module) -> ProcDecl* {
    if (parent_ptr) return parent_ptr;
    return parent_module->top_level_func;
}

void src::LocalDecl::set_captured() {
    if (is_captured) return;
    is_captured = true;
    if (parent_ptr) parent_ptr->captured_locals.push_back(this);
}

auto src::Expr::_ignore_labels() -> Expr* {
    if (auto l = dyn_cast<LabelExpr>(this)) return l->expr->ignore_labels;
    return this;
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
    if (auto t = dyn_cast<TupleExpr>(this)) return t->elements.empty();
    return false;
}

bool src::Expr::_is_protected() {
    return isa<DeferExpr, LocalDecl, ParamDecl>(ignore_labels);

    /// TODO: Assertions and if expressions that act as guards cannot be
    ///       branched into either.
}

bool src::Expr::_is_smp() {
    if (auto d = dyn_cast<DeclRefExpr>(this)) return d->decl->is_smp;
    if (auto p = dyn_cast<ProcType>(this)) return p->smp_parent != nullptr;
    if (auto p = dyn_cast<ProcDecl>(this)) return p->type->is_smp;
    if (auto o = dyn_cast<OverloadSetExpr>(this))
        return rgs::any_of(o->overloads, [](Expr* e) { return e->is_smp; });
    return false;
}

auto src::Expr::_scope_name() -> std::string {
    switch (kind) {
        default:
            return "<?>";

        case Kind::ModuleRefExpr:
            return std::string{cast<ModuleRefExpr>(this)->module->name.sv()};

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
            return std::string{s->name.sv()};
        }

        case Kind::ProcDecl: {
            auto p = cast<ProcDecl>(this);
            if (p->name.empty()) return "<anonymous>";
            if (p->parent_or_null) return fmt::format("{}::{}", p->parent_or_null->scope_name, p->name);
            if (p->module->is_logical_module) return fmt::format("{}::{}", p->module->name, p->name);
            return std::string{p->name.sv()};
        }
    }
}

auto src::Expr::_type() -> Type {
    switch (kind) {
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
            return Type::NoReturn;

        case Kind::OverloadSetExpr:
            return Type::OverloadSet;

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
#define SOURCE_AST_TYPED_EXPR(name) case Kind::name:
#include <source/Frontend/AST.def>
            return cast<TypedExpr>(this)->stored_type;

            /// Already a type.
#define SOURCE_AST_TYPE(name) case Kind::name:
#include <source/Frontend/AST.def>
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
    String sname,
    SmallVector<FieldDecl*> fields,
    SmallVector<ProcDecl*> inits,
    MemberProcedures member_procs,
    ProcDecl* deleter,
    BlockExpr* scope,
    Mangling mangling,
    Location loc
) : RecordType(Kind::StructType, std::move(fields), loc),
    Named(mod, sname, mangling),
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

        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
            return ctx->align_of_pointer;

        case Expr::Kind::SliceType:
        case Expr::Kind::ClosureType:
            return std::max(Type::VoidRef.align(ctx), Type::Int.align(ctx));

        case Expr::Kind::ArrayType:
        case Expr::Kind::EnumType:
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem->type.align(ctx);

        case Expr::Kind::StructType:
        case Expr::Kind::TupleType:
            return cast<RecordType>(ptr)->stored_alignment;

        case Expr::Kind::TypeofType:
            return cast<TypeofType>(ptr)->expr->type.align(ctx);

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

#define SOURCE_AST_EXPR(name) case Expr::Kind::name:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
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
    if (auto t = dyn_cast<TypeofType>(ptr)) return t->expr->type.desugared;
    return *this;
}

auto src::Type::_desugared_underlying() const -> Type {
    auto d = desugared;
    if (auto e = dyn_cast<EnumType>(d)) return e->underlying_type.desugared_underlying;
    return d;
}

bool src::Type::is_int(bool bool_is_int) {
    if (isa<SugaredType, ScopedType, TypeofType>(ptr)) return desugared.is_int(bool_is_int);
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
    if (auto t = dyn_cast<TupleType>(ptr)) return t->all_fields.empty();
    return false;
}

bool src::Type::_is_noreturn() {
    return *this == NoReturn;
}

auto src::EnumType::_parent_enum() -> EnumType* {
    return dyn_cast<EnumType>(elem.desugared);
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

        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
            return ctx->size_of_pointer;

        case Expr::Kind::TypeofType:
            return cast<TypeofType>(ptr)->expr->type.size(ctx);

        case Expr::Kind::SliceType:
        case Expr::Kind::ClosureType: {
            auto ptr_size = Type::VoidRef.size(ctx);
            auto int_size = Type::Int.size(ctx);
            auto ptr_align = Type::VoidRef.align(ctx);
            auto int_align = Type::Int.align(ctx);
            return (ptr_size.align_to(int_align) + int_size).align_to(std::max(ptr_align, int_align));
        }

        case Expr::Kind::EnumType:
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem->type.size(ctx);

        case Expr::Kind::ArrayType: {
            if (not ptr->sema.ok) return {};
            auto a = cast<ArrayType>(ptr);
            return Size::Bytes(a->elem->type.size(ctx).bytes()) * a->dimension().getZExtValue();
        }

        case Expr::Kind::StructType:
        case Expr::Kind::TupleType:
            return cast<RecordType>(ptr)->stored_size;

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

#define SOURCE_AST_EXPR(name) case Expr::Kind::name:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
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
        case Expr::Kind::OptionalType: WriteSElem("?"); break;
        case Expr::Kind::ReferenceType: WriteSElem("&"); break;
        case Expr::Kind::ScopedPointerType: WriteSElem("^"); break;
        case Expr::Kind::SliceType: WriteSElem("[]"); break;

        case Expr::Kind::SugaredType: {
            out += C(Yellow);
            out += cast<SugaredType>(ptr)->name.sv();
        } break;

        case Expr::Kind::ScopedType: {
            auto sc = cast<ScopedType>(ptr);
            out += fmt::format("{}::{}", sc->object->scope_name, sc->name);
        } break;

        case Expr::Kind::TypeofType: {
            out += "typeof";
            auto e = cast<TypeofType>(ptr)->expr;
            if (e->sema.ok) out += fmt::format("{}:{}", C(Red), e->type.str(use_colour));
            else out += ":???";
        } break;

        case Expr::Kind::BuiltinType: {
            auto bk = cast<BuiltinType>(ptr)->builtin_kind;
            switch (bk) {
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

            if (not p->parameters.empty()) {
                out += " (";
                for (auto [i, param] : vws::enumerate(p->parameters)) {
                    if (param.passed_by_reference) out += "ref ";
                    out += param.type.str(use_colour);
                    if (usz(i) != p->parameters.size() - 1) out += fmt::format("{}, ", C(Red));
                }
                out += C(Red);
                out += ")";
            }

            if (p->call_conv == CallConv::Native) out += " native";
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
            if (not s->name.empty()) {
                out += fmt::format("{}{}", C(Cyan), s->name);
            } else {
                out += fmt::format(
                    "{} struct <anonymous {}{}{} at {}{}{}+{}{}{}>",
                    C(Red),
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

        case Expr::Kind::EnumType: {
            auto s = cast<EnumType>(ptr);
            if (not s->name.empty()) {
                out += fmt::format("{}{}", C(Cyan), s->name);
            } else {
                out += fmt::format(
                    "{} enum <anonymous {}{}{} at {}{}{}+{}{}{}>",
                    C(Red),
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

        case Expr::Kind::TupleType: {
            auto s = cast<TupleType>(ptr);
            out += fmt::format("{}(", C(Red));
            for (auto [i, elem] : vws::enumerate(s->field_types())) {
                if (i != 0) out += fmt::format("{}, ", C(Red));
                out += elem->type.str(use_colour);
            }
            out += fmt::format("{})", C(Red));
        } break;

        case Expr::Kind::OpaqueType:
            out += fmt::format("{}{}", C(Cyan), cast<OpaqueType>(ptr)->name);
            break;

#define SOURCE_AST_EXPR(name) case Expr::Kind::name:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
            return fmt::format("{}<???>", C(Cyan));
    }

done: // clang-format off
    auto d = desugared;
    bool print_canonical = [&] -> bool {
        /// Type is not sugared.
        if (not include_desugared or d.ptr == ptr) return false;
        auto s = dyn_cast<SugaredType>(ptr);
        if (not s) return false;

        /// Structs and enums may be behind a sugared type with the
        /// same name; do not print the 'aka' in that case.
        if (auto t = dyn_cast<StructType>(d.ptr)) return t->name != s->name;
        if (auto e = dyn_cast<EnumType>(d.ptr)) return e->name != s->name;
        return true;
    }();
    if (print_canonical) out += fmt::format(" {}aka {}", C(Reset), d.str(use_colour));
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
        case Expr::Kind::OptionalType:
        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::SliceType:
            return true;

        case Expr::Kind::ClosureType:
        case Expr::Kind::OpaqueType:
        case Expr::Kind::ProcType:
        case Expr::Kind::ReferenceType:
            return false;

        /// Enums are trivial if zero is a valid value.
        case Expr::Kind::EnumType:
            return rgs::any_of(
                cast<EnumType>(ptr)->enumerators,
                [](auto e) { return e->value.isZero(); }
            );

        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
        case Expr::Kind::TypeofType:
            return desugared.trivial;

        case Expr::Kind::ArrayType:
            return cast<ArrayType>(ptr)->elem.trivial;

        case Expr::Kind::TupleType:
            return rgs::all_of(cast<TupleType>(ptr)->field_types(), std::identity{}, &Type::_trivial);


        case Expr::Kind::StructType: {
            auto s = cast<StructType>(ptr);
            return s->initialisers.empty() and not s->deleter;
        }

#define SOURCE_AST_EXPR(name) case Expr::Kind::name:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
            Unreachable("Not a type");
    }
}

bool src::Type::_trivially_copyable() {
    Assert(ptr->sema.ok);
    switch (ptr->kind) {
        case Expr::Kind::BuiltinType:
        case Expr::Kind::ClosureType:
        case Expr::Kind::EnumType:
        case Expr::Kind::IntType:
        case Expr::Kind::OpaqueType:
        case Expr::Kind::OptionalType:
        case Expr::Kind::ProcType:
        case Expr::Kind::ReferenceType:
        case Expr::Kind::SliceType:
            return true;

        case Expr::Kind::ScopedPointerType:
            return false;

        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
        case Expr::Kind::TypeofType:
            return desugared.trivial;


        case Expr::Kind::ArrayType:
            return cast<ArrayType>(ptr)->elem.trivial;

        case Expr::Kind::TupleType:
            return rgs::all_of(cast<TupleType>(ptr)->field_types(), std::identity{}, &Type::_trivial);

        /// TODO: Attribute similar to Rust’s #[derive(Copy)] for cases where
        /// a type has an initialiser, but is still trivially copyable.
        case Expr::Kind::StructType: {
            auto s = cast<StructType>(ptr);
            return s->initialisers.empty() and not s->deleter;
        }

#define SOURCE_AST_EXPR(name) case Expr::Kind::name:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
            Unreachable("Not a type");
    }
}

auto src::EnumType::_underlying_type() -> Type {
    auto d = elem.desugared;
    if (auto n = dyn_cast<EnumType>(d)) return n->underlying_type;
    return d;
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
    if (isa<SugaredType, ScopedType, TypeofType>(a)) return a.desugared == b;
    if (isa<SugaredType, ScopedType, TypeofType>(b)) return a == b.desugared;

    /// Types of different kinds are never equal.
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
        case Expr::Kind::TypeofType:
            Unreachable();

        case Expr::Kind::OpaqueType: {
            auto oa = cast<OpaqueType>(a);
            auto ob = cast<OpaqueType>(b);
            return oa->name == ob->name and oa->module == ob->module;
        }

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

            if (pa->parameters.size() != pb->parameters.size()) return false;
            if (pa->call_conv != pb->call_conv) return false;
            if (pa->variadic != pb->variadic) return false;
            for (auto [p1, p2] : llvm::zip_equal(pa->parameters, pb->parameters))
                if (p1 != p2)
                    return false;

            return pa->ret_type == pb->ret_type;
        }

        case Expr::Kind::TupleType: {
            auto ta = cast<TupleType>(a);
            auto tb = cast<TupleType>(b);
            return RecordType::LayoutCompatible(ta, tb);
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
            return RecordType::LayoutCompatible(sa, sb);
        }

        /// Different enums are never equal.
        case Expr::Kind::EnumType:
            return false;

#define SOURCE_AST_EXPR(name) case Expr::Kind::name:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
            Unreachable("Not a type");
    }

    Unreachable();
}

auto src::TypeBase::DenseMapInfo::getHashValue(const Expr* t) -> usz {
    usz hash = 0;

    /// Hash names for structs.
    if (auto d = dyn_cast<StructType>(t)) hash = llvm::hash_combine(hash, d->name.value());

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
bool src::RecordType::LayoutCompatible(RecordType* a, RecordType* b) {
    /// This also checks for padding and alignment since those
    /// are encoded as extra padding fields.
    return a->stored_size == b->stored_size and
           a->stored_alignment == b->stored_alignment and
           rgs::equal(a->field_types(), b->field_types());
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
class ASTPrinter {
public:
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

    /// Print a constant value.
    void PrintValue(const EvalResult& v) { // clang-format off
        visit(v.value, utils::overloaded {
            [&](Type t) { out += t.str(use_colour); },
            [&](String s) { out += fmt::format("{}\"{}\"", C(Yellow), utils::Escape(s)); },
            [&](std::monostate) { out += "<invalid>"; },
            [&](std::nullptr_t) { out += fmt::format("{}nil", C(Red)); },
            [&](OverloadSetExpr* os) { out += fmt::format("{}{}", C(Green), os->overloads.front()->name); },
            [&](const APInt& a) {
                if (v.type == Type::Bool) out += fmt::format("{}{}", C(Red), a.getBoolValue() ? "true" : "false");
                else out += fmt::format("{}{}", C(Magenta), a);
            },
            [&](const EvalResult::TupleElements& els) {
                out += fmt::format("{}(", C(Red));
                bool first = true;
                for (auto& el : els) {
                    if (first) first = false;
                    else out += fmt::format("{}, ", C(Red));
                    PrintValue(el);
                }
                out += fmt::format("{})", C(Red));
            },
        });
    } // clang-format on

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
    template <std::derived_from<Expr> Expression = Expr>
    void PrintChildren(std::type_identity_t<ArrayRef<Expression*>> exprs, std::string leading_text) {
        for (usz i = 0; i < exprs.size(); i++) {
            if (exprs[i] == nullptr) continue;
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
            /// We don’t print most types here.
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
            case K::TupleType:
            case K::TypeofType:
            print_type:
                PrintBasicNode("Type", e, e);
                return;

            case K::EmptyExpr: PrintBasicNode("EmptyExpr", e, nullptr); return;
            case K::OverloadSetExpr: PrintBasicNode("OverloadSetExpr", e, nullptr); return;
            case K::ReturnExpr: PrintBasicNode("ReturnExpr", e, nullptr); return;
            case K::WhileExpr: PrintBasicNode("WhileExpr", e, nullptr); return;

            case K::AssignExpr: PrintBasicNode("AssignExpr", e, static_cast<Expr*>(e->type)); return;
            case K::ExportExpr: PrintBasicNode("ExportExpr", e, static_cast<Expr*>(e->type)); return;
            case K::ImplicitThisExpr: PrintBasicNode("ImplicitThisExpr", e, static_cast<Expr*>(e->type)); return;
            case K::InvokeExpr: PrintBasicNode("InvokeExpr", e, static_cast<Expr*>(e->type)); return;
            case K::MaterialiseTemporaryExpr: PrintBasicNode("MaterialiseTemporaryExpr", e, static_cast<Expr*>(e->type)); return;
            case K::ParenExpr: PrintBasicNode("ParenExpr", e, static_cast<Expr*>(e->type)); return;
            case K::SubscriptExpr: PrintBasicNode("SubscriptExpr", e, static_cast<Expr*>(e->type)); return;
            case K::TupleExpr: PrintBasicNode("TupleExpr", e, static_cast<Expr*>(e->type)); return;
            case K::TupleIndexExpr: PrintBasicNode("TupleIndexExpr", e, static_cast<Expr*>(e->type)); return;
            case K::WithExpr: PrintBasicNode("WithExpr", e, static_cast<Expr*>(e->type)); return;

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

            case K::ConstExpr: {
                auto c = cast<ConstExpr>(e);
                PrintBasicHeader("ConstExpr", e);
                out += fmt::format(" {} ", c->type.str(use_colour));
                PrintValue(c->value);
                out += "\n";
                return;
            }

            case K::LocalDecl:
            case K::ParamDecl: {
                auto v = cast<LocalDecl>(e);
                PrintBasicHeader(isa<ParamDecl>(v) ? "ParamDecl" : "LocalDecl", e);
                out += fmt::format(
                    "{}{}{} {}{}{}{}{}",
                    C(White),
                    v->name.empty() ? "" : " ",
                    v->name,
                    v->type.str(use_colour),
                    C(Blue),
                    v->is_lvalue ? " lvalue" : "",
                    v->definitely_moved    ? " moved"
                    : v->partially_moved   ? " partially-moved"
                    : v->potentially_moved ? " potentially-moved"
                                           : "",
                    v->captured ? " captured" : ""
                );

                if (auto p = dyn_cast<ParamDecl>(v)) {
                    if (p->info->with) out += fmt::format(" with");
                    out += " ";
                    out += stringify(p->info->intent);
                }

                out += "\n";
                return;
            }

            case K::BoolLitExpr: {
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

            case K::IntLitExpr: {
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

            case K::StrLitExpr: {
                auto i = cast<StrLitExpr>(e);
                PrintBasicHeader("StringLiteral", e);
                out += fmt::format(
                    " {}\"{}\" {}\n",
                    C(Yellow),
                    utils::Escape(i->string),
                    i->type.str(use_colour)
                );
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
                    " {}{}{}{}\n",
                    n->type.str(use_colour),
                    C(Blue),
                    n->is_lvalue ? " lvalue" : "",
                    n->parent != n->decl->parent(mod)
                        ? fmt::format(" chain{}:{}{}", C(Red), C(Green), n->decl->parent(mod)->name)
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
                    case CastKind::BitCast: out += " BitCast"; break;
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
                    case Parameter: out += " parameter"; break;
                    case Copy: out += " trivial"; break;
                    case SliceFromParts: out += " slice"; break;
                    case InitialiserCall: out += " init"; break;
                    case ArrayInitialiserCall: out += " array-init"; break;
                    case ArrayBroadcast: out += " broadcast"; break;
                    case ArrayListInit: out += " list-init"; break;
                    case ArrayZeroinit: out += " array-zero"; break;
                    case RecordListInit: out += " record-list-init"; break;
                }
                if (auto els = c->elems(); els != 1) out += fmt::format("{}:{}{}", C(Red), C(Yellow), els);
                out += '\n';
                return;
            }

            case K::EnumType: {
                auto a = cast<EnumType>(e);
                PrintBasicHeader("EnumType", e);
                out += fmt::format(
                    "{}{}{}{}{} {}: {}\n",
                    C(Cyan),
                    a->name.empty() ? "" : " ",
                    a->name,
                    C(Yellow),
                    a->mask ? " mask" : "",
                    C(Red),
                    a->elem.str(use_colour)
                );
                return;
            }

            case K::EnumeratorDecl: {
                auto a = cast<EnumeratorDecl>(e);
                PrintBasicHeader("EnumeratorDecl", e);
                out += fmt::format(
                    " {}{}\n",
                    C(Blue),
                    a->name
                );
                return;
            }

            case K::AssertExpr: {
                auto a = cast<AssertExpr>(e);
                PrintBasicHeader("AssertExpr", e);
                out += fmt::format(
                    "{}{}\n",
                    C(Yellow),
                    a->is_static ? " static" : ""
                );
                return;
            }

            case K::IfExpr: {
                auto i = cast<IfExpr>(e);
                PrintBasicHeader("IfExpr", e);
                out += fmt::format(
                    " {}{}{}\n",
                    i->type.str(use_colour),
                    C(Yellow),
                    i->is_static ? " static" : ""
                );
                return;
            }

            case K::FieldDecl: {
                auto f = cast<FieldDecl>(e);
                PrintBasicHeader("FieldDecl", e);
                out += fmt::format(
                    " {} {}{} {}at {}{}\n",
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

                /// Unnamed structs are just printed in-line.
                goto print_type;
            }
        }

        PrintBasicNode(R"(<???>)", e, static_cast<Expr*>(e->type));
    }

    void PrintNodeChildren(const Expr* e, std::string leading_text = "") {
        if (not print_children_of_children) return;

        /// Print the children of a node.
        using K = Expr::Kind;
        switch (e->kind) {
            /// These don’t have children.
            case K::BoolLitExpr:
            case K::EmptyExpr:
            case K::FieldDecl:
            case K::IntLitExpr:
            case K::ModuleRefExpr:
            case K::StrLitExpr:
                break;

            /// We don’t print children of most types here.
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
            case K::TupleType:
            case K::TypeofType:
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

            case K::WithExpr: {
                auto m = cast<WithExpr>(e);
                SmallVector<Expr*, 2> children{};
                if (m->object) children.push_back(m->object);
                if (m->body) children.push_back(m->body);
                PrintChildren(children, leading_text);
            } break;

            case K::EnumType: {
                auto n = cast<EnumType>(e);
                PrintChildren<EnumeratorDecl>(n->enumerators, leading_text);
            } break;

            case K::ScopeAccessExpr: {
                auto m = cast<ScopeAccessExpr>(e);
                PrintChildren(m->object, leading_text);
            } break;

            case K::BinaryExpr: {
                auto b = cast<BinaryExpr>(e);
                PrintChildren({b->lhs, b->rhs}, leading_text);
            } break;

            case K::AssignExpr: {
                auto a = cast<AssignExpr>(e);
                PrintChildren({a->lvalue, a->ctor}, leading_text);
            } break;

            case K::TupleExpr: {
                auto t = cast<TupleExpr>(e);
                PrintChildren(t->elements, leading_text);
            } break;

            case K::TupleIndexExpr: {
                auto t = cast<TupleIndexExpr>(e);
                PrintChildren({t->object, t->field}, leading_text);
            } break;

            case K::EnumeratorDecl: {
                auto t = cast<EnumeratorDecl>(e);
                if (t->initialiser) PrintChildren(t->initialiser, leading_text);
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
                PrintChildren({w->iter, w->index, w->range, w->body}, leading_text);
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

            case K::LabelExpr:
                PrintChildren(cast<LabelExpr>(e)->expr, leading_text);
                break;

            case K::MaterialiseTemporaryExpr:
                PrintChildren(cast<MaterialiseTemporaryExpr>(e)->ctor, leading_text);
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
        if (e->sema.errored) {
            out += "<<<ERROR>>>\n";
            return;
        }

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
