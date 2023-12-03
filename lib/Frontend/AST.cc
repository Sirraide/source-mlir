#include <source/Frontend/AST.hh>
#include <source/Support/Utils.hh>

namespace {
src::BuiltinType IntTypeInstance{src::BuiltinTypeKind::Int, {}};
src::BuiltinType UnknownTypeInstance{src::BuiltinTypeKind::Unknown, {}};
src::BuiltinType VoidTypeInstance{src::BuiltinTypeKind::Void, {}};
src::BuiltinType BoolTypeInstance{src::BuiltinTypeKind::Bool, {}};
src::BuiltinType NoReturnTypeInstance{src::BuiltinTypeKind::NoReturn, {}};
src::BuiltinType OverloadSetTypeInstance{src::BuiltinTypeKind::OverloadSet, {}};
src::BuiltinType EmptyArrayTypeInstance{src::BuiltinTypeKind::EmptyArray, {}};
src::ReferenceType VoidRefTypeInstance{&VoidTypeInstance, {}};
src::ReferenceType VoidRefRefTypeInstance{&VoidTypeInstance, {}};
src::IntType IntType8Instance{src::Size::Bits(8), {}};
src::IntType IntType16Instance{src::Size::Bits(16), {}};
src::IntType IntType32Instance{src::Size::Bits(32), {}};
src::IntType IntType64Instance{src::Size::Bits(64), {}};
src::FFIType FFITypeCCharInstance{src::FFITypeKind::CChar, {}};
src::FFIType FFITypeCIntInstance{src::FFITypeKind::CInt, {}};
src::Nil NilInstance{{}};
} // namespace
src::Expr* const src::detail::UnknownType = &UnknownTypeInstance;
src::BuiltinType* const src::Type::Int = &IntTypeInstance;
src::BuiltinType* const src::Type::Void = &VoidTypeInstance;
src::BuiltinType* const src::Type::Unknown = &UnknownTypeInstance;
src::BuiltinType* const src::Type::Bool = &BoolTypeInstance;
src::BuiltinType* const src::Type::NoReturn = &NoReturnTypeInstance;
src::BuiltinType* const src::Type::OverloadSet = &OverloadSetTypeInstance;
src::BuiltinType* const src::Type::EmptyArray = &EmptyArrayTypeInstance;
src::ReferenceType* const src::Type::VoidRef = &VoidRefTypeInstance;
src::ReferenceType* const src::Type::VoidRefRef = &VoidRefRefTypeInstance;
src::IntType* const src::Type::I8 = &IntType8Instance;
src::IntType* const src::Type::I16 = &IntType16Instance;
src::IntType* const src::Type::I32 = &IntType32Instance;
src::IntType* const src::Type::I64 = &IntType64Instance;
src::FFIType* const src::Type::CChar = &FFITypeCCharInstance;
src::FFIType* const src::Type::CInt = &FFITypeCIntInstance;
src::Nil* const src::Type::Nil = &NilInstance;

/// ===========================================================================
///  Expressions
/// ===========================================================================
src::ProcDecl::ProcDecl(
    Module* mod,
    ProcDecl* parent,
    std::string name,
    Expr* type,
    SmallVector<LocalDecl*> param_decls,
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

bool src::Expr::_is_active_optional() {
    auto local = dyn_cast<LocalDecl>(this->ignore_paren_refs);
    return isa<OptionalType>(this->type) and local and local->has_value;
}

bool src::Expr::_is_nil() {
    return isa<Nil>(this);
}

auto src::Expr::_scope_name() -> std::string {
    switch (kind) {
        case Kind::BuiltinType:
        case Kind::FFIType:
        case Kind::IntType:
        case Kind::ReferenceType:
        case Kind::ScopedPointerType:
        case Kind::SliceType:
        case Kind::ArrayType:
        case Kind::OptionalType:
        case Kind::ProcType:
        case Kind::ClosureType:
        case Kind::AssertExpr:
        case Kind::ReturnExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::ForInExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
        case Kind::LabelExpr:
        case Kind::EmptyExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::ConstExpr:
        case Kind::CastExpr:
        case Kind::MemberAccessExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::DeclRefExpr:
        case Kind::LocalRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::LocalDecl:
        case Kind::OverloadSetExpr:
        case Kind::ImplicitThisExpr:
        case Kind::ParenExpr:
        case Kind::SubscriptExpr:
        case Kind::Nil:
            return "<?>";

        case Kind::ModuleRefExpr:
            return cast<ModuleRefExpr>(this)->module->name;

        case Kind::ExportExpr:
            return cast<ExportExpr>(this)->expr->scope_name;

        case Kind::SugaredType:
            return cast<SugaredType>(this)->as_type.desugared->scope_name;

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

auto src::Expr::_type() -> TypeHandle {
    switch (kind) {
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
            return Type::NoReturn;

        case Kind::OverloadSetExpr:
            return Type::OverloadSet;

        case Kind::AssertExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::ForInExpr:
        case Kind::ExportExpr:
        case Kind::ModuleRefExpr:
        case Kind::LabelExpr:
        case Kind::EmptyExpr:
            return Type::Void;

        /// Typed exprs.
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::ConstExpr:
        case Kind::MemberAccessExpr:
        case Kind::ScopeAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::LocalRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ProcDecl:
        case Kind::CastExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::LocalDecl:
        case Kind::ImplicitThisExpr:
        case Kind::ParenExpr:
        case Kind::SubscriptExpr:
            return cast<TypedExpr>(this)->stored_type->as_type;

        /// Already a type.
        case Kind::BuiltinType:
        case Kind::FFIType:
        case Kind::ReferenceType:
        case Kind::ScopedPointerType:
        case Kind::OptionalType:
        case Kind::ProcType:
        case Kind::ClosureType:
        case Kind::IntType:
        case Kind::SliceType:
        case Kind::StructType:
        case Kind::ArrayType:
        case Kind::SugaredType:
        case Kind::ScopedType:
        case Kind::Nil:
            return as_type;
    }
}

auto src::Expr::_unwrapped_type() -> TypeHandle {
    if (is_active_optional) {
        auto opt = cast<OptionalType>(this->type);
        return opt->type->unwrapped_type;
    }

    return type.strip_refs_and_pointers;
}

auto src::ProcDecl::_ret_type() -> TypeHandle {
    return cast<ProcType>(stored_type)->ret_type->as_type;
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
    SmallVector<Field> fields,
    SmallVector<ProcDecl*> inits,
    BlockExpr* scope,
    Location loc
) : Type(Kind::StructType, loc),
    module(mod),
    all_fields(std::move(fields)),
    initialisers(std::move(inits)),
    name(std::move(sname)),
    scope(scope) {
    if (not name.empty()) mod->named_structs.push_back(this);
}

/// FIXME: Use llvm::Align for this.
auto src::Expr::TypeHandle::align([[maybe_unused]] src::Context* ctx) -> Align {
    switch (ptr->kind) {
        case Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::Unknown: return Align(1);
                case BuiltinTypeKind::Void: return Align(1);
                case BuiltinTypeKind::Int: return Align(8); /// FIXME: Use context.
                case BuiltinTypeKind::Bool: return Align(1);
                case BuiltinTypeKind::NoReturn: return Align(1);
                case BuiltinTypeKind::OverloadSet: return Align(1);
                case BuiltinTypeKind::EmptyArray: return Align(1);
            }

            Unreachable();

        case Kind::FFIType:
            switch (cast<FFIType>(ptr)->ffi_kind) {
                case FFITypeKind::CChar: return Align(1);
                case FFITypeKind::CInt: return Align(4); /// FIXME: Use context.
            }

            Unreachable();

        case Kind::IntType:
            /// FIXME: Use context.
            return Align(std::min<usz>(1, std::bit_ceil(usz(cast<IntType>(ptr)->size.bytes()))));

        case Kind::Nil:
            return Align(1);

        case Kind::ReferenceType:
            return Align(8); /// FIXME: Use context.

        case Kind::ScopedPointerType:
            return Align(8); /// FIXME: Use context.

        case Kind::SliceType:
            return Align(8); /// FIXME: Use context.

        case Kind::ArrayType:
        case Kind::SugaredType:
        case Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem->type.align(ctx);

        case Kind::StructType:
            return cast<StructType>(ptr)->stored_alignment;

        case Kind::ClosureType:
            return Align(8); /// FIXME: Use context.

        case Kind::OptionalType: {
            auto opt = cast<OptionalType>(ptr);
            if (isa<ReferenceType>(opt->elem)) return opt->elem->as_type.align(ctx);
            Todo();
        }

        /// Invalid.
        case Kind::ProcType:
            Unreachable(".align accessed on function type");

        case Kind::ExportExpr:
        case Kind::DeclRefExpr:
        case Kind::ModuleRefExpr:
        case Kind::LocalRefExpr:
        case Kind::AssertExpr:
        case Kind::ReturnExpr:
        case Kind::ConstExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
        case Kind::LabelExpr:
        case Kind::EmptyExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::ForInExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::CastExpr:
        case Kind::MemberAccessExpr:
        case Kind::ScopeAccessExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ProcDecl:
        case Kind::LocalDecl:
        case Kind::OverloadSetExpr:
        case Kind::ImplicitThisExpr:
        case Kind::ParenExpr:
        case Kind::SubscriptExpr:
            Unreachable(".align accessed on non-type expression");
    }

    Unreachable();
}

auto src::Expr::TypeHandle::_callable() -> ProcType* {
    if (auto p = dyn_cast<ProcType>(ptr)) return p;
    if (auto p = dyn_cast<ClosureType>(ptr)) return p->proc_type;
    Unreachable("Type '{}' is not callable", str(true));
}

bool src::Expr::TypeHandle::_default_constructible() {
    Assert(ptr->sema.ok);
    switch (ptr->kind) {
        case Kind::BuiltinType:
        case Kind::FFIType:
        case Kind::IntType:
        case Kind::OptionalType:
        case Kind::ScopedPointerType:
        case Kind::SliceType:
        case Kind::Nil:
            return true;

        case Kind::ClosureType:
        case Kind::ProcType:
        case Kind::ReferenceType:
            return false;

        case Kind::SugaredType:
        case Kind::ScopedType:
            return desugared.default_constructible;

        case Kind::ArrayType:
            return cast<ArrayType>(ptr)->elem->as_type.default_constructible;

        case Kind::StructType:
            /// If the type has no initialisers, or an initialiser that takes
            /// no argument, it is default-constructible.
            return rgs::all_of(cast<StructType>(ptr)->initialisers, [](auto init) {
                return init->params.empty();
            });

        case Kind::AssertExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::ForInExpr:
        case Kind::ExportExpr:
        case Kind::LabelExpr:
        case Kind::EmptyExpr:
        case Kind::ReturnExpr:
        case Kind::GotoExpr:
        case Kind::LoopControlExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::ConstExpr:
        case Kind::CastExpr:
        case Kind::MemberAccessExpr:
        case Kind::ScopeAccessExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::DeclRefExpr:
        case Kind::ModuleRefExpr:
        case Kind::LocalRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::LocalDecl:
        case Kind::ProcDecl:
        case Kind::OverloadSetExpr:
        case Kind::ImplicitThisExpr:
        case Kind::ParenExpr:
        case Kind::SubscriptExpr:
            Unreachable("Not a type");
    }
}

auto src::Expr::TypeHandle::_desugared() -> TypeHandle {
    if (auto s = dyn_cast<SugaredType>(ptr)) return s->elem->as_type.desugared;
    if (auto s = dyn_cast<ScopedType>(ptr)) return s->elem->as_type.desugared;
    return *this;
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

bool src::Expr::TypeHandle::_is_nil() {
    return isa<Nil>(ptr);
}

bool src::Expr::TypeHandle::_is_noreturn() {
    return Type::Equal(ptr, Type::NoReturn);
}

auto src::Expr::TypeHandle::_ref_depth() -> isz {
    isz depth = 0;
    for (
        auto th = *this;
        isa<ReferenceType, ScopedPointerType>(th);
        th = cast<SingleElementTypeBase>(th)->elem
    ) depth++;
    return depth;
}

/// FIXME: Use llvm::APInt for this.
auto src::Expr::TypeHandle::size([[maybe_unused]] src::Context* ctx) -> Size {
    switch (ptr->kind) {
        case Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::Unknown: return {};
                case BuiltinTypeKind::Void: return {};
                case BuiltinTypeKind::Int: return Size::Bits(64); /// FIXME: Use context.
                case BuiltinTypeKind::Bool: return Size::Bits(1);
                case BuiltinTypeKind::NoReturn: return {};
                case BuiltinTypeKind::OverloadSet: return {};
                case BuiltinTypeKind::EmptyArray: return {};
            }

            Unreachable();

        case Kind::FFIType:
            switch (cast<FFIType>(ptr)->ffi_kind) {
                case FFITypeKind::CChar: return Size::Bits(8);
                case FFITypeKind::CInt: return Size::Bits(32); /// FIXME: Use context.
            }

            Unreachable();

        case Kind::IntType:
            return cast<IntType>(ptr)->size;

        case Kind::Nil:
            return Size::Bits(0);

        case Kind::ReferenceType:
            return Size::Bits(64); /// FIXME: Use context.

        case Kind::ScopedPointerType:
            return Size::Bits(64); /// FIXME: Use context.

        case Kind::SliceType:
        case Kind::ClosureType:
            return Size::Bits(128); /// FIXME: Use context.

        case Kind::SugaredType:
        case Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem->type.size(ctx);

        case Kind::ArrayType: {
            if (not ptr->sema.ok) return {};
            auto a = cast<ArrayType>(ptr);
            return a->elem->type.size(ctx) * a->dimension().getZExtValue();
        }

        case Kind::StructType:
            return cast<StructType>(ptr)->stored_size;

        case Kind::OptionalType:
            Todo();

        /// Invalid.
        case Kind::ProcType:
            Unreachable(".size accessed on function type");

        case Kind::ExportExpr:
        case Kind::DeclRefExpr:
        case Kind::ModuleRefExpr:
        case Kind::LocalRefExpr:
        case Kind::AssertExpr:
        case Kind::ReturnExpr:
        case Kind::ConstExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
        case Kind::LabelExpr:
        case Kind::EmptyExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::ForInExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::CastExpr:
        case Kind::MemberAccessExpr:
        case Kind::ScopeAccessExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::IfExpr:
        case Kind::BinaryExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ProcDecl:
        case Kind::LocalDecl:
        case Kind::OverloadSetExpr:
        case Kind::ImplicitThisExpr:
        case Kind::ParenExpr:
        case Kind::SubscriptExpr:
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
        case Kind::IntType: out += fmt::format("i{}", cast<IntType>(ptr)->size); break;
        case Kind::SugaredType: out += cast<SugaredType>(ptr)->name; break;
        case Kind::Nil: out += "nil"; break;

        case Kind::ScopedType: {
            auto sc = cast<ScopedType>(ptr);
            out += fmt::format("{}::{}", sc->object->scope_name, sc->name);
        } break;

        case Kind::BuiltinType: {
            auto bk = cast<BuiltinType>(ptr)->builtin_kind;
            switch (bk) {
                case BuiltinTypeKind::Unknown: out += "<unknown>"; goto done;
                case BuiltinTypeKind::Void: out += "void"; goto done;
                case BuiltinTypeKind::Int: out += "int"; goto done;
                case BuiltinTypeKind::Bool: out += "bool"; goto done;
                case BuiltinTypeKind::NoReturn: out += "noreturn"; goto done;
                case BuiltinTypeKind::OverloadSet: out += "<overload set>"; goto done;
                case BuiltinTypeKind::EmptyArray: out += "<empty array literal>"; goto done;
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

        case Kind::ArrayType: {
            auto a = cast<ArrayType>(ptr);
            out += fmt::format(
                "{}{}[{}{}{}]",
                a->elem->as_type.str(use_colour),
                C(Red),
                C(Magenta),
                a->sema.ok ? std::to_string(a->dimension().getZExtValue()) : "???",
                C(Red)
            );
        } break;

        case Kind::ClosureType: {
            auto c = cast<ClosureType>(ptr);
            out += fmt::format("{}closure ", C(Red));
            out += c->proc_type->type.str(use_colour);
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

        case Kind::StructType: {
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

        case Kind::AssertExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::ForInExpr:
        case Kind::ModuleRefExpr:
        case Kind::ExportExpr:
        case Kind::LabelExpr:
        case Kind::EmptyExpr:
            return Type::Void->as_type.str(use_colour);

        case Kind::OverloadSetExpr:
            return Type::OverloadSet->as_type.str(use_colour);

        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
            return Type::NoReturn->as_type.str(use_colour);

        /// Typed exprs.
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::ConstExpr:
        case Kind::MemberAccessExpr:
        case Kind::ScopeAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::LocalRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ProcDecl:
        case Kind::CastExpr:
        case Kind::IfExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::BinaryExpr:
        case Kind::LocalDecl:
        case Kind::ImplicitThisExpr:
        case Kind::ParenExpr:
        case Kind::SubscriptExpr:
            return cast<TypedExpr>(ptr)->stored_type->type.str(use_colour);
    }

done:
    out += C(Reset);
    return out;
}

auto src::Expr::TypeHandle::_strip_refs() -> TypeHandle {
    auto d = desugared;
    if (auto ref = dyn_cast<ReferenceType>(d.ptr)) return ref->elem->as_type.strip_refs;
    else return d;
}

auto src::Expr::TypeHandle::_strip_refs_and_pointers() -> TypeHandle {
    auto d = desugared;
    if (isa<ReferenceType, ScopedPointerType>(d.ptr))
        return cast<SingleElementTypeBase>(d.ptr)->elem->as_type.strip_refs_and_pointers;
    return d;
}

bool src::Expr::TypeHandle::_yields_value() {
    return not Type::Equal(ptr, Type::Void) and not Type::Equal(ptr, Type::NoReturn);
}

bool src::Type::Equal(Expr* a, Expr* b) {
    /// Any instance of a type is equal to itself.
    if (a == b) return true;

    /// Non-types are never equal.
    if (not isa<Type>(a) or not isa<Type>(b)) return false;

    /// If either is a sugared type, look through the sugar.
    if (isa<SugaredType, ScopedType>(a)) return Type::Equal(a->as_type.desugared, b);
    if (isa<SugaredType, ScopedType>(b)) return Type::Equal(a, b->as_type.desugared);

    /// Types of different kinds are never equal.
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case Kind::SugaredType:
        case Kind::ScopedType:
            Unreachable();

        case Kind::Nil:
            return true;

        case Kind::BuiltinType:
            return cast<BuiltinType>(a)->builtin_kind == cast<BuiltinType>(b)->builtin_kind;

        case Kind::FFIType:
            return cast<FFIType>(a)->ffi_kind == cast<FFIType>(b)->ffi_kind;

        case Kind::IntType:
            return cast<IntType>(a)->size == cast<IntType>(b)->size;

        case Kind::ArrayType: {
            auto arr_a = cast<ArrayType>(a);
            auto arr_b = cast<ArrayType>(b);
            if (not a->sema.ok or not b->sema.ok) return false;
            if (arr_a->dimension() != arr_b->dimension()) return false;
            [[fallthrough]];
        }

        case Kind::ReferenceType:
        case Kind::ScopedPointerType:
        case Kind::SliceType:
        case Kind::OptionalType:
        case Kind::ClosureType: {
            return Type::Equal(
                cast<SingleElementTypeBase>(a)->elem,
                cast<SingleElementTypeBase>(b)->elem
            );
        }

        case Kind::ProcType: {
            auto pa = cast<ProcType>(a);
            auto pb = cast<ProcType>(b);

            if (pa->param_types.size() != pb->param_types.size()) return false;
            if (pa->variadic != pb->variadic) return false;
            for (auto [p1, p2] : llvm::zip_equal(pa->param_types, pb->param_types))
                if (not Type::Equal(p1, p2))
                    return false;

            return Type::Equal(pa->ret_type, pb->ret_type);
        }

        case Kind::StructType: {
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
            /// TODO: and neither has a user-defined constructor or destructor.
            return StructType::LayoutCompatible(sa, sb);
        }

        case Kind::ExportExpr:
        case Kind::AssertExpr:
        case Kind::ConstExpr:
        case Kind::ReturnExpr:
        case Kind::LoopControlExpr:
        case Kind::GotoExpr:
        case Kind::LabelExpr:
        case Kind::EmptyExpr:
        case Kind::DeferExpr:
        case Kind::WhileExpr:
        case Kind::ForInExpr:
        case Kind::BlockExpr:
        case Kind::InvokeExpr:
        case Kind::InvokeBuiltinExpr:
        case Kind::MemberAccessExpr:
        case Kind::ScopeAccessExpr:
        case Kind::DeclRefExpr:
        case Kind::ModuleRefExpr:
        case Kind::LocalRefExpr:
        case Kind::BoolLiteralExpr:
        case Kind::IntegerLiteralExpr:
        case Kind::ArrayLiteralExpr:
        case Kind::StringLiteralExpr:
        case Kind::ProcDecl:
        case Kind::CastExpr:
        case Kind::IfExpr:
        case Kind::UnaryPrefixExpr:
        case Kind::BinaryExpr:
        case Kind::LocalDecl:
        case Kind::OverloadSetExpr:
        case Kind::ImplicitThisExpr:
        case Kind::ParenExpr:
        case Kind::SubscriptExpr:
            Unreachable("Not a type");
    }

    Unreachable();
}

auto src::Type::DenseMapInfo::getHashValue(const Expr* t) -> usz {
    usz hash = 0;

    /// Hash names for structs.
    if (auto d = dyn_cast<StructType>(t)) hash = llvm::hash_combine(hash, d->name);

    /// Include element types for types that have them.
    else if (auto* s = dyn_cast<SingleElementTypeBase>(t)) {
        do {
            hash = llvm::hash_combine(hash, s->elem.ptr->kind);
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
        [](auto&& t) { return Type::Equal(t); }
    );
}

auto src::BlockExpr::NCAInFunction(src::BlockExpr* a, src::BlockExpr* b) -> BlockExpr* {
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
            case Linkage::Local: out += "Local "; return;
            case Linkage::Internal: out += "Internal "; return;
            case Linkage::Exported: out += "Exported "; return;
            case Linkage::Imported: out += "Imported "; return;
            case Linkage::Reexported: out += "Reexported "; return;
            case Linkage::LinkOnceODR: out += "LinkOnceODR "; return;
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

            case K::LocalDecl: {
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

                if (v->sema.ok) {
                    switch (v->ctor.kind) {
                        using enum Constructor::Kind;
                        case Invalid: out += " <invalid-ctor-kind>"; break;
                        case MoveParameter: out += " move"; break;
                        case Zeroinit: out += " zeroinit"; break;
                        case TrivialCopy: out += " trivial"; break;
                        case InitialiserCall: out += " init"; break;
                        case Uninitialised: out += " uninit"; break;
                        case SliceFromParts: out += " slice"; break;
                    }
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
                    case CastKind::Implicit: out += " Implicit"; break;
                    case CastKind::ReferenceToLValue: out += " ReferenceToLValue"; break;
                    case CastKind::LValueRefToLValue: out += " LValueRefToLValue"; break;
                    case CastKind::LValueToRValue: out += " LValueToRValue"; break;
                    case CastKind::LValueToReference: out += " LValueToReference"; break;
                    case CastKind::OptionalNilTest: out += " OptionalNilTest"; break;
                    case CastKind::OptionalUnwrap: out += " OptionalUnwrap"; break;
                    case CastKind::OptionalWrap: out += " OptionalWrap"; break;
                    case CastKind::ArrayToElemRef: out += " ArrayToElemRef"; break;
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

            case K::InvokeExpr: PrintBasicNode("InvokeExpr", e, e->type); return;

            case K::InvokeBuiltinExpr: {
                static const auto String = [](Builtin b) -> std::string_view {
                    switch (b) {
                        case Builtin::New: return "new";
                        case Builtin::Destroy: return "__srcc_destroy";
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

            case K::WhileExpr: PrintBasicNode("WhileExpr", e, nullptr); return;
            case K::ReturnExpr: PrintBasicNode("ReturnExpr", e, nullptr); return;
            case K::AssertExpr: PrintBasicNode("AssertExpr", e, nullptr); return;
            case K::EmptyExpr: PrintBasicNode("EmptyExpr", e, nullptr); return;
            case K::OverloadSetExpr: PrintBasicNode("OverloadSetExpr", e, nullptr); return;
            case K::Nil: PrintBasicNode("Nil", e, nullptr); return;
            case K::IfExpr: PrintBasicNode("IfExpr", e, e->type); return;
            case K::ParenExpr: PrintBasicNode("ParenExpr", e, e->type); return;
            case K::SubscriptExpr: PrintBasicNode("SubscriptExpr", e, e->type); return;
            case K::ConstExpr: PrintBasicNode("ConstExpr", e, e->type); return;
            case K::ExportExpr: PrintBasicNode("ExportExpr", e, e->type); return;
            case K::ImplicitThisExpr: PrintBasicNode("ImplicitThisExpr", e, e->type); return;
            case K::ArrayLiteralExpr: PrintBasicNode("ArrayLiteralExpr", e, e->type); return;

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
                    out += fmt::format(
                        " {}{} {}[{}{}{}] {}{}{}/{}{}\n",
                        C(Cyan),
                        s->name,
                        C(Red),
                        C(Cyan),
                        s->sema.ok ? s->as_type.mangled_name : "???",
                        C(Red),
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
            case K::ProcType:
            case K::ClosureType:
            case K::ArrayType:
            case K::BuiltinType:
            case K::FFIType:
            case K::ReferenceType:
            case K::ScopedPointerType:
            case K::OptionalType:
            case K::IntType:
            case K::SliceType:
            case K::SugaredType:
            case K::ScopedType:
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
            /// These don’t have children.
            case K::BoolLiteralExpr:
            case K::IntegerLiteralExpr:
            case K::StringLiteralExpr:
            case K::ModuleRefExpr:
            case K::EmptyExpr:
                break;

            /// We don’t print children of most types here.
            case K::BuiltinType:
            case K::ProcType:
            case K::ClosureType:
            case K::FFIType:
            case K::ReferenceType:
            case K::ScopedPointerType:
            case K::OptionalType:
            case K::IntType:
            case K::ArrayType:
            case K::SliceType:
            case K::SugaredType:
            case K::ScopedType:
            case K::Nil:
                break;

            case K::StructType: {
                auto s = cast<StructType>(e);
                for (auto& f : s->all_fields) {
                    out += fmt::format(
                        "{}{}{}Field {} {}{} {}at {}{}\n",
                        C(Red),
                        leading_text,
                        &f == &s->all_fields.back() and s->initialisers.empty() ? "└─" : "├─",
                        f.type->as_type.str(use_colour),
                        C(Magenta),
                        f.name,
                        C(Red),
                        C(Yellow),
                        s->sema.ok ? std::to_string(f.offset.bits()) : "?"
                    );
                }

                auto inits = ArrayRef<Expr*>(reinterpret_cast<Expr* const*>(s->initialisers.data()), s->initialisers.size());
                PrintChildren(inits, leading_text);
            } break;

            case K::ProcDecl: {
                auto f = cast<ProcDecl>(e);
                printed_functions.insert(f);
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

            case K::LocalDecl: {
                auto v = cast<LocalDecl>(e);
                PrintChildren(v->init_args, leading_text);
            } break;

            case K::InvokeBuiltinExpr: {
                auto c = cast<InvokeBuiltinExpr>(e);
                PrintChildren(c->args, leading_text);
            } break;

            case K::MemberAccessExpr: {
                auto m = cast<MemberAccessExpr>(e);
                if (m->object) PrintChildren(m->object, leading_text);
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

void src::Module::print_ast(bool use_colour) const {
    /// Ok because ASTPrinter does not attempt to mutate this.
    ASTPrinter{const_cast<Module*>(this), use_colour, true}.print();
}
