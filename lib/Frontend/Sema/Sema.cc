#include <source/Frontend/Sema.hh>

/// ===========================================================================
///  Helpers
/// ===========================================================================
namespace detail {
template <typename Type>
class [[nodiscard]] TCast {
    Type* ty;

public:
    TCast(Type* ty) : ty(ty) {}
    explicit operator bool() { return ty; }

    template <typename To, typename Proj = std::identity>
    auto cast(Proj proj = {}) -> TCast<To> {
        if (not ty) return nullptr;
        auto res = dyn_cast<To>(std::invoke(proj, ty));
        if (not res) return nullptr;
        return {res};
    }

    template <typename Pred>
    auto test(Pred pred) -> TCast<Type> {
        if (not ty) return nullptr;
        if (not std::invoke(pred, ty)) return nullptr;
        return {ty};
    }

    [[nodiscard]] auto get() -> Type* { return ty; }
};
} // namespace detail

template <typename To, typename From>
auto tcast(From from) -> detail::TCast<To> {
    auto res = dyn_cast<To>(from);
    if (not res) return nullptr;
    return {res};
}

#define λ(param, body) ([&](auto&& param) { return body; })

void src::Sema::ConversionSequence::ApplyCast(Sema& S, Expr*& e, CastKind kind, Type to) {
    /// Silently drop lvalue-to-rvalue conversion if the expression
    /// is not an lvalue as we may not have known this when building
    /// this cast.
    if (kind == CastKind::LValueToRValue and not e->is_lvalue) return;
    e = new (S.mod) CastExpr(kind, e, to, e->location);
    S.Analyse(e);
}

void src::Sema::ConversionSequence::ApplyConstExpr(Sema& S, Expr*& e, EvalResult res) {
    e = new (S.mod) ConstExpr(e, std::move(res), e->location);
}

void src::Sema::ConversionSequence::ApplyConstructor(
    [[maybe_unused]] Sema& S,
    [[maybe_unused]] Expr*& e,
    [[maybe_unused]] ProcDecl* ctor,
    [[maybe_unused]] ArrayRef<Expr*> args
) {
    Todo();
}

void src::Sema::ConversionSequence::ApplyOverloadSetToProc(Sema& S, Expr*& e, ProcDecl* proc) {
    auto d = new (S.mod) DeclRefExpr(proc->name, nullptr, e->location);
    d->stored_type = proc->type;
    d->sema.set_done();
    d->decl = proc;
    e = d;
}

void src::Sema::ApplyConversionSequence(Expr*& e, std::same_as<ConversionSequence> auto&& seq) { // clang-format off
    using CS = ConversionSequence;
    for (auto& conv : seq.entries) {
        visit(conv, utils::overloaded {
            [&] (CS::BuildConstExpr) { CS::ApplyConstExpr(*this, e, std::forward_like<decltype(seq)>(*seq.constant)); },
            [&] (CS::BuildCast& cast) { CS::ApplyCast(*this, e, cast.kind, cast.to); },
            [&] (CS::CallConstructor& ctor) { CS::ApplyConstructor(*this, e, ctor.ctor, ctor.args); },
            [&] (CS::OverloadSetToProc& proc) { CS::ApplyOverloadSetToProc(*this, e, proc.proc); }
        });
    }
} // clang-format on

template <bool perform_conversion>
auto src::Sema::ConversionContext<perform_conversion>::cast(CastKind kind, Type to) -> Type {
    if constexpr (perform_conversion) ConversionSequence::ApplyCast(S, *e, kind, to);
    else seq->entries.push_back(ConversionSequence::BuildCast{kind, to});
    if (kind != CastKind::LValueToRValue) score++;
    has_expr = false;
    return to;
}

template <bool perform_conversion>
void src::Sema::ConversionContext<perform_conversion>::lvalue_to_rvalue() {
    /// Type of LValueToRValue casts is set later.
    std::ignore = cast(CastKind::LValueToRValue, Type::Unknown);
}

template <bool perform_conversion>
auto src::Sema::ConversionContext<perform_conversion>::overload_set_to_proc(ProcDecl* proc) -> Type {
    if constexpr (perform_conversion) ConversionSequence::ApplyOverloadSetToProc(S, *e, proc);
    else seq->entries.push_back(ConversionSequence::OverloadSetToProc{proc});
    has_expr = false;
    score++;
    return proc->type;
}

template <bool perform_conversion>
auto src::Sema::ConversionContext<perform_conversion>::replace_with_constant(EvalResult&& res) -> Type {
    auto type = res.type;

    if constexpr (perform_conversion) ConversionSequence::ApplyConstExpr(S, *e, std::move(res));
    else {
        Assert(not seq->constant.has_value(), "At most one ConstExpr per conversion sequence!");
        seq->constant = std::move(res);
        seq->entries.push_back(ConversionSequence::BuildConstExpr{});
    }

    has_expr = false;
    score++;
    return type;
}

template <bool perform_conversion>
bool src::Sema::ConversionContext<perform_conversion>::try_evaluate(EvalResult& out) {
    if (has_expr) return S.Evaluate(*e, out, false);
    if (not perform_conversion) {
        if (std::holds_alternative<ConversionSequence::BuildConstExpr>(seq->entries.back())) {
            out = *seq->constant;
            return true;
        }
    }

    /// Not a constant expression.
    return false;
}

bool src::Sema::ClassifyParameter(ParamInfo* info) {
    if (info->sema.analysed) return info->sema.ok;
    if (not AnalyseAsType(info->type) or not MakeDeclType(info->type))
        return info->sema.set_errored();

    /// Cannot perform type inference on parameters.
    if (info->type == Type::Unknown) {
        info->sema.set_errored();
        return Error(info->type.ptr, "Type inference is not permitted here");
    }

    /// Determine whether this type is cheap to copy (and whether it can
    /// be copied trivially).
    auto CheapToCopy = [&] {
        return info->type.trivially_copyable and
               info->type.size(ctx) <= Size::Bits(128); /// FIXME: Target-dependent.
    };

    /// Determine whether this should be passed by value or by reference.
    switch (info->intent) {
        /// By value if cheap to copy; by reference otherwise.
        case Intent::Copy:
            info->cls = CheapToCopy() ? ParamInfo::Class::ByVal : ParamInfo::Class::CopyAsRef;
            return true;

        /// Always by value. We let LLVM take care of what this actually
        /// means. This is different from Copy because we may not agree
        /// with C++ as to what is considered cheap to copy.
        case Intent::CXXByValue:
            info->cls = ParamInfo::Class::ByVal;
            return true;

        /// Always by reference.
        case Intent::Inout:
        case Intent::Out:
            info->cls = ParamInfo::Class::LValueRef;
            return true;

        /// By value if trivially copyable and small.
        case Intent::Move:
        case Intent::In:
            info->cls = CheapToCopy() ? ParamInfo::Class::ByVal : ParamInfo::Class::AnyRef;
            return true;
    }

    Unreachable();
}

template <bool perform_conversion>
bool src::Sema::ConvertImpl(
    ConversionContext<perform_conversion>& ctx,
    Type from_ty,
    Type type
) {
    /// Sanity checks.
    if (ctx.expr and (ctx.expr->sema.errored or type->sema.errored)) return false;
    Assert(from_ty->sema.ok and type->sema.ok, "Cannot convert to unanalysed type");
    Assert(isa<TypeBase>(type));
    auto to = type.desugared;
    auto from = from_ty;

    /// If the types are equal, then they’re convertible to one another.
    if (from == to) return true;

    /// Active optionals are convertible to the type they contain. There
    /// are no other valid conversions involving optionals at the moment.
    if (auto ty = LValueState.GetActiveOptionalType(ctx.expr)) {
        from = ctx.cast(CastKind::OptionalUnwrap, ty->elem);
        return ConvertImpl<perform_conversion>(ctx, from, type);
    }

    /// Place conversions involving references first, as we may
    /// have to chain several conversions to get e.g. from an
    /// `i32&` to an `i64`.
    if (
        auto to_ref = dyn_cast<ReferenceType>(to);
        to_ref or isa<ReferenceType, ScopedPointerType>(from)
    ) {
        auto from_base = from.strip_refs_and_pointers;
        auto to_base = to.strip_refs;
        auto from_depth = from.ref_depth;
        auto to_depth = to.ref_depth;

        /// Reference to void& conversion.
        if (from_depth == to_depth and to_base == Type::Void) {
            from = ctx.cast(CastKind::Implicit, to);
            return true;
        }

        /// Base types are equal.
        else if (from_base == to_base) {
            /// If the depth we’re converting to is one greater than
            /// the depth of the expression, and the expression is an
            /// lvalue, then this is reference binding.
            if (to_depth == from_depth + 1 and ctx.is_lvalue)
                from = ctx.cast(CastKind::LValueToReference, new (mod) ReferenceType(from, {}));

            /// If the depth of the type we’re converting to is less than
            /// the depth of the type we’re converting from, then this is
            /// implicit dereferencing.
            else if (to_depth < from_depth) {
                for (isz i = from_depth - to_depth; i; i--)
                    from = ctx.cast(CastKind::ReferenceToLValue, cast<ReferenceType>(from)->elem);
            }
        }

        /// Array references are convertible to references to the base type.
        else if (
            auto from_ref = dyn_cast<ReferenceType>(from);
            to_ref and
            from_ref and
            tcast<ArrayType>(from_ref->elem).test λ(a, a->elem == to_ref->elem)
        ) {
            from = ctx.cast(CastKind::ReferenceToLValue, cast<ReferenceType>(from)->elem);
            from = ctx.cast(CastKind::ArrayToElemRef, type);
            return true;
        }

        /// Array lvalues can be converted to references to their first element.
        else if (
            auto array = dyn_cast<ArrayType>(from);
            ctx.is_lvalue and array and array->elem == to_ref->elem
        ) {
            from = ctx.cast(CastKind::ArrayToElemRef, type);

            /// This involves both adding a reference level and changing the
            /// type, so make this a worse conversion than either one on its
            /// own.
            ctx.score++;
            return true;
        }
    }

    /// Check for equality one more time.
    if (from == to) return true;

    /// Procedures are convertible to closures, but *not* the other way around.
    if (isa<ProcType>(from) and isa<ClosureType>(to)) {
        if (from != cast<ClosureType>(to)->proc_type) return false;
        ctx.lvalue_to_rvalue();
        ctx.cast(CastKind::Implicit, type);
        return true;
    }

    /// Overload sets are convertible to each of the procedure types in the
    /// set, as well as any closures thereof.
    if (auto os = dyn_cast_if_present<OverloadSetExpr>(ctx.expr)) {
        auto to_proc = dyn_cast<ProcType>(isa<ClosureType>(to) ? cast<ClosureType>(to)->proc_type : to);
        if (not to_proc) return false;
        for (auto o : os->overloads) {
            if (o->type == to_proc) {
                /// Replace the overload set with a DeclRefExpr to the referenced procedure...
                from = ctx.overload_set_to_proc(o);

                /// ... and make it a closure, if need be.
                if (isa<ClosureType>(to)) {
                    ctx.lvalue_to_rvalue();
                    from = ctx.cast(CastKind::Implicit, type);
                }

                return true;
            }
        }

        return false;
    }

    /// Integer-to-integer conversions.
    if (from.is_int(true) and to.is_int(true)) {
        auto from_size = from.size(this->ctx);
        auto to_size = to.size(this->ctx);

        /// Try evaluating the expression. We allow implicit conversions
        /// to a smaller type if the value is known to fit at compile time.
        if (ctx.expr) {
            EvalResult value;
            if (ctx.try_evaluate(value) and value.as_int().getActiveBits() <= to_size.bits()) {
                value.type = to;

                /// Even though the *value* might require fewer bits than what we’re
                /// converting to, the *type* (i.e. bit width) of the APInt might still
                /// be larger, so we might have to truncate or extend independently of
                /// that.
                value.as_int() = value.as_int().sextOrTrunc(unsigned(to_size.bits()));
                from = ctx.replace_with_constant(std::move(value));
                return true;
            }
        }

        /// Smaller integer types can be converted to larger integer types.
        if (from_size <= to_size) {
            ctx.lvalue_to_rvalue();
            from = ctx.cast(CastKind::Implicit, type);
            return true;
        }

        /// No other valid integer conversions.
        return false;
    }

    /// Enumerations are convertible to their derived enums.
    if (isa<EnumType>(from) and isa<EnumType>(to)) {
        auto to_enum = cast<EnumType>(to);
        for (auto parent = to_enum->parent_enum; parent; parent = parent->parent_enum) {
            if (parent == from) {
                ctx.lvalue_to_rvalue();
                from = ctx.cast(CastKind::Implicit, type);
                return true;
            }
        }
    }

    /// Array lvalues are convertible to slices.
    if (
        ctx.is_lvalue and
        isa<ArrayType>(from) and
        isa<SliceType>(to) and
        cast<ArrayType>(from)->elem == cast<SliceType>(to)->elem
    ) {
        from = ctx.cast(CastKind::Implicit, type);
        return true;
    }

    /// Nil is convertible to any optional type.
    if (from.is_nil and isa<OptionalType>(to)) {
        ctx.lvalue_to_rvalue();
        from = ctx.cast(CastKind::Implicit, type);
        return true;
    }

    /// Any type is convertible to the optional type of that type.
    if (auto opt = dyn_cast<OptionalType>(to); opt and from == opt->elem) {
        ctx.lvalue_to_rvalue();
        from = ctx.cast(CastKind::OptionalWrap, opt);
        return true;
    }

    /// If we need to construct an array or struct, try temporary materialisation.
    /// TODO: Actually support this w/ TryConvert()...
    if constexpr (perform_conversion) {
        if (isa<ArrayType, StructType, TupleType>(to.desugared)) {
            if (not MaterialiseTemporary(*ctx.e, to)) return false;
            ctx.score++;
            return true;
        }
    }

    /// No other conversions are supported.
    return false;
}

bool src::Sema::Convert(Expr*& e, Type type, bool lvalue) {
    ConversionContext<true> ctx{*this, e};
    bool ok = ConvertImpl(ctx, e->type, type);
    if (ok and not lvalue) InsertLValueToRValueConversion(e);
    return ok;
}

auto src::Sema::CreateImplicitDereference(Expr* e, isz depth) -> Expr* {
    for (isz i = depth; i; i--) {
        e = new (mod) CastExpr(
            CastKind::ReferenceToLValue,
            e,
            cast<ReferenceType>(e->type)->elem,
            e->location
        );

        Analyse(e);
    }

    return e;
}

auto src::Sema::DeclContext::find(Sema& S, String name) -> Entry {
    if (auto e = dyn_cast<EnumType>(scope)) return S.SearchEnumScope(e, name);
    Unreachable("Invalid decl context");
}

bool src::Sema::EnsureCondition(Expr*& e) {
    /// Optionals can be tested for nil.
    if (isa<OptionalType>(e->type)) {
        InsertLValueToRValueConversion(e);
        e = new (mod) CastExpr(CastKind::OptionalNilTest, e, Type::Bool, e->location);
        Analyse(e);
        return true;
    }

    /// Try other possible conversions to bool.
    UnwrapInPlace(e);
    if (Convert(e, Type::Bool)) return true;
    return Error(
        e->location,
        "Type '{}' of condition must be convertible to '{}'",
        e->type.str(mod->context->use_colours, true),
        Type::Bool.str(mod->context->use_colours, true)
    );
}

bool src::Sema::FinaliseInvokeArgument(Expr*& arg, const ParamInfo* param) {
    /// Variadic argument.
    if (not param) {
        /// Variadic arguments cannot be overload sets.
        if (arg->type == Type::OverloadSet) return Error(
            arg,
            "Cannot pass an overload set as a variadic argument"
        );

        /// Variadic arguments are always rvalues.
        InsertLValueToRValueConversion(arg);
        return true;
    }

    /// Check value category and perform temporary materialisation
    /// or copying if need be.
    switch (param->cls) {
        /// Only lvalues are allowed here.
        case ParamInfo::Class::LValueRef:
            if (not arg->is_lvalue) return Error(
                arg,
                "rvalue argument cannot bind to a parameter with intent '{}'. "
                "Try passing a variable instead.",
                stringify(param->intent)
            );
            return true;

        /// Lvalues are passed as-is. Rvalues undergo temporary materialisation.
        case ParamInfo::Class::AnyRef:
            if (not arg->is_lvalue) return MaterialiseTemporary(arg, param->type);
            return true;

        /// Requires an rvalue.
        case ParamInfo::Class::ByVal:
            InsertLValueToRValueConversion(arg);

            /// If the argument is a trivial ConstructExpr, pass through
            /// the underlying value. Otherwise, materialise a temporary
            /// to construct into and load from it.
            if (auto c = dyn_cast<ConstructExpr>(arg)) {
                if (c->ctor_kind == ConstructKind::Copy and c->args().size() == 1) arg = c->args().front(); /// FIXME: fidelity.
                else if (not MaterialiseTemporary(arg, param->type)) return false;
                InsertLValueToRValueConversion(arg);
            }

            return true;

        /// Create a copy on the stack and pass it by reference.
        case ParamInfo::Class::CopyAsRef:
            return MaterialiseTemporary(arg, param->type);
    }

    Unreachable("Invalid param class");
}

void src::Sema::InsertImplicitDereference(Expr*& e, isz depth) {
    e = CreateImplicitDereference(e, depth);
}

void src::Sema::InsertLValueToRValueConversion(Expr*& e) {
    if (not e->sema.errored and e->is_lvalue) {
        Expr* cast = new (mod) CastExpr(
            CastKind::LValueToRValue,
            e,
            Type::Unknown,
            e->location
        );

        Analyse(cast);
        e = cast;
    }
}

bool src::Sema::IsInParameter(Expr* e) {
    auto p = dyn_cast<ParamDecl>(e->ignore_paren_refs);
    return p and p->info->intent == Intent::In;
}

template <bool in_array>
bool src::Sema::MakeDeclType(Type& e) {
    if (not AnalyseAsType(e)) return false;
    if (e == Type::Void) {
        if constexpr (in_array) return Error(e.ptr, "Cannot declare an array of type 'void'");
        else return Error(e.ptr, "Cannot declare a variable of type 'void'");
    }

    if (auto o = dyn_cast<OpaqueType>(e)) {
        if constexpr (in_array) return Error(e.ptr, "Cannot declare an array of incomplete type '{}'", o->name);
        else return Error(e.ptr, "Cannot declare a variable of incomplete type '{}'", o->name);
    }

    /// Procedure types decay to closures.
    if (auto p = dyn_cast<ProcType>(e)) e = new (mod) ClosureType(p);
    return true;
}

bool src::Sema::MaterialiseTemporary(Expr*& e, Type type) {
    /// If this is already a temporary, keep it instead of creating *another* copy.
    if (isa<MaterialiseTemporaryExpr>(e) and e->type == type) return true;

    /// Actually create a temporary.
    auto ctor = Construct(e->location, type, {e}, nullptr);
    if (not ctor) return false;
    e = new (mod) MaterialiseTemporaryExpr(type, ctor, e->location);
    return true;
}

void src::Sema::MaterialiseTemporaryIfRValue(Expr*& e) {
    if (e->is_lvalue) return;
    Assert(MaterialiseTemporary(e, e->type));
}

auto src::Sema::SearchEnumScope(
    EnumType* enum_type,
    String name
) -> DeclContext::Entry {
    for (auto it = enum_type; it; it = it->parent_enum) {
        if (auto [syms, escapes] = it->scope->find(name, true)) {
            Assert(not escapes, "Should never be set for enum lookups");
            Assert(syms->size() == 1, "Enum scope entries must contain exactly one symbol");

            /// Set the type of this to the underlying type if we are still analysing
            /// the enum, or to the type of the enum that we searched (!) for the enumerator
            /// otherwise.
            const bool open = rgs::contains(open_enums, enum_type);
            Type ty = open ? enum_type->underlying_type : enum_type;
            return {syms->front(), ty};
        }
    }

    return DeclContext::Entry{};
}

bool src::Sema::TryConvert(ConversionSequence& seq, Expr* e, Type to, bool lvalue) {
    ConversionContext<false> ctx(*this, seq, &e);
    bool ok = ConvertImpl(ctx, e->type, to);
    seq.score = ok ? ctx.score : Candidate::InvalidScore;
    if (ok and not lvalue) ctx.lvalue_to_rvalue();
    return ok;
}

auto src::Sema::Unwrap(Expr* e, bool keep_lvalues) -> Expr* {
    Assert(e->sema.ok, "Unwrap() called on broken or unanalysed expression");

    /// Unwrap active optionals.
    if (auto opt = LValueState.GetActiveOptionalType(e)) {
        e = new (mod) CastExpr(CastKind::OptionalUnwrap, e, opt->elem, e->location);
        Analyse(e);
        return Unwrap(e, keep_lvalues);
    }

    /// Load references and pointers.
    if (isa<ReferenceType, ScopedPointerType>(e->type)) {
        InsertImplicitDereference(e, 1);
        return Unwrap(e, keep_lvalues);
    }

    /// Convert to an rvalue, if requested.
    if (not keep_lvalues) InsertLValueToRValueConversion(e);
    return e;
}

void src::Sema::UnwrapInPlace(Expr*& e, bool keep_lvalues) {
    e = Unwrap(e, keep_lvalues);
}

auto src::Sema::UnwrappedType(Expr* e) -> Type {
    if (auto opt = LValueState.GetActiveOptionalType(e)) return opt->elem.strip_refs_and_pointers;
    return e->type.strip_refs_and_pointers;
}

/// ===========================================================================
///  Optional tracking.
/// ===========================================================================
src::Sema::LValueState::ScopeGuard::ScopeGuard(src::Sema& S)
    : S(S),
      previous(std::exchange(S.LValueState.guard, this)) {}

src::Sema::LValueState::ScopeGuard::~ScopeGuard() {
    for (auto&& [e, old_state] : changes) S.LValueState.tracked[e] = std::move(old_state);
    S.LValueState.guard = previous;
}

src::Sema::LValueState::OptionalActivationGuard::OptionalActivationGuard(Sema& S, Expr* expr)
    : S(S), expr(expr) {
    S.LValueState.ActivateOptional(expr);
}

src::Sema::LValueState::OptionalActivationGuard::~OptionalActivationGuard() {
    S.LValueState.DeactivateOptional(expr);
}

auto src::Sema::LValueState::GetObjectPath(MemberAccessExpr* m) -> std::pair<LocalDecl*, Path> {
    if (not m or not m->field or not m->object) return {nullptr, {}};

    /// Find object root.
    Path path;
    Expr* object;
    do {
        auto f = dyn_cast<FieldDecl>(m->field);
        if (not f) return {nullptr, {}};
        path.push_back(f->index);
        object = m->object;
    } while (m = dyn_cast<MemberAccessExpr>(m->object->ignore_parens), m);

    /// Got a local.
    if (auto var = dyn_cast<LocalDecl>(object->ignore_paren_refs)) {
        Assert(isa<StructType>(var->type.desugared), "Only structs can contain paths");
        rgs::reverse(path);
        return {var, std::move(path)};
    }

    return {nullptr, {}};
}

void src::Sema::LValueState::ChangeOptionalState(Expr* e, auto cb) {
    if (not e) return;
    e = e->ignore_paren_refs;

    if (auto var = dyn_cast<LocalDecl>(e)) {
        if (not guard->changes.contains(var)) guard->changes[var] = tracked[var];
        std::invoke(cb, var);
        return;
    }

    if (auto m = dyn_cast<MemberAccessExpr>(e)) {
        if (auto [var, path] = GetObjectPath(m); var) {
            Assert(not path.empty());
            if (not guard->changes.contains(var)) guard->changes[var] = tracked[var];
            std::invoke(cb, var, std::move(path));
        }
    }
}

void src::Sema::LValueState::ActivateOptional(Expr* e) { // clang-format off
    ChangeOptionalState(e, utils::overloaded {
        [&] (LocalDecl* var) { tracked[var].active_optional = true; },
        [&] (LocalDecl* var, Path path) {
            /// Activating a field only makes sense if the thing
            /// itself is active, so if we get here, it must be
            /// active.
            tracked[var].active_optional = true;
            tracked[var].active_fields.push_back(std::move(path));
        }
    });
} // clang-format on

void src::Sema::LValueState::DeactivateOptional(Expr* e) { // clang-format off
    ChangeOptionalState(e, utils::overloaded {
        [&] (LocalDecl* var) { tracked[var].active_optional = false; },
        [&] (LocalDecl* var, Path path) {
            /// Delete all paths that *start with* this path, as nested
            /// objects are now part of a different object.
            llvm::erase_if(tracked[var].active_fields, [&](Path& p) {
                return utils::starts_with(p, path);
            });
        }
    });
} // clang-format on

auto src::Sema::LValueState::GetActiveOptionalType(Expr* e) -> OptionalType* {
    if (not e) return nullptr;
    e = e->ignore_paren_refs;

    if (auto var = dyn_cast<LocalDecl>(e)) {
        auto obj = tracked.find(var);
        if (obj == tracked.end() or not obj->second.active_optional) return nullptr;
        return dyn_cast<OptionalType>(var->type.desugared);
    }

    if (auto m = dyn_cast<MemberAccessExpr>(e)) {
        if (auto [var, path] = GetObjectPath(m); var) {
            Assert(not path.empty());
            auto obj = tracked.find(var);
            if (obj == tracked.end() or not obj->second.active_optional) return nullptr;
            if (
                rgs::any_of(
                    obj->second.active_fields,
                    [&](Path& p) { return utils::starts_with(p, path); }
                )
            ) {
                auto s = cast<StructType>(var->type.desugared);
                FieldDecl* f{};
                for (auto idx : path) {
                    f = s->fields[idx];
                    s = dyn_cast<StructType>(f->type.desugared);
                }

                return dyn_cast<OptionalType>(f->type.desugared);
            }
        }
    }

    return nullptr;
}

auto src::Sema::LValueState::MatchOptionalNilTest(Expr* test) -> Expr* {
    auto c = dyn_cast<CastExpr>(test->ignore_lv2rv);
    if (
        c and
        c->cast_kind == CastKind::OptionalNilTest
    ) {
        auto o = c->operand->ignore_lv2rv;
        if (auto local = dyn_cast<LocalRefExpr>(o)) return local->decl;
        if (auto member = dyn_cast<MemberAccessExpr>(o)) return member;
    }

    return nullptr;
}

void src::Sema::LValueState::SetDefinitelyMoved(Expr* expr) {
    if (not expr) return;
    auto e = expr->ignore_paren_refs;

    /// Ignore trivially copyable types and non-lvalues.
    if (e->type.trivially_copyable) return;
    if (not e->is_lvalue) return;

    /// Check if the variable has already been moved.
    auto AlreadyMoved = [&](LocalDecl* var) {
        if (not var->already_moved) return false;
        S->Error(
            expr->location,
            "Variable '{}' has already been {}moved from",
            var->name,
            var->partially_moved ? "partially" : ""
        );
        S->Note(last_moves[var].loc, "Previous move was here");
        return true;
    };

    if (auto var = dyn_cast<LocalDecl>(e)) {
        if (AlreadyMoved(var)) return;
        var->definitely_moved = true;
        last_moves[var].loc = expr->location;
    }

    if (auto mem = dyn_cast<MemberAccessExpr>(e)) {
        if (auto [var, path] = GetObjectPath(mem); var) {
            Assert(not path.empty());
            if (AlreadyMoved(var)) return;

            /// If the field we’re trying to move—or one of its subobjects—has already
            /// been moved, then we can’t do this.
            auto entry = llvm::find_if(var->partially_moved_fields, [&](Path& p) {
                return utils::starts_with(p, path);
            });

            /// Got one.
            if (entry != var->partially_moved_fields.end()) {
                S->Error(expr->location, "Cannot moved already moved-from value");
                S->Note(last_moves[var].subobjects[*entry], "Previous move was here");
                return;
            }

            /// Ok.
            last_moves[var].subobjects[path] = expr->location;
            var->partially_moved_fields.push_back(std::move(path));
        }
    }
}

/// ===========================================================================
///  Overload Resolution
/// ===========================================================================
src::Sema::OverloadResolutionResult::ResolutionFailure::~ResolutionFailure() {
    using enum utils::Colour;
    if (suppress_diagnostics) return;
    utils::Colours C{S.ctx->use_colours};
    S.Error(where, "Overload resolution failed");

    /// Print all argument types.
    fmt::print(stderr, "  {}{}Arguments:\n", C(Bold), C(White));
    for (auto [i, e] : llvm::enumerate(args))
        fmt::print(stderr, "    {}{}{}. {}\n", C(Bold), C(White), i + 1, e->type.str(S.mod->context->use_colours, true));

    /// Print overloads and why each one was invalid.
    fmt::print(stderr, "\n  {}{}Overloads:\n", C(Bold), C(White));
    for (auto [i, o] : llvm::enumerate(overloads)) {
        if (i != 0) fmt::print("\n");
        fmt::print(
            stderr,
            "    {}{}{}.{} {}{}{}\n",
            C(Bold),
            C(White),
            i + 1,
            C(Reset),
            o.proc->type.str(S.mod->context->use_colours, true),
            C(White),
            C(Reset)
        );

        if (o.proc->location.seekable(S.mod->context)) {
            auto lc = o.proc->location.seek_line_column(S.mod->context);
            fmt::print(
                stderr,
                "       at {}:{}:{}\n",
                S.mod->context->file(o.proc->location.file_id)->path().string(),
                lc.line,
                lc.col
            );
        }

        fmt::print(stderr, "       ");
        switch (o.s) {
            case Candidate::Status::ArgumentCountMismatch:
                fmt::print(stderr, "Requires {} arguments\n", o.type->parameters.size());
                break;

            case Candidate::Status::NoViableArgOverload:
                fmt::print(stderr, "Overload set for argument {} contains no viable overload\n", o.mismatch_index + 1);
                break;

            case Candidate::Status::ArgumentTypeMismatch:
                fmt::print(
                    stderr,
                    "Incompatible type for argument {}\n",
                    o.mismatch_index + 1
                );
                break;

            /// Viable here means that the overload was ambiguous.
            case Candidate::Status::Viable:
                fmt::print("Viable, but ambiguous\n");
                break;
        }
    }
}

auto src::Sema::PerformOverloadResolution(
    Location where,
    ArrayRef<ProcDecl*> overloads,
    MutableArrayRef<Expr*> args
) -> OverloadResolutionResult {
    /// First, analyse all arguments.
    for (auto& a : args)
        if (not Analyse(a))
            return OverloadResolutionResult::IllFormed{};

    /// Create a candidate for each element of the overload set; each
    /// candidate is initially viable.
    SmallVector<Candidate> candidates;
    candidates.reserve(overloads.size());
    for (auto p : overloads) candidates.emplace_back(p);

    /// Determine which candidates are viable.
    for (auto& ci : candidates) {
        auto& params = ci.type->parameters;
        ci.arg_convs.resize(args.size());

        /// If there are not enough arguments, the candidate is not viable.
        if (params.size() > args.size()) {
            ci.s = Candidate::Status::ArgumentCountMismatch;
            continue;
        }

        /// If there are too many arguments, and the candidate is not a variadic
        /// procedure, the candidate is not viable.
        if (not ci.type->variadic and params.size() < args.size()) {
            ci.s = Candidate::Status::ArgumentCountMismatch;
            continue;
        }

        /// Otherwise, check that all non-variadic arguments (i.e. arguments that
        /// actually bind to parameters) are convertible to the corresponding
        /// parameter, noting the conversion sequence for each argument.
        ///
        /// If an argument cannot be converted to a parameter, the candidate is
        /// not viable. Otherwise, a candidate’s score is the sum of the scores
        /// of all its argument conversions.
        for (usz j = 0; j < params.size(); j++) {
            if (not TryConvert(ci.arg_convs[j], args[j], params[j].type)) {
                ci.s = Candidate::Status::ArgumentTypeMismatch;
                ci.mismatch_index = j;
                break;
            } else {
                ci.score += ci.arg_convs[j].score;
            }
        }
    }

    /// Find the viable candidate with the minimum score.
    auto min = utils::UniqueMin(
        candidates,
        [](auto& c) { return c.s == Candidate::Status::Viable; },
        &Candidate::score
    );

    /// If there are no viable candidates, or there is more than one
    /// candidate with the minimum score, the program is ill-formed.
    if (min == candidates.end()) return OverloadResolutionResult::ResolutionFailure{
        *this,
        where,
        std::move(candidates),
        args
    };

    /// Apply conversions from arguments to parameter types.
    auto proc = min->proc;
    auto& params = cast<ProcType>(proc->type)->parameters;
    for (usz i = 0; i < args.size(); i++) {
        const bool variadic = i >= params.size();
        if (not variadic) ApplyConversionSequence(args[i], std::move(min->arg_convs[i]));
        if (not FinaliseInvokeArgument(args[i], variadic ? nullptr : &params[i]))
            return OverloadResolutionResult::IllFormed{};
    }

    return OverloadResolutionResult::Ok{proc};
}

/// ===========================================================================
///  Unwinding
/// ===========================================================================
/// A direct branch transfers control flow from one point in the program
/// to another; care must be taken that, in so doing, we do not corrupt
/// the program state. To do this effectively, we have to introduce a
/// number of terms and concepts.
///
/// Some expressions, when executed, ‘register’ some action to be taken
/// when the scope that contains them is exited. For instance, deferred
/// expressions ‘register’ the execution of the deferred material and
/// local variable declarations (may) register a destructor call. For the
/// sake of brevity, we shall refer to such expressions as *protected*.
///
/// Consider now a situation such as the following:
///
/// \code
/// goto label;
/// {
///     var x = ... /// Construct an x.
///     label:
///     print x;
/// }
/// \endcode
///
/// The branch to `label` in this example would skip the initialisation
/// of `x`, and not only cause the `print` call to use an uninitialised
/// variable, but even without that call, the destructor of `x` would
/// still execute at end of scope and attempt to delete an uninitialised
/// variable. This branch is therefore ill-formed, and it shows that, in
/// general, control flow must not ‘jump over’ (that is, move forward
/// across) a protected expression.
///
/// However, this does not mean that all jumps become ill-formed as soon
/// as protected expressions are involved:
///
/// \code
/// {
///     again:
///     var x = ... /// Construct an x.
///     print x;
///     goto again;
/// }
/// \endcode
///
/// This jump is well-formed, so long as it always calls the destructor
/// of `x`. To explain why this is the case, we need to adjust our
/// definition of a protected expression: instead of requiring that
/// the actions ‘registered’ by such expressions are executed whenever
/// the scope containing them is exited, we instead say that they are
/// executed whenever the *rest* of the scope after that expression
/// is exited; this includes cases where the same scope is reëntered
/// further up before the protected expression.
///
/// These examples show that not all branches are created equal; in
/// fact, we can categorise branches according to two properties: a
/// branch can involve upwards and downwards movement, sometimes even
/// both, but it can only move forwards or backwards.
///
/// To elaborate, a branch involves a source (e.g. a GotoExpr) and a target
/// (a LabelExpr); either the source or target are in the same scope,
/// or they aren’t. If they are, then the branch moves either forwards
/// or backwards, and there is no upwards or downwards movement.
///
/// If they aren’t, then the source and the target have some NCA (nearest
/// common ancestor) scope. To transfer control flow from the source to
/// the target, we must unwind from the source up to the NCA, then move
/// either forward or backward in the NCA—depending on whether the Goto
/// precedes the Label or vice versa—and then move down to the label.
///
/// The latter can be illustrated with the following example:
///
/// \code
/// { nested: }
/// /// ...
/// { goto nested; }
/// \endcode
///
/// Here, the AST of this program looks something like this:
///
/// \code
///  BlockExpr (NCA)
///  ├─BlockExpr
///  │ └─LabelExpr (Target)
///  ⋮
///  └─BlockExpr
///    └─GotoExpr (Source)
/// \endcode
///
/// In order to handle such branches in a sane manner, it helps to
/// operate based on the assumption that control flow never actually
/// ‘jumps’ around randomly, but instead always flows from a parent
/// scope to a direct child or vice versa: in the case above, we do not
/// simply jump from the source to the target, but rather, we move
/// up one scope from the source into the NCA block, then backwards
/// in that block to the BlockExpr containing the target, and then down
/// to the target.
///
/// Decomposing jumps into downwards, upwards, and forwards/backwards
/// movement like this allows us to reason about each part of the jump
/// separately. This categorisation yields four different kinds of jumps:
///
/// 1. jumps that involve no upwards or downwards movement and thus only
///    move backwards or forwards in the same scope, which we shall call
///    *same-scope jumps*;
///
/// 2. jumps that move upwards, but not downwards (that is, the source is
///    a (direct) child of the target), leaving scopes until control flow
///    reaches the scope of the target, which we shall call *upwards jumps*;
///
/// 3. jumps that, conversely, move downwards, but not upwards (that is,
///    the target is a (direct) child of the source), entering scopes until
///    control flow reaches the scope of the target, which we shall call
///    *downwards jumps*; and finally
///
/// 4. jumps that move both upwards and downwards (that is, the target and
///    source are unrelated, save that they have some common ancestor), which
///    we shall call *cross jumps*.
///
/// However, instead of analysing each kind of jump separately, we can
/// combine some of the logic: first, note that a cross jump needs to move
/// from the source up to the NCA, then forward or backward in the NCA,
/// and then down from the NCA to the target. Thus, a cross jump can be
/// decomposed into combinations of the other three.
///
/// Furthermore, upwards jumps involve leaving scopes; this means that we
/// conceptually move to the beginning of a scope, and then up to its parent;
/// downwards scopes do something similar, except that they move from the
/// start of a scope to somewhere in the middle of it in order to enter a
/// nested scope.
///
/// Using this decomposition analysis, we can reduce all the jump kinds above
/// to combinations of four primitives:
///
/// 1. Moving forward within a scope.
/// 2. Moving backward within a scope.
/// 3. Leaving a scope from its start.
/// 4. Entering a scope at its start.
///
/// We established in one of the examples above that forwards movement across
/// a protected expression is forbidden; however, since this can only really
/// happen when executing the first of these four primitives, we only need to
/// check for that in portions of a jump that correspond to that primitive. This
/// is one example of how this analysis allows us to reason about jumps step
/// by step.
///
/// Conversely, moving backwards is always fine, but if we move backwards across
/// a protected expression, we need to *unwind* it (e.g. call the destructor of
/// a variable or execute a deferred expression).
///
/// Lastly, since we only enter and leave scopes at their starting points, the
/// last two operations cannot possibly cross a protected expression, and we
/// thus don’t have to consider them at all.
///
/// As an aside, forwards/backwards movement requires us to know what expressions
/// precede other expressions in a scope; this may not be obvious in cases such as
/// the following:
///
/// \code
/// {
///     int x;
///     back: foo (int y); /// Declare `y` and pass it as an arg to `foo`.
///     x = y + { goto back; };
/// }
/// \endcode
///
/// This is rather problematic since it would involve decomposing arbitrarily
/// complex subexpressions and searching them for protected expressions; while
/// potentially doable, it isn’t particularly useful. A deferred expression in
/// the middle of another expression is a fairly stupid construct; furthermore,
/// labels in the middle of expressions are similarly problematic, as what would
/// a branch to `foo` in `x = 3 + foo: 4` even mean?
///
/// A solution to this problem is to forbid certain expressions in any position
/// except at the top-level of an (implicit) block expression; this is one of
/// the reasons why we introduced the concept of a Statement in the grammar even
/// though this language is mainly expression-based. See the grammar for more
/// information on this.
///
/// TODO: Once we have temporary materialisation, deal w/ destroying temporaries
///       properly. Does that influence us being able to branch *out* of the middle
///       of an expression? What about initialisers? Can we branch out of those?
///
///       Answer: For simplicity’s stake, we should make `goto` a statement as well
///       and disallow branching into or out of BlockExprs in expression position
///       (i.e. which are not statements). This way, we never end up with half an
///       expression.
///
/// Now that we have come to a detailed understanding of how jumps work, let
/// us consider an actual algorithm for validating jumps (and determining what,
/// if anything, we need to unwind).
///
/// First, to simplify the implementation of this, we don’t actually decompose
/// jumps all the way to just forwards and backwards movement, but instead
/// combine some of the principles described above. Still, we need to determine
/// what kind of jump we are dealing with, so we first find the NCA of the source
/// and target scopes.
///
/// Next, since downwards movement involves checking that we don’t cross a protected
/// expression, but no unwinding, we can get it out of the way early: In cases where
/// downwards movement is involved (that is, if the target is not the NCA), instead
/// of moving down from the NCA to the target, we instead move up from the target
/// to the NCA; this means we only need to implement moving up the scope stack, not
/// down. Note, however, that his is *conceptually* still downwards movement.
///
/// Now, we only have to consider potential upwards movement from the source to the
/// NCA, as well as forwards or backwards movement within the NCA. As we established
/// before, upwards movement across several scopes is just moving back to the start
/// of a scope, moving up one scope, and repeating both until we have reached the
/// target scope. Thus, if any of the expressions that this involves moving over
/// contain protected subexpressions, we need to take care to unwind them.
///
/// Finally, after processing all downwards and upwards movement, we now need to
/// take care of any remaining forwards and backwards movement.
///
/// This algorithm is implemented in Unwind(), UnwindLocal(), and ValidateDirectBr()
/// below, with the entry point being the latter. UnwindUpTo() is used solely to
/// unwind entire scopes (i.e. it only requires upwards/backwards movement) and
/// can thus never error. This is used for other types of control flow (viz. break,
/// continue, return).

/// \brief Handle forwards/backwards movement within a single block.
///
/// If we’re moving forwards, \p ctx is a pointer to the branch
/// from which the forwards movement originates; in this case,
/// crossing protected expressions is invalid, so we emit an error
/// if we find any.
///
/// If we’re moving backwards, \p ctx is a pointer to a vector to
/// which we append any protected expressions we encounter along
/// the way.
///
/// \see \c Unwind(), \c ValidateDirectBr().
///
/// \param ctx Unwinding context, as described above.
/// \param InBlock The block in which we’re moving forward/backward. Must not be empty!
/// \param From The statement in \p InBlock containing the jump.
/// \param To The statement we’re unwinding to.
/// \return Always \c true if we’re moving backwards, \c false if we’re
///         moving forward and encounter a protected expression.
auto src::Sema::UnwindLocal(
    UnwindContext ctx,
    BlockExpr* InBlock,
    Expr* From,
    Expr* To
) -> bool {
    /// Find the positions of `From` and `To` in `InBlock` so we know which one
    /// comes first. Both must be direct children of the block.
    auto FromIter = rgs::find(InBlock->exprs, From);
    auto ToIter = rgs::find(InBlock->exprs, To);
    Assert(FromIter != InBlock->exprs.end());
    Assert(ToIter != InBlock->exprs.end());

    /// We want to include `From` only if we’re actually unwinding (i.e. *emitting*
    /// protected subexpressions during backwards movement)—in which case the `ctx`
    /// will not be an expression.
    ///
    /// Note that `FromIter` will be used as the *end* iterator below, so *advancing*
    /// it means we *include* `From` in the loop in that case.
    if (not ctx.is<Expr*>()) ++FromIter;

    /// Check all expressions between `To` (inclusive) and `From` (possibly inclusive).
    ///
    /// Note that LabelExprs are syntactically Statements that may contain other Statements
    /// (this is because e.g. `foo: bar: int x` is valid), so make sure to look through any
    /// labels here. Protected expressions must be statements, so it suffices to check just
    /// the statements themselves (protected expressions in nested blocks have no effect on
    /// anything outside those blocks, so those don’t matter here).
    for (auto E : rgs::subrange(ToIter, FromIter) | vws::reverse) {
        /// We are moving forward; crossing a protected expression is an error.
        if (auto expr = ctx.dyn_cast<Expr*>()) {
            if (E->is_protected) {
                Error(expr, "Jump is ill-formed");
                Diag::Note(
                    mod->context,
                    E->ignore_labels->location,
                    "Because it would bypass {} here",
                    isa<DeferExpr>(E->ignore_labels)
                        ? "deferred expression"s
                        : "variable declaration"s
                );

                /// Bail out on the first protected expression; no need to report more than one.
                return false;
            }
        }

        /// We are moving backwards; collect any protected expressions. This is *not* an error.
        else {
            auto prot = ctx.get<SmallVectorImpl<Expr*>*>();
            if (E->is_protected) prot->push_back(E->ignore_labels);
        }
    }

    /// Everything was ok.
    return true;
}

/// \brief Handle upwards/downwards movement between two blocks.
///
/// Note that we implement downwards movement as upwards movement to
/// simplify the implementation of this.
///
/// Note that this does nothing if \p From = \p To when the function
/// is called. Use \c UnwindLocal() for that instead. This implements
/// upwards/downwards movement *only*. However, this *does* take care
/// of any forward/backward movement that is required to enter or leave
/// a scope starting at any Statement.
///
/// \see UnwindLocal(), ValidateDirectBr().
///
/// \param ctx Unwinding context passed to \c UnwindLocal().
/// \param From The block that we are unwinding from, i.e. the block that
///        contains \p E.
/// \param E The statement in \p From that we are unwinding from.
/// \param To The block that we are unwinding to; this must be an ancestor
///        of \p From (or equal to it).
/// \return The parent Statement in \p To that contains \p E, or \c nullptr
///         if there was an error.
auto src::Sema::Unwind(
    UnwindContext ctx,
    BlockExpr* From,
    Expr* E,
    BlockExpr* const To
) -> Expr* {
    /// We haven’t reached the block we want to unwind to just yet, i.e.
    /// `From` is still a *child* of `To`.
    while (From != To) {
        /// Handle backwards/forwards movement.
        ///
        /// Conceptually, this moves from somewhere in the middle in the block
        /// to the first Statement in the block (or vice versa, if we’re doing
        /// ‘downwards’ movement).
        if (
            not From->exprs.empty() and
            not UnwindLocal(ctx, From, E, From->exprs.front())
        ) return nullptr;

        /// Note that `E` is a Statement in `From`; this sets it to the parent
        /// Statement of `From`, i.e. the Statement in `Parent(From)`, i.e. the
        /// Statement that contains `From` and also `E` (and, by transitivity,
        /// our original `E` on function entry).
        E = From->parent_full_expression;

        /// Move up a scope. This is the upwards/downwards movement proper, and
        /// as we’ve already discussed at length, it is a no-op.
        From = From->parent;
    }

    /// `E` is now the parent of `From` (and, by transitivity, also `E`) in `To`.
    return E;
}

/// \brief Unwind one or more scopes.
///
/// This implements any control-flow transfer that isn’t a Goto, i.e. \c break,
/// \c continue, and \c return. Thus, this is always upwards movement and thus
/// should never error.
///
/// Note that \c To is included in the unwinding process; for example, if \c UW
/// is a \c return expression, then \c To will be the top-level scope of the
/// function containing the \c return.
///
/// \param From The scope that contains \p UW, i.e. which we are unwinding from.
/// \param To The scope we are unwinding to (inclusive).
/// \param UW The Statement from which we’re unwinding.
void src::Sema::UnwindUpTo(
    BlockExpr* From,
    BlockExpr* const To,
    UnwindExpr* const UW
) {
    /// `UW` may be a subexpression of some other expression; get the Statement
    /// that contains it, here `E`.
    auto E = UW->parent_full_expression;

    /// Always unwind at least the scope that contains `UW`.
    for (;;) {
        /// This kind of stack unwinding must never fail.
        if (not From->exprs.empty()) {
            Assert(
                UnwindLocal(&UW->unwind, From, E, From->exprs.front()),
                "Unwinding non-direct branch should always succeed"
            );
        }

        /// If we’ve just handled everything in `To`, stop.
        if (From == To) break;

        /// Same procedure as in UnwindLocal(): find the Statement that contains `E`
        /// (and thus `UW`) in the parent of `From`, and set `From` to its parent.
        E = From->parent_full_expression;
        From = From->parent;
    }
}

/// \brief This is called for every GotoExpr in the module.
///
/// Check whether a direct branch (= GotoExpr) to a label is sound,
/// and collect any protected expressions that we need to unwind.
///
/// \see Unwind(), UnwindLocal().
void src::Sema::ValidateDirectBr(GotoExpr* g, BlockExpr* source) {
    if (not g->sema.ok) return;

    /// `g`:      the goto from which we’re transferring control.
    /// `label`:  the LabelExpr to which we’re transferring control.
    /// `source`: the block containing `g`.
    /// `target`: the block containing `label`.
    const auto label = g->target;
    const auto target = label->parent;

    /// Sanity checks.
    /// FIXME: Check for this in the parser instead.
    /// FIXME: Either prevent declarations/defer in subexpressions or
    ///        actually handle that case properly.
    if (label != label->parent_full_expression) {
        Error(g, "Target label of jump may not be a subexpression");
        return;
    }

    if (g != g->parent_full_expression) {
        Error(g, "'goto' must not be a subexpression");
        return;
    }

    /// Find the NCA of the source and target blocks.
    BlockExpr* const nca = BlockExpr::NCAInFunction(source, target);
    Assert(nca, "Goto and label blocks must have a common ancestor");

    /// If NCA != target, then the target is a child of the NCA,
    /// i.e. we have downwards movement; handle that part of the
    /// jump first. Unwind() takes care of that. Note that we
    /// implement this downwards movement from the NCA to `target`
    /// via upwards movement *from* `target` up to NCA. This way,
    /// we don’t actually need to implement moving down the scope
    /// stack.
    ///
    /// `label_nca` is the closest parent of `label` in NCA, i.e.
    /// `label` itself if `label` is a direct child of NCA, or the
    /// Statement in NCA that contains `label`.
    Expr* label_nca = label;
    if (nca != target) {
        label_nca = Unwind(g, target, label, nca);

        /// Since this ‘upwards movement’ is actually downwards
        /// movement, we bail out early if we end up crossing any
        /// protected expressions.
        if (not label_nca) return;
    }

    /// Handle upwards movement from the source to the NCA. This
    /// is genuine upwards movement, so no errors on protected
    /// expressions here, but store them in `g->unwind` so we can
    /// unwind them later.
    ///
    /// `goto_nca` is to `goto` what `label_nca` is to `label`.
    Expr* const goto_nca = Unwind(&g->unwind, source, g, nca);

    /// Convert `goto_nca` and `label_nca` to iterators in NCA
    /// so we can figure out which one comes first.
    auto goto_it = rgs::find(nca->exprs, goto_nca);
    auto label_it = rgs::find(nca->exprs, label_nca);
    Assert(goto_it != nca->exprs.end());
    Assert(label_it != nca->exprs.end());

    /// `unwind_to` and `unwind_from` are the Statements in NCA
    /// that we’re unwinding to (i.e. the destination) and from
    /// (i.e. the source), respectively.
    auto unwind_to = std::min(goto_it, label_it);
    auto unwind_from = std::max(goto_it, label_it);

    /// Finally, handle forwards or backwards movement in the NCA,
    /// if any. If we’re moving *forwards*, i.e. if `goto_it` <
    /// `label_it`, tell UnwindLocal() to error on any protected
    /// expressions; otherwise, we’re moving backwards, so simply
    /// add them to `g->unwind` so we can unwind them later.
    UnwindLocal(
        goto_it < label_it ? UnwindContext(g) : &g->unwind,
        nca,
        *unwind_from,
        *unwind_to
    );
}

/// ===========================================================================
///  Construction and Initialisation
/// ===========================================================================
/// Handle object construction.
///
/// This handles everything related to variable initialisation and construction
/// of objects. Note that there is a big difference between certain builtin types
/// and all other types. Some builtin types are trivial, that is, they have no
/// constructor. All that needs to be done to create a value of such a type is
/// simply to... emit the value. This is the case for numbers, empty slices, ranges
/// and empty optionals.
///
/// Some builtin types, such as references, always require an initialiser; moreover,
/// the aforementioned builtin types *can* take a single initialiser. However, all
/// of these cases still only involve simply producing a single value. Even scoped
/// pointers, which require a heap allocation, are fully self-contained in terms
/// of their initialisation.
///
/// Where it gets more complicated is once arrays and structs are introduced. Handling
/// arrays in a similar fashion to all other types, although possible for arrays of
/// trivial types or with a single initialiser value that is broadcast across the
/// entire array, becomes infeasible for arrays initialised by an array literal or
/// for arrays of structure type: not only would it be quite the pain to associate
/// each element of a (nested) array literal with the corresponding location in the
/// in-memory array, but creating structures may require constructor calls, and those
/// take a `this` pointer to an in-memory location, but array literals are rvalues and
/// thus not in memory.
///
/// This means that, for arrays and structures initialised by array and structure
/// literals, respectively, we borrow a page out of C++’s book: instead of emitting
/// an array literal and then storing it into the variable, we instead evaluate the
/// literal with the variable as its ‘result object’, constructing and storing values
/// directly into the memory provided for the variable as we go. Thus, in those cases,
/// the actual construction is handled by the code that emits the literal.
///
/// Side note: this also provides a way for us to emit literals that are not assigned
/// to anything since the constructors invoked by those literals still need a memory
/// location to construct into. For those literals, the backend simply hallucinates
/// a memory location out of thin air for them to construct into; this is similar to
/// the concept of temporary materialisation in C++.
auto src::Sema::Construct(
    Location loc,
    Type type,
    MutableArrayRef<Expr*> init_args,
    Expr* target,
    bool raw
) -> ConstructExpr* {
    /// If the type being constructed is an enum type, then its
    /// enumerators are in scope here.
    auto ty = type.desugared;
    std::optional<DeclContext::Guard> guard = std::nullopt;
    if (auto e = dyn_cast<EnumType>(ty)) guard.emplace(this, e);

    /// Validate initialisers.
    if (not rgs::all_of(init_args, [&](auto*& e) { return Analyse(e); }))
        return nullptr;

    /// If the only initialiser is already a construct expression, just keep it.
    if (
        init_args.size() == 1 and
        isa<ConstructExpr>(init_args[0]) and
        init_args[0]->type == type
    ) return cast<ConstructExpr>(init_args[0]);

    /// Helper to emit an error that construction is not possible.
    auto InvalidArgs = [&] -> ConstructExpr* {
        utils::Colours C{ctx->use_colours};
        auto FormatArg = [&](auto* e) { return e->type.str(ctx->use_colours, true); };
        Error(
            loc,
            "Cannot construct '{}' from arguments {}",
            type,
            fmt::join(
                vws::transform(init_args, FormatArg),
                fmt::format("{}, ", C(utils::Colour::Red))
            )
        );
        return nullptr;
    };

    /// Helper to emit a trivial copy.
    auto CopyInit = [&](Type ty) -> ConstructExpr* {
        if (not Convert(init_args[0], ty)) {
            Error(
                init_args[0],
                "Cannot convert '{}' to '{}'",
                init_args[0]->type,
                ty
            );
            return nullptr;
        }

        return ConstructExpr::CreateCopy(mod, init_args[0]);
    };

    /// Try and resolve a constructor call.
    auto LookUpConstructor = [&](StructType* s) -> ProcDecl* {
        auto res = PerformOverloadResolution(loc, s->initialisers, init_args);
        if (res.ill_formed()) return nullptr;

        /// If there is no viable constructor, but the arguments consist of
        /// a single TupleExpr, try to unpack it and see if that works.
        if (res.failure() and init_args.size() == 1 and isa<TupleExpr>(init_args[0])) {
            auto tuple = cast<TupleExpr>(init_args[0]);
            auto res2 = PerformOverloadResolution(loc, s->initialisers, tuple->elements);

            /// FIXME: This will print all overloads twice. The question here is: how
            /// much information does it make sense to provide here? We don’t know whether
            /// the the user intended the tuple to be unpacked or not, after all. Perhaps
            /// we should print only the results of the second attempt iff there are no
            /// overloads that take a tuple, and only the results of the first attempt
            /// otherwise. Is that a good-enough heuristic? Either way, should we also
            /// provide a command-line flag to always print all overloads? This could
            /// potentially also be incorporated into something like an '-fverbose-diagnostics'
            /// flag.
            if (not res2) return nullptr;

            /// Unpacking worked. Suppress diagnostics from the first attempt.
            init_args = tuple->elements;
            std::get<OverloadResolutionResult::ResolutionFailure>(res.result).suppress();
            res.result = std::move(res2.result);
        }

        if (res.failure()) return nullptr;
        return std::get<OverloadResolutionResult::Ok>(res.result);
    };

    /// Initialisation is dependent on the type.
    switch (ty->kind) {
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
        case Expr::Kind::TypeofType:
            Unreachable("Should have been desugared");

        case Expr::Kind::OpaqueType:
            Unreachable("Don’t know how to construct a '{}'", ty.str(ctx->use_colours, true));

        /// These take zero or one argument.
        case Expr::Kind::BuiltinType:
        case Expr::Kind::IntType: {
            if (init_args.empty()) return ConstructExpr::CreateZeroinit(mod);
            if (init_args.size() == 1) return CopyInit(type);
            return InvalidArgs();
        }

        /// References must always be initialised.
        case Expr::Kind::ReferenceType: {
            if (init_args.empty()) {
                Error(loc, "Reference must be initialised");
                return nullptr;
            }

            if (init_args.size() == 1) return CopyInit(type);
            return InvalidArgs();
        }

        case Expr::Kind::StructType: {
            auto s = cast<StructType>(ty);

            /// If the type has constructors, try to find one that works.
            if (not raw and not s->initialisers.empty()) {
                auto ctor = LookUpConstructor(s);
                if (not ctor) return nullptr;
                return ConstructExpr::CreateInitialiserCall(
                    mod,
                    ctor,
                    init_args
                );
            }

            /// Otherwise, this basically acts just like a tuple.
            [[fallthrough]];
        }

        case Expr::Kind::TupleType: {
            auto t = cast<RecordType>(ty);

            /// If there are no arguments, perform default-construction.
            if (init_args.empty()) {
                /// If all elements of the tuple are trivial, this is zero-initialisation.
                if (Type(t).trivially_constructible) return ConstructExpr::CreateZeroinit(mod);

                /// Otherwise, all types must be default-constructible.
                Assert(not t->fields.empty(), "Empty record type should be trivially-constructible");
                SmallVector<Expr*, 16> ctors;
                for (auto elem : t->fields) {
                    auto ctor = Construct(loc, elem->type, {});
                    if (not ctor) return nullptr;
                    ctors.push_back(ctor);
                }

                return ConstructExpr::CreateRecordListInit(mod, ctors, t);
            }

            /// If there is one argument that is also a tuple with the
            /// exact same type as this, just use it.
            if (init_args[0]->type == ty) return CopyInit(ty);

            /// If there is one argument that is a tuple, take its elements.
            SmallVector<Expr*, 16> args_storage;
            if (
                auto tuple = dyn_cast<TupleType>(init_args[0]->type.desugared);
                init_args.size() == 1 and tuple
            ) {
                /// If this is a literal, don’t bother materialising a temporary
                /// and instead just extract the elements directly.
                auto lit = dyn_cast<TupleExpr>(init_args[0]);
                if (not lit) MaterialiseTemporaryIfRValue(init_args[0]);
                for (auto [i, f] : tuple->fields | vws::enumerate) {
                    auto idx = lit ? lit->elements[usz(i)] : new (mod) TupleIndexExpr(init_args[0], f, loc); // FIXME: fidelity.
                    args_storage.push_back(idx);
                }

                /// The unwrapped tuple is now the argument list.
                init_args = args_storage;
            }

            /// If there are more arguments than elements, this is an error.
            if (init_args.size() > t->fields.size()) {
                auto ToStr = [&](Expr* e) { return e->type.str(ctx->use_colours, true); };
                Error(
                    loc,
                    "Too many arguments in construction of '{}' from [{}]",
                    type,
                    fmt::join(init_args | vws::transform(ToStr), ", ")
                );
            }

            /// Same loop as earlier, but this time with one argument each
            /// time. If there are fewer arguments than elements, default
            /// construct the rest.
            SmallVector<Expr*, 16> ctors;
            for (auto [i, elem] : t->fields | vws::enumerate) {
                auto args = usz(i) < init_args.size() ? init_args[usz(i)] : MutableArrayRef<Expr*>{};
                auto ctor = Construct(loc, elem->type, args);
                if (not ctor) return nullptr;
                ctors.push_back(ctor);
            }

            return ConstructExpr::CreateRecordListInit(mod, ctors, t);
        }

        /// This is the complicated one.
        case Expr::Kind::ArrayType: {
            auto [base, total_size, _] = ty.strip_arrays;
            auto no_args = init_args.empty();
            auto tuple = no_args ? nullptr : dyn_cast<TupleExpr>(init_args[0]->ignore_parens);

            /// No args or a single empty tuple.
            if (no_args or (tuple and tuple->elements.empty())) {
                /// An array of trivial types is itself trivial.
                if (base.trivially_constructible) return ConstructExpr::CreateZeroinit(mod);

                /// Raw construction.

                /// Otherwise, if this is a struct type with a constructor
                /// that takes no elements, call that constructor for each
                /// element.
                if (auto ctor = base.default_constructor)
                    return ConstructExpr::CreateArrayInitialiserCall(mod, {}, ctor, total_size);

                /// Otherwise, this type cannot be constructed from nothing.
                Error(loc, "Type '{}' is not default-constructible", type);
                return nullptr;
            }

            if (init_args.size() != 1) Diag::ICE(
                mod->context,
                loc,
                "Array initialiser must have exactly one argument. Did you mean to use an array literal?",
                init_args.size()
            );

            /// If the argument is an array literal, construct the array from it.
            if (tuple) {
                auto array_type = cast<ArrayType>(ty);
                auto size = array_type->dimension().getZExtValue();
                auto elem = array_type->elem.desugared;

                /// Array literal must not be larger than the array type.
                if (size < tuple->elements.size()) {
                    Error(
                        loc,
                        "Cannot initialise array '{}' from tuple containing '{}' elements",
                        type,
                        size
                    );
                    return nullptr;
                }

                /// Construct each element.
                SmallVector<Expr*, 16> ctors;
                for (auto e : tuple->elements)
                    if (auto ctor = Construct(e->location, elem, e))
                        ctors.push_back(ctor);

                /// If the array literal is too small, that’s fine, so long
                /// as the element type is default-constructible.
                if (auto rem = size - tuple->elements.size()) {
                    rem *= elem.strip_arrays.total_dimension;
                    if (base.trivially_constructible) {
                        ctors.push_back(ConstructExpr::CreateArrayZeroinit(mod, rem));
                    } else if (auto ctor = base.default_constructor) {
                        ctors.push_back(ConstructExpr::CreateArrayInitialiserCall(mod, {}, ctor, rem));
                    } else {
                        Error(
                            loc,
                            "Cannot create array '{}' from tuple containing '{}' "
                            "elements as '{}' is not default-constructible",
                            type,
                            tuple->elements.size(),
                            base
                        );
                        return nullptr;
                    }
                }

                /// Create a constructor that calls all of these constructors.
                return ConstructExpr::CreateArrayListInit(mod, ctors);
            }

            /// If the argument is an array, simply copy/move it.
            if (init_args[0]->type.strip_refs_and_pointers == ty) return CopyInit(ty);

            /// Otherwise, attempt to broadcast a single value. If the
            /// value is simply convertible to the target type, then
            /// just copy it.
            if (base.trivially_constructible) {
                ConversionSequence seq;
                if (not TryConvert(seq, init_args[0], base)) {
                    Error(
                        loc,
                        "Cannot initialise an object of type '{}' with a value of type '{}'",
                        base,
                        init_args[0]->type
                    );
                    return nullptr;
                }

                ApplyConversionSequence(init_args[0], std::move(seq));
                return ConstructExpr::CreateArrayBroadcast(mod, init_args[0], total_size);
            }

            /// If not, see if there is a constructor with that type.
            if (auto s = dyn_cast<StructType>(base)) {
                auto ctor = LookUpConstructor(s);
                if (not ctor) return nullptr;
                return ConstructExpr::CreateArrayInitialiserCall(mod, init_args, ctor, total_size);
            }

            /// Otherwise, we can’t do anything here.
            return InvalidArgs();
        }

        case Expr::Kind::SliceType: {
            if (init_args.empty()) return ConstructExpr::CreateZeroinit(mod);
            if (init_args.size() == 1) return CopyInit(type);

            /// A slice can be constructed from a reference+size.
            auto s = cast<SliceType>(ty);
            if (init_args.size() == 2) {
                Type ref = new (mod) ReferenceType(s->elem, loc);
                AnalyseAsType(ref);

                /// First argument must be a reference.
                if (not Convert(init_args[0], ref)) {
                    Error(
                        loc,
                        "Cannot convert '{}' to '{}'",
                        init_args[0]->type,
                        ref
                    );
                    return nullptr;
                }

                /// Second argument must be an integer.
                if (not Convert(init_args[1], Type::Int)) {
                    Error(
                        loc,
                        "Cannot convert '{}' to '{}'",
                        init_args[1]->type,
                        Type::Int
                    );
                    return nullptr;
                }

                return ConstructExpr::CreateSliceFromParts(mod, init_args[0], init_args[1]);
            }

            return InvalidArgs();
        }

        case Expr::Kind::OptionalType: {
            auto opt = cast<OptionalType>(ty);
            if (not isa<ReferenceType>(opt->elem)) Todo("Non-reference optionals");

            /// Nil.
            if (init_args.empty()) return ConstructExpr::CreateZeroinit(mod);

            /// Single argument.
            if (init_args.size() == 1) {
                /// If the initialiser is already an optional,
                /// just leave it as is.
                if (init_args[0]->type == Type::Nil) return ConstructExpr::CreateZeroinit(mod);
                if (type == init_args[0]->type) return CopyInit(type);

                /// Otherwise, attempt to convert it to the wrapped type.
                LValueState.ActivateOptional(target);
                return CopyInit(opt->elem);
            }

            return InvalidArgs();
        }

        case Expr::Kind::EnumType: {
            auto enum_ty = cast<EnumType>(ty);

            /// Enums that have a zero value can be initialised with no arguments.
            if (init_args.empty()) {
                /// Zero is never valid for mask enums.
                if (enum_ty->mask) {
                    Error(loc, "Initialisation of bitmask enum '{}' requires a value", enum_ty);
                    return nullptr;
                }

                /// Check if there is a zero value in the enum or any of its parents.
                for (auto enum_type = enum_ty; enum_type; enum_type = enum_type->parent_enum)
                    for (auto n : enum_type->enumerators)
                        if (n->value.isZero())
                            return ConstructExpr::CreateZeroinit(mod);

                /// Otherwise, this can’t be default-initialised.
                Error(
                    loc,
                    "Initialisation of enum '{}' requires a value; 0 is not a valid "
                    "enum value for this type!",
                    enum_ty
                );
                return nullptr;
            }

            /// Enums can be initialised with a single argument, iff that
            /// argument is either of the same type as the enum or one of
            /// its parent types.
            if (init_args.size() == 1) {
                for (auto enum_type = enum_ty; enum_type; enum_type = enum_type->parent_enum)
                    if (init_args[0]->type == enum_type)
                        return CopyInit(enum_type);
            }

            return InvalidArgs();
        }

        case Expr::Kind::ProcType: Todo();
        case Expr::Kind::ScopedPointerType: Todo();
        case Expr::Kind::ClosureType:
            Todo();

            /// Ignore non types.
#define SOURCE_AST_EXPR(nontype) case Expr::Kind::nontype:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
            Unreachable("Invalid type for variable declaration");
    }
}

/// ===========================================================================
///  Analysis
/// ===========================================================================
bool src::Sema::Analyse(Expr*& e) {
    /// Don’t analyse the same expression twice.
    if (e->sema.analysed or e->sema.in_progress) return e->sema.ok;
    if (e->sema.errored) return false;
    e->sema.set_in_progress();

    /// Set this as analysed when we return from here.
    defer { e->sema.set_done(); };

    /// Save whether this is the direct child of a block.
    const bool direct_child_of_block = at_block_level;
    at_block_level = false;

    /// Analyse the expression.
    switch (e->kind) {
        /// Marked as type checked in the constructor or on creation.
        case Expr::Kind::AssignExpr:
        case Expr::Kind::BuiltinType:
        case Expr::Kind::ConstExpr:
        case Expr::Kind::ConstructExpr:
        case Expr::Kind::EmptyExpr:
        case Expr::Kind::ImplicitThisExpr:
        case Expr::Kind::IntType:
        case Expr::Kind::MaterialiseTemporaryExpr:
        case Expr::Kind::ModuleRefExpr:
        case Expr::Kind::OpaqueType:
        case Expr::Kind::ScopedType:
        case Expr::Kind::SugaredType:
        case Expr::Kind::TupleIndexExpr:
            Unreachable();

        /// Handled elsewhere.
        case Expr::Kind::FieldDecl:
            Unreachable("FieldDecl should be handled when analysing RecordTypes");
        case Expr::Kind::EnumeratorDecl:
            Unreachable("EnumeratorDecl should be handled when analysing EnumTypes");

        /// Perform overload resolution against an initialiser.
        case Expr::Kind::InvokeInitialiserExpr: {
            if (curr_proc->smp_kind != SpecialMemberKind::Constructor) return Error(
                e,
                "Initialiser invocation is only allowed in initialisers"
            );

            auto i = cast<InvokeInitialiserExpr>(e);
            auto parent = curr_proc->smp_parent;
            auto os = PerformOverloadResolution(e->location, parent->initialisers, i->args);
            if (not os) return false;
            i->initialiser = std::get<OverloadResolutionResult::Ok>(os.result);
            i->stored_type = i->initialiser->ret_type;
        } break;

        case Expr::Kind::RawLitExpr: {
            auto r = cast<RawLitExpr>(e);

            /// If no type was specified, we must be in an initialiser.
            if (r->type == Type::Unknown) {
                if (curr_proc->smp_kind != SpecialMemberKind::Constructor) return Error(
                    e,
                    "Raw initialiser invocation outside initialiser requires explicit type."
                );

                r->stored_type = curr_proc->smp_parent;
            }

            auto ctor = Construct(r->location, r->type, r->args, nullptr, true);
            if (not ctor) return e->sema.set_errored();
            r->ctor = ctor;
        } break;

        /// Each overload of an overload set must be unique.
        /// TODO: Check that that is actually the case.
        case Expr::Kind::OverloadSetExpr: {
            auto os = cast<OverloadSetExpr>(e);
            for (auto p : os->overloads)
                if (not AnalyseProcedureType(p))
                    e->sema.set_errored();
        } break;

        /// An alias provides a different name for a compile-time entity.
        case Expr::Kind::AliasExpr: {
            auto a = cast<AliasExpr>(e);
            if (not Analyse(a->expr)) return e->sema.set_errored();
            if (not isa< // clang-format off
                LocalDecl,
                ProcDecl,
                OverloadSetExpr,
                TypeBase
            >(a->expr->ignore_paren_refs)) return Error(
                e,
                "Alias must reference a declaration, procedure, or type"
            ); // clang-format on

            /// Add this alias to the current scope.
            curr_scope->declare(a->alias, a);
        } break;

        /// Bools are of type bool.
        case Expr::Kind::BoolLitExpr:
            cast<BoolLitExpr>(e)->stored_type = BuiltinType::Bool(mod, e->location);
            break;

        /// Integers are of type int.
        case Expr::Kind::IntLitExpr: {
            auto i = cast<IntLitExpr>(e);
            if (i->value.getBitWidth() > 64) Diag::ICE(
                mod->context,
                e->location,
                "Sorry, integer literals with a bit width of {} are currently not supported",
                i->value.getBitWidth()
            );

            i->stored_type = BuiltinType::Int(mod, e->location);
            i->value = i->value.zext(64);
        } break;

        /// Parens just forward whatever is inside.
        case Expr::Kind::ParenExpr: {
            auto p = cast<ParenExpr>(e);
            if (not Analyse(p->expr)) return e->sema.set_errored();
            p->stored_type = p->expr->type;
            p->is_lvalue = p->expr->is_lvalue;
        } break;

        /// String literals are u8 slices.
        case Expr::Kind::StrLitExpr: {
            auto str = cast<StrLitExpr>(e);
            auto loc = str->location;

            /// Unlike in C++, string literals are *not* lvalues; rather a
            /// new string slice is constructed every time a string literal
            /// is used.
            str->stored_type = new (mod) SliceType(Type::I8, loc);
            AnalyseAsType(str->stored_type);
        } break;

        /// Check each element.
        ///
        /// By default, a tuple literal yields an rvalue of tuple type. Contexts
        /// that want e.g. an array literal instead have to process this manually
        /// instead.
        case Expr::Kind::TupleExpr: {
            auto tuple = cast<TupleExpr>(e);
            for (auto& elem : tuple->elements)
                if (not Analyse(elem))
                    return e->sema.set_errored();

            tuple->stored_type = new (mod) TupleType(
                mod,
                tuple->elements | vws::transform([](Expr* e) { return e->type; }),
                e->location
            );

            Assert(AnalyseAsType(tuple->stored_type));
        } break;

        /// Handled out of line because it’s too complicated.
        case Expr::Kind::ProcDecl:
            AnalyseProcedure(cast<ProcDecl>(e));
            break;

        /// Array size must be a valid constant expression.
        /// TODO: Type inference and checking that we don’t have e.g. `void[]`.
        case Expr::Kind::ArrayType: {
            auto arr = cast<ArrayType>(e);
            if (not Analyse(arr->dim_expr)) return e->sema.set_errored();

            /// Determine size.
            if (not EvaluateAsIntegerInPlace(arr->dim_expr)) return e->sema.set_errored();

            /// Element type must be legal in a declaration.
            if (not AnalyseAsType(arr->elem) or not MakeDeclType<true>(arr->elem))
                return e->sema.set_errored();
        } break;

        /// No restrictions here, but check the element types.
        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::OptionalType: /// TODO: `bool?` should be ill-formed or require special syntax.
        case Expr::Kind::SliceType:
        case Expr::Kind::ClosureType:
            Analyse(cast<SingleElementTypeBase>(e)->elem.ptr);
            break;

        /// Parameters and return type must be complete.
        case Expr::Kind::ProcType: {
            auto type = cast<ProcType>(e);
            for (auto& param : type->parameters)
                if (not ClassifyParameter(&param))
                    e->sema.set_errored();
            if (not AnalyseAsType(type->ret_type)) e->sema.set_errored();
        } break;

        /// Structures.
        case Expr::Kind::StructType:
        case Expr::Kind::TupleType:
            AnalyseRecord(cast<RecordType>(e));
            break;

        /// Get the type of an expression.
        case Expr::Kind::TypeofType: {
            auto t = cast<TypeofType>(e);
            tempset unevaluated = true;
            if (not Analyse(t->expr)) return e->sema.set_errored();
        } break;

        /// Enumerations.
        case Expr::Kind::EnumType: {
            auto n = cast<EnumType>(e);

            /// Check the underlying type and make sure it is an integer
            /// type or another enum type.
            if (not AnalyseAsType(n->elem)) return e->sema.set_errored();
            if (not isa<EnumType>(n->elem.desugared) and not n->elem.desugared.is_int(false)) return Error(
                e,
                "Underlying type of enum must be an integer type or another enum type"
            );

            /// Analyse our parent, if we have one.
            if (n->parent_enum) {
                Type t{n->parent_enum};
                if (not AnalyseAsType(t)) return e->sema.set_errored();
                Assert(t.ptr == n->parent_enum, "Must not wrap parent enum");
            }

            /// Mask enums can only extend mask enums (or integer types, of
            /// course), but vice versa is fine.
            if (n->mask and n->parent_enum and not n->parent_enum->mask) return Error(
                e,
                "Bitmask enum cannot extend non-bitmask enum '{}'",
                Type{n->parent_enum}
            );

            /// Enter the enum’s scope and push a new DeclContext for name lookup.
            DeclContext::Guard _{this, n};
            tempset curr_scope = n->scope;
            open_enums.push_back(n);
            defer { open_enums.pop_back(); };

            /// For regular enums, all enumerators must either have constant
            /// initialisers or they will be assigned the value of the previous
            /// enumerator plus one. In the initialiser of an enum, all previous
            /// enumerators as well as inherited enumerators are in scope, but
            /// unlike outside the enum, their type in the enum is the underlying
            /// integer type.
            auto underlying = n->underlying_type;
            auto bits = u32(underlying.size(ctx).bits());
            const APInt one = {bits, 1};
            const APInt* last_value = nullptr;

            /// If any of our parents have enumerators, pick the value of the last one.
            for (auto it = n->parent_enum; it; it = it->parent_enum) {
                if (not it->enumerators.empty()) {
                    last_value = &it->enumerators.back()->value;
                    break;
                }
            }

            /// Assign enumerator values.
            for (auto [i, enumerator] : vws::enumerate(n->enumerators)) {
                /// Check if this enum, or any of its parents, already has
                /// an enumerator with this name.
                for (auto it = n; it; it = it->parent_enum) {
                    if (auto [prev, escapes] = it->scope->find(enumerator->name, true)) {
                        Assert(not escapes, "Should never be set for enum lookups");
                        Assert(prev->size() == 1, "Enum scope should contain one declaration per name");
                        Error(
                            enumerator,
                            "An enumerator named '{}' already exists in this enum",
                            enumerator->name
                        );

                        Diag::Note(
                            ctx,
                            prev->front()->location,
                            "Previous declaration is here"
                        );

                        goto next_enumerator;
                    }
                }

                /// Add 1 to the previous value, or multiply by 2 if this is a mask enum.
                if (not enumerator->initialiser) {
                    APInt this_val;

                    /// Bitmask enums start at 1, other enums at 0.
                    if (not last_value) { this_val = {bits, n->mask ? 1 : 0}; }

                    /// Otherwise, add 1 to the previous value.
                    else {
                        bool ov{};
                        this_val = n->mask ? last_value->ushl_ov(one, ov) : last_value->uadd_ov(one, ov);
                        if (ov) {
                            auto err_val = last_value->sext(bits + 1);
                            if (n->mask) err_val <<= 1;
                            else ++err_val;
                            Error(
                                enumerator,
                                "Computed enumerator value '{}' is not representable by type '{}'",
                                err_val,
                                underlying
                            );
                            continue;
                        }
                    }

                    /// This value is now the last value.
                    auto c = new (mod) ConstExpr(
                        nullptr,
                        EvalResult(std::move(this_val), underlying),
                        enumerator->location
                    );

                    enumerator->initialiser = c;
                    last_value = &c->value.as_int();
                }

                /// Validate the initialiser.
                else {
                    if (not Analyse(enumerator->initialiser)) continue;

                    /// Make sure the value is an integer constant of the right type.
                    if (not EvaluateAsIntegerInPlace(enumerator->initialiser, true, underlying)) continue;

                    /// For mask enums, the value must be a power of two.
                    auto* this_val = &enumerator->value;
                    if (n->mask and (this_val->isZero() or not this_val->isPowerOf2())) {
                        Error(
                            enumerator->initialiser->location,
                            "Initialiser '{}' of mask enum must be a non-zero power of two",
                            *this_val
                        );
                        enumerator->sema.set_errored();
                        continue;
                    }

                    /// This value is now the last value.
                    last_value = this_val;
                }

                /// Add the enumerator *value* to the scope; this way the type of the
                /// enumerator within the enum is the underlying type.
                n->scope->declare(enumerator->name, enumerator->initialiser);
            next_enumerator:
            }

            /// This enum is only ok if all of its enumerators are.
            if (rgs::any_of(n->enumerators, [](auto e) { return e->sema.errored; }))
                return e->sema.set_errored();
        } break;

        /// Defer expressions have nothing to typecheck really, so
        /// we just check the operand and leave it at that. Even
        /// nested `defer defer` expressions, albeit degenerate, are
        /// accepted.
        case Expr::Kind::DeferExpr: {
            tempset curr_defer = cast<DeferExpr>(e);
            Analyse(curr_defer->expr);
        } break;

        /// For labelled expressions, the labels’ uniqueness has already
        /// been checked at parse time, so just check the labelled expr
        /// here; note that a labelled expression always returns void so
        /// as to disallow branching into the middle of a full-expression.
        case Expr::Kind::LabelExpr: {
            auto l = cast<LabelExpr>(e);
            l->parent = curr_scope;
            needs_link_to_full_expr.push_back(e);
            Analyse(l->expr);
        } break;

        /// Loop control expressions.
        case Expr::Kind::LoopControlExpr: {
            auto l = cast<LoopControlExpr>(e);
            needs_link_to_full_expr.push_back(e);

            /// No label means branch to the parent.
            if (l->label.empty()) {
                /// No loop to break out of or continue.
                if (loop_stack.empty()) {
                    return Error(
                        l->location,
                        "'{}' is invalid outside of loops",
                        l->is_continue ? "continue" : "break"
                    );
                } else {
                    l->target = loop_stack.back();
                }
            }

            /// Make sure the label exists.
            else {
                auto target = curr_proc->labels.find(l->label);
                if (target == curr_proc->labels.end()) return Error(l->location, "Unknown label '{}'", l->label);

                /// Make sure the label labels a loop.
                auto loop = dyn_cast<Loop>(target->second->expr);
                if (not loop) return Error(l->location, "Label '{}' does not label a loop", l->label);

                /// Set the target to the label and make sure we’re
                /// actually inside that loop.
                l->target = loop;
                if (not utils::contains(loop_stack, loop)) return Error(
                    l->target,
                    "Cannot {} to label '{}' from outside loop",
                    l->is_continue ? "continue" : "break",
                    l->label
                );
            }

            /// Unwind to the target.
            unwind_entries.emplace_back(curr_scope, l, l->target->body);
        } break;

        /// Unconditional branch.
        case Expr::Kind::GotoExpr: {
            /// First, make sure the label exists.
            auto g = cast<GotoExpr>(e);
            auto l = curr_proc->labels.find(g->label);
            if (l == curr_proc->labels.end()) return Error(
                e,
                "Unknown label '{}'",
                g->label
            );

            /// Mark the label as used.
            g->target = l->second;
            l->second->used = true;

            /// We need to check this later.
            needs_link_to_full_expr.push_back(e);
            unwind_entries.emplace_back(curr_scope, g);
        } break;

        /// Return expressions.
        case Expr::Kind::ReturnExpr: {
            auto r = cast<ReturnExpr>(e);
            if (r->value and not Analyse(r->value)) return e->sema.set_errored();
            needs_link_to_full_expr.push_back(e);

            /// If we’re in a `= <expr>` procedure, and the return type
            /// is unspecified, infer the return type from this return
            /// expression.
            auto ret = curr_proc->ret_type;
            if (ret == Type::Unknown) {
                cast<ProcType>(curr_proc->stored_type)->ret_type =
                    not r->value ? Type::Void : r->value->type;
            }

            /// Check for noreturn.
            else if (ret == Type::NoReturn) {
                if (r->value and r->value->type != Type::NoReturn) return Error(
                    e,
                    "'noreturn' function may not return"
                );
            }

            /// Return expression returns a value. Check that the return value
            /// is convertible to the return type.
            else if (r->value) {
                if (not Convert(r->value, ret)) return Error(
                    e,
                    "Cannot return a value of type '{}' from a function with return type '{}'",
                    r->value->type,
                    Type::Void
                );
            }

            /// Return expression has no argument.
            else {
                if (ret != Type::Void) {
                    return Error(
                        e,
                        "Function declared with return type '{}' must return a value",
                        r->value->type
                    );
                }
            }

            /// Unwind the stack.
            unwind_entries.emplace_back(curr_scope, r, curr_proc->body);
        } break;

        /// Assertions take a bool and an optional message.
        case Expr::Kind::AssertExpr: {
            Assert(e->location.seekable(mod->context), "Assertion requires location information");
            auto a = cast<AssertExpr>(e);

            /// Handle static assertions.
            if (a->is_static) {
                if (not Analyse(a->cond) or (a->msg and not Analyse(a->msg)))
                    return e->sema.set_errored();

                /// Condition and message must be constant expressions.
                EvalResult msg;
                if (not EvaluateAsBoolInPlace(a->cond, true)) return e->sema.set_errored();
                if (a->msg) {
                    if (not Evaluate(a->msg, msg, true)) return e->sema.set_errored();
                    if (not msg.is_str()) return Error(
                        a->msg->location,
                        "Static assert message must be a string"
                    );
                }

                /// If the condition is false, emit an error.
                if (not cast<ConstExpr>(a->cond)->value.as_int().getBoolValue()) {
                    e->sema.set_errored();
                    return Error(
                        a->cond->location,
                        "Static assertion failed{}{}",
                        a->msg ? ": " : "",
                        a->msg ? msg.as_str() : ""
                    );
                }

                /// No need to do anything else here since this is never
                /// going to be emitted anyway.
                break;
            }

            /// Condition must be a bool.
            if (Analyse(a->cond) and EnsureCondition(a->cond)) {
                /// If this asserts that an optional is not nil, mark it as active.
                if (auto o = LValueState.MatchOptionalNilTest(a->cond))
                    LValueState.ActivateOptional(o);
            }

            /// Message must be an i8[].
            auto i8slice = new (mod) SliceType(Type::I8, {});
            i8slice->sema.set_done();
            if (a->msg and Analyse(a->msg) and not Convert(a->msg, i8slice)) Error(
                a->msg->location,
                "Message of 'assert' must be of type '{}', but was '{}'",
                i8slice,
                a->msg->type
            );

            /// Create string literals for the condition and file name.
            a->cond_str = new (mod) StrLitExpr(
                mod->save(a->cond->location.text(mod->context)),
                a->location
            );

            a->file_str = new (mod) StrLitExpr(
                mod->save(mod->context->file(a->location.file_id)->path().string()),
                a->location
            );

            Analyse(a->cond_str);
            Analyse(a->file_str);
        } break;

        /// The type of a block is the type of the last expression that is not
        /// a named procedure or type declaration.
        case Expr::Kind::BlockExpr: {
            auto b = cast<BlockExpr>(e);
            tempset curr_scope = b;

            /// Track what optionals were made active in this scope and reset
            /// their active state when we leave it; same for with expressions.
            LValueState::ScopeGuard _{*this};
            const auto with_stack_size = with_stack.size();
            defer { with_stack.resize(with_stack_size); };

            /// Skip ProcDecls for the purpose of determining the type of the block.
            isz last = std::ssize(b->exprs) - 1;
            while (last >= 0) {
                if (isa<ProcDecl, StructType>(b->exprs[usz(last)])) last--;
                else break;
            }

            /// Analyse the block.
            for (auto&& [i, expr] : vws::enumerate(b->exprs)) {
                tempset needs_link_to_full_expr = SmallVector<Expr*>{};
                at_block_level = true;

                if (not Analyse(expr)) {
                    if (i == last) e->sema.set_errored();
                    continue;
                }

                if (i == last) {
                    b->stored_type = expr->type;
                    b->is_lvalue = expr->is_lvalue;
                }

                /// Update links.
                for (auto subexpr : needs_link_to_full_expr) {
                    if (auto n = dyn_cast<BlockExpr>(subexpr)) n->parent_full_expression = expr;
                    else if (auto d = dyn_cast<LabelExpr>(subexpr)) d->parent_full_expression = expr;
                    else if (auto g = dyn_cast<UnwindExpr>(subexpr)) g->parent_full_expression = expr;
                    else Unreachable();
                }
            }

            /// This needs to know what full expression it is in.
            needs_link_to_full_expr.push_back(b);

            /// Unwind the stack.
            if (not b->exprs.empty()) UnwindLocal(&b->unwind, b, b->exprs.back(), b->exprs.front());

            /// If the type could not be determined, set it to void.
            if (b->stored_type == Type::Unknown) b->stored_type = Type::Void;
        } break;

        /// An invoke expression may be a procedure call, or a declaration.
        case Expr::Kind::InvokeExpr: return AnalyseInvoke(e, direct_child_of_block);

        /// Builtins are handled out of line.
        case Expr::Kind::InvokeBuiltinExpr: return AnalyseInvokeBuiltin(e);

        /// Cast expression.
        case Expr::Kind::CastExpr: {
            auto m = cast<CastExpr>(e);
            Analyse(m->operand);
            switch (m->cast_kind) {
                /// Only generated by sema. Converts an lvalue to an rvalue by moving
                /// or copying in the case of a trivially-copyable type.
                case CastKind::LValueToRValue:
                    LValueState.SetDefinitelyMoved(m->operand);
                    m->stored_type = m->operand->type;
                    break;

                /// Only generated by sema. Dereference a reference
                /// once, yielding an lvalue.
                case CastKind::ReferenceToLValue:
                    if (m->operand->is_lvalue) m->cast_kind = CastKind::LValueRefToLValue;
                    Assert(m->type == cast<SingleElementTypeBase>(m->operand->type)->elem);
                    m->is_lvalue = true;
                    break;

                /// Should only be generated by the case above.
                case CastKind::LValueRefToLValue: Unreachable();

                /// Only generated by sema. Convert an lvalue to a reference. The
                /// result is an *rvalue* of reference type.
                case CastKind::LValueToReference:
                    AnalyseAsType(m->stored_type);
                    Assert(isa<ReferenceType>(m->type));
                    Assert(cast<ReferenceType>(m->type)->elem == m->operand->type);
                    break;

                /// Only generated by sema. Convert an optional to a bool.
                case CastKind::OptionalNilTest:
                    m->stored_type = Type::Bool;
                    break;

                /// Only generated by sema. Access the value of an optional.
                case CastKind::OptionalUnwrap:
                    m->stored_type = cast<OptionalType>(m->operand->type)->elem;
                    m->is_lvalue = m->operand->is_lvalue;
                    break;

                /// Only generated by sema. Wrap a value with an optional. This
                /// is never an lvalue, and the type is already set.
                case CastKind::OptionalWrap:
                    m->stored_type = new (mod) OptionalType(m->operand->type, m->location);
                    AnalyseAsType(m->stored_type);
                    break;

                /// Only generated by sema. Convert an array lvalue to the element type.
                case CastKind::ArrayToElemRef:
                    m->stored_type = new (mod) ReferenceType(cast<ArrayType>(m->operand->type)->elem, m->location);
                    AnalyseAsType(m->stored_type);
                    break;

                /// A bit cast converts an rvalue to some other type by reinterpreting
                /// the bit pattern without performing any actual conversion.
                case CastKind::BitCast:
                    Assert(m->operand->type.size(ctx) == m->type.size(ctx), "BitCast requires same-size types");
                    Assert(not m->operand->is_lvalue, "BitCast can only operate on rvalues");
                    m->is_lvalue = false;
                    break;

                /// Only generated by sema. No-op here. Currently, there
                /// is no implicit cast that yields an lvalue.
                case CastKind::Implicit: break;

                /// Explicit cast.
                case CastKind::Soft:
                case CastKind::Hard:
                    AnalyseExplicitCast(e, m->cast_kind == CastKind::Hard);
                    break;
            }
        } break;

        /// Member access into a type.
        case Expr::Kind::MemberAccessExpr: {
            auto m = cast<MemberAccessExpr>(e);

            /// We may have to try several different objects.
            auto TryAccess = [&](Expr*& object) -> Result<void> {
                /// Analyse the accessed object.
                if (not Analyse(object)) return Diag();
                auto unwrapped = UnwrappedType(object);

                /// A slice type has a `data` and a `size` member.
                ///
                /// Neither of these are lvalues since slices are supposed
                /// to be pretty much immutable, and you should create a new
                /// one rather than changing the size or the data pointer.
                if (isa<SliceType>(unwrapped)) {
                    auto ty = dyn_cast<SingleElementTypeBase>(unwrapped);
                    if (m->member == "data") {
                        UnwrapInPlace(object);
                        m->stored_type = new (mod) ReferenceType(ty->elem, m->location);
                        Assert(AnalyseAsType(m->stored_type));
                        return {};
                    }

                    if (m->member == "size") {
                        UnwrapInPlace(object);
                        m->stored_type = BuiltinType::Int(mod, m->location);
                        return {};
                    }

                    return Diag::Error(
                        mod->context,
                        m->location,
                        "Type '{}' has no '{}' member",
                        unwrapped.str(mod->context->use_colours, true),
                        m->member
                    );
                }

                /// Arrays only have a `size`.
                if (auto arr = dyn_cast<ArrayType>(unwrapped)) {
                    /// Type is known at compile time, so the object is never emitted. Note:
                    /// we should emit a warning if evaluating the object has side effects,
                    /// as it will never be evaluated.
                    if (m->member == "size") {
                        e = new (mod) ConstExpr(
                            e,
                            EvalResult{arr->dimension(), Type::Int},
                            m->location
                        );

                        return {};
                    }

                    return Diag::Error(
                        mod->context,
                        m->location,
                        "Type '{}' has no '{}' member",
                        unwrapped.str(mod->context->use_colours, true),
                        m->member
                    );
                }

                /// Struct field accesses are lvalues if the struct is an lvalue.
                if (auto s = dyn_cast<StructType>(unwrapped)) {
                    auto MakeMemberProcRef = [&](ArrayRef<ProcDecl*> procs) {
                        if (procs.size() == 1) m->field = new (mod) DeclRefExpr(procs[0], m->location);
                        else m->field = new (mod) OverloadSetExpr(SmallVector<ProcDecl*>(procs), m->location);
                        Assert(Analyse(m->field));
                        m->stored_type = Type::MemberProc;
                        m->is_lvalue = false;
                    };

                    /// "init" calls a constructor.
                    if (m->member == "init") {
                        UnwrapInPlace(object, true);

                        /// If the struct has no initialisers, we can’t call one.
                        /// TODO: Allow this anyway with no arguments so long
                        ///       as the type is default-constructible. Use a
                        ///       DefaultInitExpr or sth like that for that
                        ///       since an empty overload set isn’t well-formed atm.
                        if (s->initialisers.empty()) return Diag::Error(
                            mod->context,
                            m->location,
                            "Type '{}' has no initialisers",
                            unwrapped.str(mod->context->use_colours, true)
                        );

                        MakeMemberProcRef(s->initialisers);
                        return {};
                    }

                    /// Search fields.
                    auto f = rgs::find(s->fields, m->member, [](auto* f) { return f->name; });
                    if (f != s->fields.end()) {
                        UnwrapInPlace(object, true);
                        m->field = *f;
                        m->stored_type = m->field->type;
                        m->is_lvalue = object->is_lvalue;
                        return {};
                    }

                    /// Search member procedures.
                    auto procs = rgs::find(s->member_procs, m->member, [](auto& e) { return e.first(); });
                    if (procs != s->member_procs.end()) {
                        MakeMemberProcRef(procs->getValue());
                        return {};
                    }

                    return Diag::Error(
                        mod->context,
                        m->location,
                        "Type '{}' has no '{}' member",
                        unwrapped.str(mod->context->use_colours, true),
                        m->member
                    );
                }

                return Diag::Error(
                    mod->context,
                    m->location,
                    "Cannot perform member access on type '{}'",
                    m->object->type.str(mod->context->use_colours, true)
                );
            };

            /// Object may be missing if this is a `.x` access.
            if (not m->object) {
                /// Iterate by value so we don’t overwrite the objects on the with stack.
                for (auto w : with_stack | vws::reverse) {
                    auto res = TryAccess(w);
                    if (not res.is_diag) {
                        m->object = w;
                        return true;
                    } else {
                        res.diag.suppress();
                    }
                }

                /// If we get here, we couldn’t find an object to access.
                return Error(m, "No object on with stack with member '{}'", m->member);
            }

            /// Otherwise, try the object itself.
            return not TryAccess(m->object).is_diag or m->sema.set_errored();
        }

        /// Scope access into something that has a scope.
        case Expr::Kind::ScopeAccessExpr: {
            auto sa = cast<ScopeAccessExpr>(e);
            if (not Analyse(sa->object)) return e->sema.set_errored();

            /// Module lookup.
            if (auto m = dyn_cast<ModuleRefExpr>(sa->object)) {
                auto exp = m->module->exports.find(sa->element);
                if (exp == m->module->exports.end()) return Error(
                    sa,
                    "Module '{}' has no export named '{}'",
                    m->module->name,
                    sa->element
                );

                /// Overloading is handled later.
                if (exp->second.size() > 1) return Error(
                    sa,
                    "Ambiguous reference to '{}' in module '{}'",
                    sa->element,
                    m->module->name
                );

                /// Only one element in vector, so pick that one.
                sa->resolved = exp->second[0];

                /// If this is a type, replace it w/ a scoped type.
                if (auto ty = dyn_cast<TypeBase>(sa->resolved)) {
                    e->sema.unset(); /// Other instances of this will have to be replaced w/ this again.
                    e = new (mod) ScopedType(sa->element, sa->object, ty, sa->location);
                }

                /// Otherwise, this stays as a ScopeAccessExpr.
                else {
                    sa->stored_type = sa->resolved->type;
                    sa->is_lvalue = sa->resolved->is_lvalue;
                }

                break;
            }

            /// Type.
            if (auto ty_base = dyn_cast<TypeBase>(sa->object)) {
                Type ty{ty_base};

                /// Enumeration.
                if (auto n = dyn_cast<EnumType>(ty.desugared)) {
                    auto entry = SearchEnumScope(n, sa->element);
                    if (not entry) return Error(
                        sa,
                        "Could not find '{}' in enum '{}'{}",
                        sa->element,
                        ty,
                        n->parent_enum ? " or any of its parents" : ""
                    );

                    sa->resolved = entry.expr;
                    sa->stored_type = entry.type;
                    sa->is_lvalue = false;
                    break;
                }
            }

            return Error(sa, "LHS of operator '::' must be a module or enum type", sa->object->type);
        }

        /// Perform name lookup in scope.
        case Expr::Kind::DeclRefExpr:
            AnalyseDeclRefExpr<false>(e);
            break;

        /// Determine the static chain offset from a variable reference to
        /// its declaration. Type and lvalueness is already set by the ctor.
        case Expr::Kind::LocalRefExpr: {
            auto var = cast<LocalRefExpr>(e);
            if (var->parent != var->decl->parent(mod)) {
                /// Synthesised variables (e.g. loop variables) should never be captured.
                if (not var->decl->is_legal_to_capture) {
                    if (IsInParameter(var->decl)) return Error(
                        var,
                        "Cannot capture 'in' parameter '{}'",
                        var->decl->name
                    );

                    return Error(
                        var,
                        "Cannot capture synthesised variable '{}'",
                        var->decl->name
                    );
                }

                var->decl->set_captured();
            }
        } break;

        /// Parameter declaration.
        case Expr::Kind::ParamDecl: {
            auto var = cast<ParamDecl>(e);
            var->set_parent(curr_proc);
            if (not ClassifyParameter(var->info)) return e->sema.set_errored();
            var->stored_type = var->info->type;
            var->ctor = ConstructExpr::CreateParam(mod);

            /// Add the variable to the current scope.
            if (not var->sema.errored) {
                if (var->info->with) with_stack.push_back(var);
                curr_scope->declare(var->name, var);

                /// 'in' parameters behave like rvalues.
                var->is_lvalue = var->info->intent != Intent::In;
            }
        } break;

        /// Variable declaration.
        case Expr::Kind::LocalDecl: {
            auto var = cast<LocalDecl>(e);
            if (not var->parent_or_null) var->set_parent(curr_proc);
            if (not AnalyseAsType(var->stored_type)) return e->sema.set_errored();

            /// Parameters should be ParamDecls instead.
            if (var->local_kind == LocalKind::Parameter) Unreachable();

            /// Synthesised variables just point somewhere else.
            else if (
                var->local_kind == LocalKind::Synthesised or
                var->local_kind == LocalKind::SynthesisedValue
            ) {
                if (not MakeDeclType(var->stored_type)) return e->sema.set_errored();
                var->ctor = ConstructExpr::CreateUninitialised(mod);
            }

            /// This is a regular variable.
            else if (not AnalyseVariableInitialisation(e, var->ctor, var->stored_type, var->init_args))
                return e->sema.set_errored();

            /// Add the variable to the current scope.
            if (not var->sema.errored) {
                curr_scope->declare(var->name, var);
                var->is_lvalue = var->local_kind != LocalKind::SynthesisedValue;
            }
        } break;

        /// Static variable declaration.
        case Expr::Kind::StaticDecl: {
            auto var = cast<StaticDecl>(e);
            mod->static_vars.push_back(var);
            if (not AnalyseAsType(var->stored_type)) return e->sema.set_errored();
            return AnalyseVariableInitialisation(e, var->ctor, var->stored_type, var->init_args);
        }

        /// If expressions.
        case Expr::Kind::IfExpr: {
            auto i = cast<IfExpr>(e);

            /// If the condition has an error, the type of the if expression
            /// itself can still be determined as it is independent of the
            /// condition.
            if (Analyse(i->cond)) EnsureCondition(i->cond);

            /// Type is void, unless one of the conditions below applies.
            i->stored_type = Type::Void;

            /// Static if.
            ///
            /// The condition must be a constant expression.
            bool static_condition{};
            if (i->is_static) {
                if (not EvaluateAsBoolInPlace(i->cond)) return e->sema.set_errored();
                static_condition = cast<ConstExpr>(i->cond)->value.as_int().getBoolValue();

                /// Analyse only one branch.
                Expr** taken{};
                if (static_condition) taken = &i->then;
                else if (i->else_) taken = &i->else_;

                /// The type of this is the type of the branch that was taken.
                if (taken) {
                    Analyse(*taken);
                    i->stored_type = (*taken)->type;
                    i->is_lvalue = (*taken)->is_lvalue;
                }

                /// Don’t do anything else.
                break;
            }

            /// Regular if. Analyse the branches.
            ///
            /// If the condition tests whether an optional is not nil, set
            /// the active state of the optional to true in the branch where
            /// it isn’t.
            if (auto o = LValueState.MatchOptionalNilTest(i->cond)) {
                {
                    LValueState::OptionalActivationGuard _{*this, o};
                    if (not Analyse(i->then)) return e->sema.set_errored();
                }

                if (i->else_ and not Analyse(i->else_)) return e->sema.set_errored();
            } // clang-format off

            /// If a condition tests whether an optional *is* nil, set the
            /// active state in the else branch. Furthermore, if the then branch
            /// is noreturn, mark the optional as active for the rest of the scope.
            else if (auto local =
                tcast<UnaryPrefixExpr>(i->cond)
                .test λ(u, u->op == Tk::Not)
                .cast<CastExpr> λ(c, c->operand)
                .test λ(c, c->is_opt_test)
                .cast<LocalRefExpr> λ(c, c->operand->ignore_lv2rv)
                .get()
            ) {
                {
                    LValueState::OptionalActivationGuard _{*this, local->decl};
                    if (i->else_ and not Analyse(i->else_)) return e->sema.set_errored();
                }

                if (not Analyse(i->then)) return e->sema.set_errored();
                if (i->then->type == Type::NoReturn) LValueState.ActivateOptional(local->decl);
            } // clang-format on

            /// Otherwise, there is nothing to infer.
            else if (not Analyse(i->then) or (i->else_ and not Analyse(i->else_)))
                return e->sema.set_errored();

            /// If there is an else clause, then the type of this expression is a bit
            /// more difficult to determine.
            if (i->else_) {
                /// In order to facilitate checking whether all paths in a function
                /// return a value, we must take care to not set the type to noreturn
                /// unless both branches are noreturn. This has to be a separate case
                /// since noreturn is convertible to any type.
                ///
                /// If only one branch is of type noreturn, that means that if this
                /// expression yields a value rather than returning, the value must
                /// be of the other branch’s type.
                if (i->then->type.is_noreturn or i->else_->type.is_noreturn) {
                    i->stored_type = not i->then->type.is_noreturn  ? i->then->type
                                   : not i->else_->type.is_noreturn ? i->else_->type
                                                                    : Type::NoReturn;
                }

                /// Otherwise, If the types of the then and else branch are convertible
                /// to one another, then the type of the if expression is that type.
                /// Furthermore, ensure that either both clauses are lvalues or neither
                /// is; in the former case, the entire expr is an lvalue.
                else if (Convert(i->else_, i->then->type, true) or Convert(i->then, i->else_->type, true)) {
                    i->stored_type = i->then->type;
                    if (i->then->is_lvalue and i->else_->is_lvalue) i->is_lvalue = true;
                    else {
                        /// Convert both to rvalues so we don’t end up w/ an lvalue in
                        /// one case and an rvalue in the other.
                        InsertLValueToRValueConversion(i->then);
                        InsertLValueToRValueConversion(i->else_);
                    }
                }
            }
        } break;

        /// While loops.
        case Expr::Kind::WhileExpr: {
            auto w = cast<WhileExpr>(e);
            if (Analyse(w->cond)) EnsureCondition(w->cond);

            /// There is nothing left to do other than analyse the body. The
            /// type of this is always void, so sema for the while expression
            /// itself can never fail.
            loop_stack.push_back(w);
            defer { loop_stack.pop_back(); };
            Expr* b = w->body;
            Analyse(b);
            Assert(b == w->body, "Body of while expression must be a block");
        } break;

        /// For in loops.
        case Expr::Kind::ForInExpr: {
            auto f = cast<ForInExpr>(e);

            /// Complain that we can’t iterate over this.
            auto NotIterable = [&] {
                return Error(
                    f->range,
                    "Type '{}' is not iterable",
                    f->range->type
                );
            };

            /// Convert a tuple to an array temporary.
            auto ConvertTupleToArray = [&](Type elem_ty, usz dimension) {
                Type arr_ty = new (mod) ArrayType(
                    elem_ty,
                    new (mod) ConstExpr(
                        nullptr,
                        EvalResult{APInt{64, dimension}, Type::Int},
                        {}
                    ),
                    f->range->location
                );

                Assert(AnalyseAsType(arr_ty));

                /// If this is a tuple literal whose type has not been set yet, adjust it in-place.
                if (auto t = dyn_cast<TupleExpr>(f->range); t and t->stored_type == Type::Unknown)
                    t->stored_type = arr_ty;

                /// Otherwise, we need to create a cast.
                else {
                    InsertLValueToRValueConversion(f->range);
                    f->range = new (mod) CastExpr(CastKind::BitCast, f->range, arr_ty, f->location);
                    Assert(Analyse(f->range));
                }

                if (not MaterialiseTemporary(f->range, arr_ty)) return e->sema.set_errored();
                return true;
            };

            /// If the range is a tuple literal, it is only iterable if all
            /// of its elements have the same type. We need to do some processing
            /// manually here because we don’t know what the type of this is
            /// supposed to be.
            if (auto tuple = dyn_cast<TupleExpr>(f->range)) {
                Type elem_ty = Type::Unknown;
                for (auto& elem : tuple->elements) {
                    if (not Analyse(elem)) return e->sema.set_errored();
                    if (elem_ty == Type::Unknown) {
                        elem_ty = elem->type;
                        if (not MakeDeclType(elem_ty)) return e->sema.set_errored();
                    } else if (not Convert(elem, elem_ty)) {
                        Error(
                            tuple->location,
                            "Cannot iterate over tuple literal with elements of "
                            "incompatible types '{}' and '{}'",
                            elem_ty,
                            elem->type
                        );
                        return e->sema.set_errored();
                    }
                }

                /// Treat the tuple as an array literal.
                tuple->sema.set_done();
                if (not ConvertTupleToArray(elem_ty, tuple->elements.size())) return false;
            }

            /// Check the range first, as we need to determine the type of the
            /// iteration variable from it.
            if (not Analyse(f->range)) return e->sema.set_errored();
            UnwrapInPlace(f->range, true);

            /// Slices can just be loaded whole.
            if (isa<SliceType>(f->range->type)) InsertLValueToRValueConversion(f->range);

            /// Arrays must be lvalues.
            else if (isa<ArrayType>(f->range->type)) {
                if (not f->range->is_lvalue and not MaterialiseTemporary(f->range, f->range->type))
                    return e->sema.set_errored();
            }

            /// Tuples are converted into arrays, as above.
            ///
            /// However, in this case, we don’t have a tuple literal, so we can’t
            /// do any transformations in-place and instead just check if we can
            /// treat this as an array (since an n-tuple of elements of the same
            /// type T is equivalent to a T[n], even in memory layout).
            else if (auto t = dyn_cast<TupleType>(f->range->type)) {
                auto tys = t->field_types();
                for (auto ty : tys | vws::drop(1))
                    if (ty != tys[0])
                        return NotIterable();
                if (not ConvertTupleToArray(tys[0], t->fields.size())) return false;
            }

            /// Anything else is not iterable.
            else { return NotIterable(); }

            /// The loop variable is an lvalue of the element type of the range.
            auto s = cast<SingleElementTypeBase>(f->range->type);

            /// Now check the loop variable. We can’t analyse the body if this fails
            /// since it is probably going to use it.
            if (f->iter) {
                f->iter->stored_type = s->elem;
                Expr* expr = f->iter;
                if (not Analyse(expr)) return e->sema.set_errored();
                Assert(expr == f->iter, "Must not wrap loop variable");
            }

            /// Check the index variable, if it exists.
            if (f->index) {
                f->index->stored_type = Type::Int;
                Expr* expr = f->index;
                if (not Analyse(expr)) return e->sema.set_errored();
                Assert(expr == f->index, "Must not wrap index variable");
            }

            /// Finally, check the body.
            loop_stack.push_back(f);
            defer { loop_stack.pop_back(); };
            Expr* expr = f->body;
            Analyse(expr);
            Assert(expr == f->body, "Body of for-in expression must be a block");
        } break;

        /// With expression as in Pascal or JavaScript.
        case Expr::Kind::WithExpr: {
            auto w = cast<WithExpr>(e);

            /// The controlling expression is kind of important for this.
            if (not Analyse(w->object)) return e->sema.set_errored();

            /// TODO: Temporary materialisation if this is an rvalue and has a body.
            UnwrapInPlace(w->object, true);
            if (not w->object->is_lvalue) return Error(
                e,
                "Object of with expression must be an lvalue"
            );

            /// Object must have members.
            if (not isa<SliceType, StructType, ArrayType>(w->object->type.desugared)) return Error(
                e,
                "Type '{}' is not a struct, array, or slice",
                w->object->type
            );

            /// These come in two forms: with and without a body. If a with
            /// expression has a body, then the controlling expression is only
            /// ‘open’ within that body; otherwise, it is open for the rest of
            /// the enclosing scope.
            with_stack.push_back(w->object);
            if (w->body) {
                defer { with_stack.pop_back(); };
                Expr* expr = w->body;
                if (Analyse(expr)) {
                    Assert(expr == w->body, "May not change block expression");

                    /// Forward return type of block.
                    w->stored_type = w->body->type;
                    w->is_lvalue = w->body->is_lvalue;
                }
            } else {
                w->stored_type = Type::Void;
            }
        } break;

        /// Export.
        case Expr::Kind::ExportExpr: {
            auto exp = cast<ExportExpr>(e);

            /// Non-modules cannot export anything.
            if (not mod->is_logical_module) return Error(
                exp->location,
                "'export' cannot be used outside of modules"
            );

            /// Check that an expression is exportable.
            auto Exportable = [&]([[maybe_unused]] BlockExpr* sc, StringRef name, StringRef entity_kind) {
                if (name.empty()) return Error(
                    exp->location,
                    "Cannot export anonymous {}",
                    entity_kind
                );

                return true;
            };

            /// Structs can be exported.
            if (auto s = dyn_cast<StructType>(exp->expr)) {
                if (not Exportable(s->scope, s->name, "struct")) return true;
                Analyse(exp->expr);
                mod->exports[s->name].push_back(s);
                break;
            }

            /// Procedures can be exported.
            if (auto p = dyn_cast<ProcDecl>(exp->expr)) {
                if (not Exportable(p->body, p->name, "procedure")) return true;
                p->linkage = p->linkage == Linkage::Imported ? Linkage::Reexported : Linkage::Exported;
                Analyse(exp->expr);
                mod->exports[p->name].push_back(p);
                break;
            }

            /// Anything else is invalid.
            return Error(exp->location, "'export' must qualify a declaration");
        }

        /// Unary expressions.
        case Expr::Kind::UnaryPrefixExpr: {
            auto u = cast<UnaryPrefixExpr>(e);
            if (not Analyse(u->operand)) return e->sema.set_errored();
            switch (u->op) {
                default: Unreachable("Invalid unary prefix operator");

                /// Boolean negation.
                case Tk::Not:
                    EnsureCondition(u->operand);
                    u->stored_type = Type::Bool;
                    break;

                /// Explicit dereference.
                case Tk::Star: {
                    auto ref = dyn_cast<ReferenceType>(u->operand->type);
                    if (not ref) return Error(
                        u,
                        "Cannot dereference value of non-reference type '{}'",
                        u->operand->type
                    );

                    /// The type is the element type of the reference.
                    u->stored_type = ref->elem;
                    u->is_lvalue = true;
                } break;
            }
        } break;

        /// Array, slice, and tuple subscripting, as well as array types.
        case Expr::Kind::SubscriptExpr: {
            auto s = cast<SubscriptExpr>(e);
            if (not Analyse(s->object)) return e->sema.set_errored();

            /// If the LHS is a type, then this is actually an array type.
            if (auto ty = dyn_cast<TypeBase>(s->object)) {
                e = new (mod) ArrayType(ty, s->index, s->location);
                return Analyse(e);
            }

            /// Handle slices.
            UnwrapInPlace(s->object, true);
            if (isa<SliceType>(s->object->type)) {
                auto ty = cast<SingleElementTypeBase>(s->object->type);

                /// Index must be an integer.
                if (Analyse(s->index) and not Convert(s->index, Type::Int)) Error(
                    s->index,
                    "Index of subscript must be an integer, but was '{}'",
                    s->index->type
                );

                /// Slice subscripts are always lvalues.
                s->stored_type = ty->elem;
                s->is_lvalue = true;
                break;
            }

            /// Handle tuples.
            if (auto t = dyn_cast<TupleType>(s->object->type)) {
                MaterialiseTemporaryIfRValue(s->object);

                /// Index must be an integer.
                if (Analyse(s->index) and not Convert(s->index, Type::Int)) return Error(
                    s->index,
                    "Index of subscript must be an integer, but was '{}'",
                    s->index->type
                );

                /// Index must be an integer and in range.
                if (not EvaluateAsIntegerInPlace(s->index, true)) break;
                auto idx = cast<ConstExpr>(s->index)->value.as_int().getZExtValue();
                if (idx >= t->fields.size()) return Error(
                    e,
                    "Index {} out of range for tuple type '{}'",
                    idx,
                    s->object->type
                );

                e = new (mod) TupleIndexExpr(
                    s->object,
                    t->fields[idx],
                    s->location
                );
                break;
            }

            Error(
                s,
                "Cannot perform subscripting on type '{}'",
                s->object->type
            );
        } break;

        /// Binary operators are complicated.
        case Expr::Kind::BinaryExpr: {
            auto b = cast<BinaryExpr>(e);

            /// Assignment (but not compound or reference assignment) has
            /// different semantics wrt analysing the RHS, so get that out
            /// of the way early.
            if (b->op == Tk::Assign) {
                /// The LHS must not be of reference type.
                if (not Analyse(b->lhs)) return e->sema.set_errored();
                UnwrapInPlace(b->lhs, true);
                if (not b->lhs->is_lvalue) {
                    /// Nicer error message for parameters.
                    if (IsInParameter(b->lhs)) return Error(b, "Cannot assign to an 'in' parameter");
                    return Error(b, "Left-hand side of `=` must be an lvalue");
                }

                /// Perform construction from the RHS.
                auto c = Construct(b->location, b->lhs->type, b->rhs);
                if (not c) return b->sema.set_errored();
                e = new (mod) AssignExpr(b->lhs, c, b->location);
                return true;
            }

            /// Check LHS and RHS.
            if (not Analyse(b->lhs) or not Analyse(b->rhs)) return e->sema.set_errored();

            /// Check if the operator is a bitwise operator.
            auto IsBitwise = [&] {
                return b->op == Tk::Land or b->op == Tk::Lor or b->op == Tk::Xor;
            };

            /// Common checks for arithmetic operators.
            auto CheckArithOperands = [&] {
                /// Both types must be (arrays of) integers.
                auto info_lhs = b->lhs->type.strip_arrays;
                auto info_rhs = b->rhs->type.strip_arrays;
                auto lhs_base = info_lhs.base_type.desugared;
                auto rhs_base = info_rhs.base_type.desugared;

                /// If this is a bitwise operation, we also support bitmask enums.
                auto lhs_enum = dyn_cast<EnumType>(lhs_base);
                auto rhs_enum = dyn_cast<EnumType>(rhs_base);
                if (IsBitwise() and lhs_enum and rhs_enum) {
                    auto ReportNotMaskEnum = [&](Type ty, Location loc) {
                        Error(
                            loc,
                            "Enum '{}' is not a bitmask enum",
                            ty
                        );
                        return b->sema.set_errored();
                    };

                    /// Check that both are mask enums and compatible.
                    if (not lhs_enum->mask) return ReportNotMaskEnum(lhs_enum, b->lhs->location);
                    if (not rhs_enum->mask) return ReportNotMaskEnum(rhs_enum, b->rhs->location);
                    if (not Convert(b->rhs, b->lhs->type) and not Convert(b->lhs, b->rhs->type)) return Error(
                        b,
                        "Cannot perform '{}' on incompatible types '{}' and '{}'",
                        Spelling(b->op),
                        b->lhs->type,
                        b->rhs->type
                    );
                }

                /// Otherwise, only integers are allowed.
                else if (not lhs_base.is_int(false) or not rhs_base.is_int(false)) {
                    return Error(
                        b,
                        "Invalid operands for '{}': '{}' and '{}'",
                        Spelling(b->op),
                        b->lhs->type,
                        b->rhs->type
                    );
                }

                /// If either one is an array, the types must be the same.
                if (
                    (isa<ArrayType>(b->lhs->type.desugared) or isa<ArrayType>(b->rhs->type.desugared)) and
                    b->lhs->type != b->rhs->type
                ) return Error( //
                    b,
                    "Operator '{}' is only valid for arrays if both operands "
                    "have the same type, but got '{}' and '{}'",
                    Spelling(b->op),
                    b->lhs->type,
                    b->rhs->type
                );

                return true;
            };

            switch (b->op) {
                default: Unreachable("Invalid binary operator");

                /// Handled above.
                case Tk::Assign: Unreachable();

                /// Note: `and` and `or` are emitted lazily, but this
                /// is irrelevant to semantic analysis, so we don’t
                /// care about that here.
                case Tk::And:
                case Tk::Or: {
                    /// Operands are rvalues of non-reference type.
                    InsertLValueToRValueConversion(b->lhs);
                    InsertLValueToRValueConversion(b->rhs);

                    /// Operands must be booleans.
                    auto Check = [&](Expr*& e) {
                        if (Convert(e, Type::Bool)) return;
                        if (e->type.is_int(false)) {
                            /// TODO: Fix-it hint.
                            Error(
                                e,
                                "Operands of '{}' must be of type '{}', but was '{}'. "
                                "Help: did you mean to use 'l{}' instead for a bitwise operation?",
                                Spelling(b->op),
                                Type::Bool,
                                e->type,
                                Spelling(b->op)
                            );
                        } else {
                            Error(
                                e,
                                "Operands of '{}' must be of type '{}', but was '{}'",
                                Spelling(b->op),
                                Type::Bool,
                                e->type
                            );
                        }
                    };

                    /// Even if this fail, we know that the type of this is bool.
                    Check(b->lhs);
                    Check(b->rhs);
                    b->stored_type = Type::Bool;
                } break;

                /// Arithmetic, bitwise, and boolean operators.
                case Tk::Plus:
                case Tk::Minus:
                case Tk::Star:
                case Tk::StarStar:
                case Tk::Slash:
                case Tk::Percent:
                case Tk::Land:
                case Tk::Lor:
                case Tk::Xor:
                case Tk::ShiftLeft:
                case Tk::ShiftRight:
                case Tk::ShiftRightLogical: {
                    /// Operands are rvalues of non-reference type.
                    InsertLValueToRValueConversion(b->lhs);
                    InsertLValueToRValueConversion(b->rhs);
                    if (not CheckArithOperands()) return false;

                    /// The smaller integer is cast to the larger type if they
                    /// don’t have the same size. Integer conversions from a
                    /// smaller to a larger type can never fail, which is why
                    /// we assert rather than error here.
                    if (b->lhs->type != b->rhs->type) {
                        auto lsz = b->lhs->type.size(ctx);
                        auto rsz = b->rhs->type.size(ctx);
                        if (lsz >= rsz) Assert(Convert(b->rhs, b->lhs->type));
                        else Assert(Convert(b->lhs, b->rhs->type));
                    }

                    /// The result type is that integer type.
                    b->stored_type = b->lhs->type;
                } break;

                /// Comparison operators.
                case Tk::EqEq:
                case Tk::Neq:
                case Tk::Lt:
                case Tk::Gt:
                case Tk::Le:
                case Tk::Ge: {
                    /// Look through rvalues, but allow comparing references if
                    /// their addresses are taken.
                    InsertLValueToRValueConversion(b->lhs);
                    InsertLValueToRValueConversion(b->rhs);

                    /// Both types must be equal.
                    if (
                        not Convert(b->lhs, b->rhs->type) and
                        not Convert(b->rhs, b->lhs->type)
                    ) return Error( //
                        b,
                        "Cannot compare '{}' with '{}'",
                        b->lhs->type,
                        b->rhs->type
                    );

                    /// If we’re comparing arrays, the result is an array of bools.
                    Type res = BuiltinType::Bool(mod, b->location);
                    if (auto l = dyn_cast<ArrayType>(b->lhs->type.desugared)) {
                        SmallVector<Expr*> dims;
                        do {
                            dims.push_back(l->dim_expr);
                            l = dyn_cast<ArrayType>(l->elem.desugared);
                        } while (l);

                        /// Build an array of the same shape.
                        for (auto d : vws::reverse(dims)) {
                            res = new (mod) ArrayType(res, d, b->location);
                            AnalyseAsType(res);
                        }
                    }

                    /// The type of a comparison is bool.
                    b->stored_type = res;
                } break;

                /// Value assignment. The LHS has to be an lvalue.
                case Tk::PlusEq:
                case Tk::MinusEq:
                case Tk::StarEq:
                case Tk::StarStarEq:
                case Tk::SlashEq:
                case Tk::PercentEq:
                case Tk::ShiftLeftEq:
                case Tk::ShiftRightEq:
                case Tk::ShiftRightLogicalEq: {
                    /// These operators never perform reference reassignment, which
                    /// means the LHS must not be of reference type.
                    UnwrapInPlace(b->lhs, true);
                    if (not b->lhs->is_lvalue) {
                        /// Nicer error message for parameters.
                        if (IsInParameter(b->lhs)) return Error(b, "Cannot assign to an 'in' parameter");
                        return Error(b, "Left-hand side of `=` must be an lvalue");
                    }

                    /// Compound assignment.
                    InsertLValueToRValueConversion(b->rhs);
                    if (not CheckArithOperands()) return false;

                    /// The RHS must be convertible to the LHS.
                    if (not Convert(b->rhs, b->lhs->type)) return Error(
                        b,
                        "Cannot assign '{}' to '{}'",
                        b->rhs->type,
                        b->lhs->type
                    );

                    /// The type of the expression is the type of the LHS.
                    b->stored_type = b->lhs->type;
                    b->is_lvalue = true;
                } break;

                /// Reference assignment.
                ///
                /// See [expr.binary.refassign] for an explanation of the algorithm below.
                /// TODO: Investigate how references and scoped pointers interact here and
                ///       whether all possible interactions are actually valid.
                case Tk::RDblArrow: {
                    /// 1.
                    if (not isa<ReferenceType, ScopedPointerType, OptionalType>(b->lhs->type)) return Error(
                        b,
                        "LHS of reference binding must be a reference, but was '{}'",
                        b->lhs->type
                    );

                    /// For error reporting *only*.
                    auto ltype_saved = b->lhs->type;
                    auto rtype_saved = b->rhs->type;

                    /// 2/3.
                    if (not b->lhs->is_lvalue) InsertImplicitDereference(b->lhs, 1);
                    if (not isa<ReferenceType, ScopedPointerType, OptionalType>(b->lhs->type)) return Error(
                        b,
                        "LHS of reference binding is not an lvalue"
                    );

                    /// A single optional level is allowed.
                    if (auto opt = dyn_cast<OptionalType>(b->lhs->type)) {
                        if (not isa<ReferenceType, ScopedPointerType>(opt->elem)) return Error(
                            b,
                            "LHS of reference binding must be an (optional) reference, but was {}",
                            b->lhs->type
                        );

                        if (not Convert(b->rhs, opt->elem)) return Error(
                            b,
                            "No implicit conversion from '{}' to '{}'",
                            b->rhs->type,
                            opt->elem
                        );

                        /// If the initialiser is non-optional reference, then it is active now.
                        LValueState.ActivateOptional(b->lhs);

                        b->stored_type = b->lhs->type;
                        b->is_lvalue = true;
                        break;
                    }

                    /// 4/5.
                    auto d_l = b->lhs->type.ref_depth;
                    auto d_r = b->rhs->type.ref_depth;
                    if (d_l < d_r) InsertImplicitDereference(b->rhs, d_r - d_l);
                    else if (d_r > d_l) InsertImplicitDereference(b->lhs, d_l - d_r);

                    /// 6.
                    if (not Convert(b->rhs, b->lhs->type)) {
                        if (
                            b->rhs->type != rtype_saved or
                            b->lhs->type != ltype_saved
                        ) {
                            Diag::Note(
                                mod->context,
                                b->location,
                                "In reference binding to '{}' from '{}'",
                                ltype_saved.str(mod->context->use_colours, true),
                                rtype_saved.str(mod->context->use_colours, true)
                            );
                        }

                        return Error(
                            b,
                            "No implicit conversion from '{}' to '{}'",
                            b->rhs->type,
                            b->lhs->type
                        );
                    }

                    /// 7.
                    b->stored_type = b->lhs->type;
                    b->is_lvalue = true;
                } break;
            }
        }
    }

    /// Can’t check for 'ok' as that may not be set yet.
    return not e->sema.errored;
}

bool src::Sema::AnalyseAsType(Type& e, bool diag_if_not_type) {
    Analyse(e.ptr);
    if (e->sema.errored) return false;
    Assert(e->sema.ok);

    if (auto t = dyn_cast<TupleExpr>(e.ptr)) {
        SmallVector<FieldDecl*> ts;
        for (auto& el : t->elements) {
            ts.emplace_back(new (mod) FieldDecl(String(), Type{el}, el->location));
            if (not AnalyseAsType(ts.back()->stored_type)) return false;
        }

        e = new (mod) TupleType(ts, t->location);
        Analyse(e.ptr);
        return true;
    }

    if (not isa<TypeBase>(e.ptr)) {
        if (diag_if_not_type) Error(e->location, "Not a type");
        return false;
    }

    return true;
}

template <bool allow_undefined>
bool src::Sema::AnalyseDeclRefExpr(Expr*& e) {
    auto d = cast<DeclRefExpr>(e);

    /// Resolve a DeclRefExpr in place.
    auto ResolveInPlace = [&](Expr* decl) {
        d->stored_type = decl->type;
        d->is_lvalue = decl->is_lvalue;
    };

    /// Some DeclRefExprs may already be resolved to a node. If this one
    /// isn’t, find the nearest declaration with the given name.
    if (not d->decl) {
        /// First, search active decl contexts.
        for (auto& decl_context : decl_contexts) {
            if (auto [decl, type] = decl_context.find(*this, d->name)) {
                d->decl = decl;
                d->is_lvalue = decl->is_lvalue;
                d->stored_type = type;
                return true;
            }
        }

        /// Find the declaration in the scope it was declared in.
        auto* const decls = [&] -> BlockExpr::Symbols* {
            /// Try to find a declared symbol with that name in the scope
            /// that the name was found in.
            if (auto [syms, escapes_isolated_context] = d->scope->find(d->name, false)) {
                /// Variable declarations may not be referenced across a context boundary,
                /// except in an unevaluated context, such as `typeof`.
                if (not unevaluated and escapes_isolated_context and isa<LocalDecl>(syms->front())) {
                    Error(
                        d,
                        "Variable '{}' cannot be accessed here.",
                        d->name
                    );

                    Note(
                        syms->front(),
                        "Access references this variable. Did you mean to declare it 'static'?"
                    );

                    return nullptr;
                }
                return syms;
            }

            /// Check if this is an imported module.
            auto m = rgs::find(mod->imports, d->name, &ImportedModuleRef::logical_name);
            if (m != mod->imports.end()) {
                e = new (mod) ModuleRefExpr(m->mod, d->location);
                return nullptr;
            }

            /// Check if this is a symbol imported from an open module.
            for (auto& i : mod->imports) {
                auto exp = i.mod->exports.find(d->name);
                if (exp == i.mod->exports.end()) continue;
                return &exp->second;
            }

            /// Check if this is this module.
            if (mod->is_logical_module and d->name == mod->name) {
                e = new (mod) ModuleRefExpr(mod, d->location);
                return nullptr;
            }

            if constexpr (not allow_undefined) Error(e, "Unknown symbol '{}'", d->name);
            return nullptr;
        }();

        /// Null here need not signal failure, but it definitely does
        /// mean that we should not attempt to do anything else.
        if (not decls) return e->sema.ok;

        /// If there are multiple declarations, and the declarations
        /// are functions, then construct an overload set.
        if (isa<ProcDecl>(decls->back()) and decls->size() > 1) {
            /// Take all declarations, starting from the back, that are
            /// procedure declarations; stop if we encounter a variable
            /// declaration as that shadows everything before it.
            SmallVector<ProcDecl*> overloads;
            for (auto*& o : rgs::subrange(decls->begin(), decls->end()) | vws::reverse) {
                if (auto p = dyn_cast<ProcDecl>(o)) {
                    if (not Analyse(o)) continue;
                    overloads.push_back(p);
                } else {
                    break;
                }
            }

            /// If there is only one procedure, don’t construct an overload set.
            if (overloads.size() == 1) {
                ResolveInPlace(overloads.front());
                return true;
            }

            /// Otherwise, construct an overload set.
            e = new (mod) OverloadSetExpr(overloads, d->location);
            return Analyse(e);
        }

        /// Otherwise, only keep the last decl.
        d->decl = decls->back();
    }

    /// The type of this is the type of the referenced expression.
    if (not Analyse(d->decl)) return d->sema.set_errored();

    /// If this is an alias, resolve it to what the alias points to.
    if (auto a = dyn_cast<AliasExpr>(d->decl)) {
        e->sema.unset(); /// Other instances of this will have to be replaced w/ this again.
        d->decl = a->expr->ignore_paren_refs;
    }

    /// If this is a type, replace it w/ a sugared type.
    if (auto ty = dyn_cast<TypeBase>(d->decl)) {
        e->sema.unset(); /// Other instances of this will have to be replaced w/ this again.
        e = new (mod) SugaredType(d->name, ty, d->location);
    }

    /// If it is a variable declaration, replace it w/ a variable reference.
    else if (isa<LocalDecl>(d->decl)) {
        e->sema.unset(); /// Other instances of this will have to be replaced w/ this again.
        e = new (mod) LocalRefExpr(curr_proc, cast<LocalDecl>(d->decl), d->location);
        return Analyse(e);
    }

    /// Otherwise, this stays as a DeclRefExpr.
    else { ResolveInPlace(d->decl); }
    return true;
}

void src::Sema::AnalyseExplicitCast(Expr*& e, [[maybe_unused]] bool is_hard) {
    auto c = cast<CastExpr>(e);

    /// Check the type. We can’t cast if there is a problem w/ it.
    if (not AnalyseAsType(c->stored_type) or not Analyse(c->operand)) {
        e->sema.set_errored();
        return;
    }

    /// If the types are convertible, then the cast is fine.
    if (Convert(c->operand, e->type)) return;

    /// Enum/Integer-to-integer conversions are fine.
    auto from = c->operand->type.desugared;
    auto to = c->type.desugared;
    if ((from.is_int(true) or isa<EnumType>(from)) and to.is_int(true)) {
        InsertLValueToRValueConversion(c->operand);
        return;
    }

    /// Integer-to-enum casts.
    if (auto enum_ty = dyn_cast<EnumType>(to); enum_ty and from.is_int(false)) {
        InsertLValueToRValueConversion(c->operand);

        /// Hard-casts are always allowed.
        if (c->cast_kind == CastKind::Hard) return;

        /// Soft casts require that the value be known, at compile time, to be
        /// representable by the enum type.
        if (not EvaluateAsIntegerInPlace(c->operand, false, enum_ty->underlying_type)) {
            Error(
                e,
                "Non-constant expression of type '{}' cannot be converted to enum type '{}'",
                c->operand->type,
                e->type.str(ctx->use_colours, true)
            );
            return;
        }

        /// Check the value exists.
        auto from_val = cast<ConstExpr>(c->operand)->value.as_int();
        for (auto n : enum_ty->enumerators) {
            auto& val = n->value;
            if (val == from_val) return;
        }

        Error(
            e,
            "Type '{}' has no enumerator with value '{}'",
            e->type.str(ctx->use_colours, true),
            from_val
        );
        return;
    }

    /// Other conversions may be added in the future.
    /// TODO: Actually add some hard casts.
    Error(
        e,
        "Unsupported cast from '{}' to '{}'",
        c->operand->type,
        e->type
    );
}

bool src::Sema::AnalyseInvoke(Expr*& e, bool direct_child_of_block) {
    /// Analyse the callee first.
    ///
    /// If it is a DeclRefExpr, we recognise builtins here,
    /// so the decl ref being undefined is not an error.
    auto invoke = cast<InvokeExpr>(e);
    if (isa<DeclRefExpr>(invoke->callee)) {
        AnalyseDeclRefExpr<true>(invoke->callee);

        /// Check for builtins.
        if (auto d = dyn_cast<DeclRefExpr>(invoke->callee); d and not d->decl) {
            auto b = llvm::StringSwitch<std::optional<Builtin>>(d->name) // clang-format off
                .Case("new", Builtin::New)
                .Case("__srcc_delete", Builtin::Destroy)
                .Case("__srcc_memcpy", Builtin::Memcpy)
                .Case("__srcc_new", Builtin::New)
                .Default(std::nullopt);

            /// Found a builtin.
            if (b.has_value()) {
                e = new (mod) InvokeBuiltinExpr(
                    *b,
                    std::move(invoke->args),
                    invoke->location
                );
                return Analyse(e);
            }

            /// Unknown symbol.
            Error(invoke->callee, "Unknown symbol '{}'", d->name);
            return e->sema.errored;
        } // clang-format on
    }

    /// Otherwise, the callee must be a valid symbol.
    if (not Analyse(invoke->callee)) { return e->sema.set_errored(); }

    /// Perform overload resolution.
    auto ResolveOverloadSet = [&](OverloadSetExpr* o, Expr*& callee_ref) {
        Assert(
            invoke->init_args.empty(),
            "Assigning to a procedure call is currently not supported"
        );

        auto res = PerformOverloadResolution(invoke->location, o->overloads, invoke->args);
        if (not res) return e->sema.set_errored();
        auto callee = res.resolved();
        invoke->stored_type = callee->type.callable->ret_type;

        /// Wrap the procedure with a DeclRefExpr, for backend reasons.
        auto dr = new (mod) DeclRefExpr(callee->name, curr_scope, invoke->location);
        dr->decl = callee;
        callee_ref = dr;
        Assert(Analyse(callee_ref));
        return true;
    };

    /// Handle regular calls.
    auto PerformSimpleCall = [&](Expr*& callee) {
        auto ptype = callee->type.callable;
        Assert(
            invoke->init_args.empty(),
            "Assigning to a procedure call is currently not supported"
        );

        /// Callee must be an rvalue.
        InsertLValueToRValueConversion(callee);

        /// Analyse the arguments.
        for (auto& arg : invoke->args)
            if (not Analyse(arg))
                e->sema.set_errored();

        /// Make sure the types match.
        for (usz i = 0; i < invoke->args.size(); i++) {
            const bool variadic = i >= ptype->parameters.size();
            if (not variadic) {
                /// Keep lvalues here. We’ll handle them below.
                auto info = &ptype->parameters[i];
                if (not Convert(invoke->args[i], info->type, true)) {
                    Error(
                        invoke->args[i],
                        "Argument type '{}' is not convertible to parameter type '{}'",
                        invoke->args[i]->type,
                        info->type
                    );
                    e->sema.set_errored();
                    continue;
                }
            }

            FinaliseInvokeArgument(
                invoke->args[i],
                variadic ? nullptr : &ptype->parameters[i]
            );
        }

        /// Make sure there are as many arguments as parameters.
        if (
            invoke->args.size() < ptype->parameters.size() or
            (invoke->args.size() != ptype->parameters.size() and not ptype->variadic)
        ) {
            Error(
                e,
                "Expected {} arguments, but got {}",
                ptype->parameters.size(),
                invoke->args.size()
            );
        }

        /// The type of the expression is the return type of the
        /// callee. Invoke expressions are never lvalues.
        invoke->stored_type = ptype->ret_type;
        return not e->sema.errored;
    };

    /// Not a type.
    if (not isa<TypeBase>(invoke->callee)) {
        /// Handle member function calls.
        if (invoke->callee->type == Type::MemberProc) {
            auto m = cast<MemberAccessExpr>(invoke->callee);

            /// Add the object as the first argument if this is not a smp.
            if (not m->field->is_smp) invoke->args.insert(invoke->args.begin(), m->object);
            if (auto o = dyn_cast<OverloadSetExpr>(m->field)) return ResolveOverloadSet(o, m->field);
            return PerformSimpleCall(m->field);
        }

        /// Perform overload resolution, if need be.
        if (auto o = dyn_cast<OverloadSetExpr>(invoke->callee))
            return ResolveOverloadSet(o, invoke->callee);

        /// If the callee is of function type, then this is a function call.
        if (isa<ProcType, ClosureType>(invoke->callee->type))
            return PerformSimpleCall(invoke->callee);

        /// Fallthrough in case this is some expression that must first
        /// be converted to a type (e.g. TupleExpr -> TupleType).
    }

    /// Otherwise, this only makes sense if this is a type.
    Type ty{invoke->callee};
    if (not AnalyseAsType(ty, false)) {
        e->sema.set_errored();
        return Error(ty->location, "Expected procedure or type");
    }

    /// This is indeed a type.
    Assert(isa<TypeBase>(ty), "Should be a type");
    if (not MakeDeclType(ty)) e->sema.set_errored();

    /// If the invoke expression’s arguments are parenthesised, then
    /// this is actually a literal, not a declaration.
    if (not invoke->naked) {
        Assert(
            invoke->init_args.empty(),
            "Assigning to a procedure call is currently not supported"
        );

        auto c = Construct(invoke->location, ty, invoke->args);
        if (not c) return e->sema.set_errored();
        c->stored_type = ty;
        c->location = invoke->location;
        e = c;
        return true;
    }

    /// Check if declarations are even allowed here.
    if (not direct_child_of_block) return Error(
        e,
        "Variable declarations cannot be subexpressions"
    );

    /// The arguments must be DeclRefExprs.
    for (auto& arg : invoke->args) {
        if (not isa<DeclRefExpr>(arg)) {
            Error(arg, "Expected identifier in declaration");
            e->sema.set_errored();
        }
    }

    /// Helper to create a var decl.
    auto MakeVar = [&](Expr* name, SmallVector<Expr*> init) -> LocalDecl* {
        return new (mod) LocalDecl(
            cast<DeclRefExpr>(name)->name,
            ty,
            std::move(init),
            LocalKind::Variable,
            name->location
        );
    };

    /// Rewrite the invocation to a declaration. Type checking
    /// for the initialiser is done elsewhere.
    if (invoke->args.size() == 1) {
        e = MakeVar(invoke->args.front(), invoke->init_args);
        return Analyse(e);
    }

    /// If the invoke expression contains multiple declarations
    /// rewrite to a VarListDecl expr.
    Todo();
}

bool src::Sema::AnalyseInvokeBuiltin(Expr*& e) {
    auto invoke = cast<InvokeBuiltinExpr>(e);
    switch (invoke->builtin) {
        /// Call a destructor.
        case Builtin::Destroy: {
            /// Destroy takes one operand.
            if (invoke->args.size() != 1) return Error(
                e,
                "Expected 1 argument, but got {}",
                invoke->args.size()
            );

            /// The operand must be an lvalue.
            if (not Analyse(invoke->args[0])) return e->sema.set_errored();
            if (not isa<LocalRefExpr>(invoke->args[0])) return Error(
                e,
                "Operand of __srcc_destroy must be a local variable."
            );

            /// Lvalue-to-rvalue conversion takes care of move semantics.
            InsertLValueToRValueConversion(invoke->args[0]);

            /// Destroy returns nothing.
            invoke->stored_type = Type::Void;
            return true;
        }

        /// Copy memory.
        case Builtin::Memcpy: {
            /// Operands must be references and a size.
            if (invoke->args.size() != 3) return Error(
                e,
                "__srcc_memcpy takes 3 arguments, but got {}",
                invoke->args.size()
            );

            auto CheckRefArg = [&](usz i) {
                if (not Analyse(invoke->args[i])) return;
                if (not Convert(invoke->args[i], Type::VoidRef)) Error(
                    invoke->args[i],
                    "No conversion from {} to a reference",
                    invoke->args[i]->type
                );
            };

            CheckRefArg(0);
            CheckRefArg(1);
            if (Analyse(invoke->args[2]) and not Convert(invoke->args[2], Type::Int)) Error(
                invoke->args[2],
                "No conversion from {} to {}",
                invoke->args[2]->type,
                Type::Int
            );

            /// This returns nothing.
            invoke->stored_type = Type::Void;
            return true;
        }

        /// Allocate a scoped pointer.
        case Builtin::New: {
            /// New takes one operand.
            if (invoke->args.size() != 1) return Error(
                e,
                "new takes 1 argument, but got {}",
                invoke->args.size()
            );

            /// Operand must be a type.
            Type ty{invoke->args[0]};
            if (not AnalyseAsType(ty)) return e->sema.set_errored();

            /// Type is a scoped pointer to the allocated type.
            invoke->stored_type = new (mod) ScopedPointerType(
                ty,
                invoke->location
            );

            return true;
        }
    }

    Unreachable();
}

void src::Sema::AnalyseProcedure(ProcDecl* proc) {
    tempset curr_proc = proc;

    /// Check the type first.
    if (not AnalyseProcedureType(proc)) return;

    /// If there is no body, then there is nothing to do.
    if (not proc->body) {
        proc->params.clear();
        return;
    }

    /// Don’t analyse procedures twice.
    Assert(not proc->sema.analysed);
    tempset curr_scope = proc->body;
    tempset unwind_entries = decltype(unwind_entries){};

    /// Assign a name to the procedure if it doesn’t have one.
    if (proc->name.empty())
        proc->name = mod->save(fmt::format("__srcc_lambda_{}", lambda_counter++));

    /// Sanity check.
    if (proc->ret_type == Type::Unknown and not proc->body->implicit) Diag::ICE(
        mod->context,
        proc->location,
        "Non-inferred procedure has unknown return type"
    );

    /// Add all named parameters to the body’s scope.
    for (auto* param : proc->params) {
        Expr* e = param;
        Analyse(e);
        Assert(e == param, "ParamDecls may not be replaced with another expression");
    }

    /// Analyse the body. If either it or the procedure
    /// contains an error, we can’t check if the procedure
    /// has a return statement.
    Expr* body = proc->body;
    Analyse(body);
    if (not body->sema.ok) return;

    /// If the body is `= <expr>`, infer the return type if
    /// there isn’t already one.
    ///
    /// Note that, if the type of the body is actually noreturn here,
    /// then this really is a noreturn function; if the body contains
    /// a return expression and is noreturn because of that, then the
    /// ret type would have already been inferred and would no longer
    /// be `unknown`, so we could never get here in that case.
    if (proc->body->implicit) {
        Assert(proc->body->exprs.size() == 1);
        Expr*& e = proc->body->exprs[0];

        /// Infer type.
        if (proc->ret_type == Type::Unknown) {
            cast<ProcType>(proc->stored_type)->ret_type = body->type;
        }

        /// Check that the type is valid.
        else if (not Convert(e, proc->ret_type)) {
            Error(
                e,
                "Cannot convert '{}' to return type '{}'",
                e->type,
                proc->ret_type
            );
            return;
        }

        /// Return value must be an rvalue.
        InsertLValueToRValueConversion(e);
    }

    /// Make sure all paths return a value.
    ///
    /// A function marked as returning void requires no checking and is allowed
    /// to not return at all. Accordingly, a function that actually never returns
    /// is also always fine, since 'noreturn' is convertible to any type.
    else if (proc->ret_type != Type::Void and body->type != Type::NoReturn) {
        if (proc->ret_type == Type::NoReturn) Error(
            proc->location,
            "Procedure '{}' returns despite being marked as 'noreturn'",
            proc->name
        );

        else Error(
            proc->location,
            "Procedure '{}' does not return a value on all paths",
            proc->name
        );
    }

    /// Handle unwinding.
    for (auto& uw : unwind_entries) {
        if (auto g = dyn_cast<GotoExpr>(uw.expr)) ValidateDirectBr(g, uw.in_scope);
        else UnwindUpTo(uw.in_scope, uw.to_scope, uw.expr);
    }
}

void src::Sema::AnalyseModule() {
    /// Resolve all imports.
    for (auto& i : mod->imports) {
        /// C++ header.
        if (i.is_cxx_header) continue;

        /// Already resolved.
        if (i.mod) continue;

        /// Regular module.
        for (auto& p : mod->context->import_paths) {
            auto mod_path = p / i.linkage_name.sv();
            mod_path.replace_extension(__SRCC_OBJ_FILE_EXT);

            std::error_code ec;
            if (auto ok = fs::exists(mod_path, ec); not ok or ec) continue;
            i.resolved_path = std::move(mod_path);

            /// Load the module.
            auto& f = mod->context->get_or_load_file(i.resolved_path);
            i.mod = Module::Deserialise(
                mod->context,
                i.linkage_name,
                i.import_location,
                {reinterpret_cast<const u8*>(f.data()), f.size()}
            );
            break;
        }

        if (not i.mod) Diag::Error(
            mod->context,
            i.import_location,
            "Could not find module '{}'",
            i.linkage_name
        );
    }

    /// Handle C++ headers.
    for (auto& h : mod->imports | vws::filter(&ImportedModuleRef::is_cxx_header)) {
        h.mod = Module::ImportCXXHeaders(
            mod->context,
            StringRef(h.linkage_name),
            h.logical_name,
            debug_cxx,
            h.import_location
        );
    }

    /// Stop if we couldn’t resolve all imports.
    if (mod->context->has_error()) return;

    /// Analyse all functions.
    for (auto f : mod->functions) {
        Expr* e = f;
        Analyse(e);
        Assert(e == f, "ProcDecl may not be replaced with another expression");
    }

    /// If a function is nested and any of its parents have
    /// captured locals, then it must take a chain pointer.
    auto TakesChainPointer = [&](ProcDecl* f) -> bool {
        if (not f->nested) return false;
        for (auto p = f->parent_or_null; p; p = p->parent_or_null)
            if (not p->captured_locals.empty())
                return true;
        return false;
    };

    /// Only after analysing all functions can we determine whether
    /// any of them take static chain pointers or not.
    for (auto f : mod->functions)
        if (TakesChainPointer(f))
            cast<ProcType>(f->type)->static_chain_parent = f->parent_or_null;
}

bool src::Sema::AnalyseProcedureType(ProcDecl* proc) {
    /// Validate the function type.
    if (not AnalyseAsType(cast<TypedExpr>(proc)->stored_type))
        return proc->sema.set_errored();

    /// Sanity check.
    auto ty = cast<TypedExpr>(proc)->stored_type;
    Assert(isa<ProcType>(ty), "Type of procedure is not a procedure type");
    return true;
}

void src::Sema::AnalyseRecord(RecordType* r) {
    /// Members must all be valid decls.
    for (auto f : r->fields) {
        if (not AnalyseAsType(f->stored_type) or not MakeDeclType(f->stored_type)) {
            f->sema.set_errored();
            r->sema.set_errored();
            continue;
        }

        /// Set offset of each field and determine the size and alignment
        /// of the struct.
        auto align = f->type.align(ctx);
        f->offset = r->stored_size.align(align);
        f->sema.set_done();
        r->stored_alignment = std::max(r->stored_alignment, align);
        r->stored_size += f->type.size(ctx);
    }

    /// Size must be a multiple of the alignment.
    r->stored_size.align(r->stored_alignment);

    /// We’re done if this is not a struct.
    auto s = dyn_cast<StructType>(r);
    if (not s) return;

    /// Mark the type as done here so member functions have a complete
    /// type to work with.
    s->sema.set_done();

    /// Analyse initialiser types first in case they call each other since
    /// we will have to perform overload resolution in that case.
    for (auto& i : s->initialisers) {
        auto ty = cast<ProcType>(i->type);
        ty->smp_parent = s;
        ty->smp_kind = SpecialMemberKind::Constructor;
        i->parent_struct = s;

        /// If the type is unknown, it defaults to the struct type.
        if (ty->ret_type == Type::Unknown) ty->ret_type = s;
        if (not AnalyseProcedureType(i)) r->sema.set_errored();
    }

    /// Member functions are isolated from everything above.
    tempset with_stack = decltype(with_stack){};

    /// Analyse deleter, if there is one.
    if (s->deleter) {
        auto ty = cast<ProcType>(s->deleter->type);
        ty->smp_parent = s;
        ty->smp_kind = SpecialMemberKind::Destructor;
        s->deleter->parent_struct = s;

        /// Analyse the destructor first since initialisers may call it.
        with_stack.push_back(new (mod) ImplicitThisExpr(s->deleter, s, {}));
        if (Analyse(s->deleter)) Assert(s->deleter->ret_type == Type::Void, "Deleter must return void");
    }

    /// Analyse initialisers.
    ///
    for (auto& i : s->initialisers) {
        /// The special initialiser `init = ()` is equivalent to
        /// `init = ::()`.
        if (
            i->body->implicit and
            i->body->exprs.size() == 1 and
            isa<TupleExpr>(i->body->exprs[0]) and
            cast<TupleExpr>(i->body->exprs[0])->elements.empty()
        ) {
            auto ctor = Construct(i->body->exprs[0]->location, s, {}, {}, true);
            if (not ctor) {
                i->sema.set_errored();
                s->sema.set_errored();
            } else {
                i->body->exprs[0] = new (mod) MaterialiseTemporaryExpr(
                    s,
                    ctor,
                    i->body->exprs[0]->location
                );
            }
        }

        with_stack.clear();
        with_stack.push_back(new (mod) ImplicitThisExpr(i, s, {}));
        if (not Analyse(i)) s->sema.set_errored();
    }

    /// Analyse member functions.
    for (auto& [_, procs] : s->member_procs) {
        for (auto m : procs) {
            m->parent_struct = s;
            if (not Analyse(m)) s->sema.set_errored();
        }
    }
}

bool src::Sema::AnalyseVariableInitialisation(
    Expr* e,
    ConstructExpr*& ctor,
    Type& type,
    SmallVectorImpl<Expr*>& init_args
) {
    /// Infer the type from the initialiser.
    if (type == Type::Unknown) {
        if (init_args.empty()) return Error(e, "Type inference requires an initialiser");
        if (init_args.size() != 1) Todo();
        if (not Analyse(init_args[0])) return e->sema.set_errored();
        InsertLValueToRValueConversion(init_args[0]);
        type = init_args[0]->type;
    }

    /// Type must be valid for a variable.
    if (not MakeDeclType(type)) return e->sema.set_errored();

    /// Type must be constructible from the initialiser args.
    ///
    /// Even if this fails, do not set the error flag for this variable
    /// since we now know its type and can still use it for other things.
    ctor = Construct(e->location, type, init_args, e);
    init_args.clear();
    return true;
}
