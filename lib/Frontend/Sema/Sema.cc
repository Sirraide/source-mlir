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
    auto to = type;
    auto from = from_ty;

    /// If the types are equal, then they’re convertible to one another.
    if (from == to) return true;

    /// Active optionals are convertible to the type they contain. There
    /// are no other valid conversions involving optionals at the moment.
    if (auto ty = Optionals.GetActiveOptionalType(ctx.expr)) {
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
        ctx.cast(CastKind::LValueToRValue, from);
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
                    from = ctx.cast(CastKind::LValueToRValue, from);
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
            if (ctx.try_evaluate(value) and value.as_int().getSignificantBits() <= to_size.bits()) {
                value.type = to;
                value.as_int() = value.as_int().trunc(unsigned(to_size.bits()));
                from = ctx.replace_with_constant(std::move(value));
                return true;
            }
        }

        /// Smaller integer types can be converted to larger integer types.
        if (from_size <= to_size) {
            from = ctx.cast(CastKind::LValueToRValue, from);
            from = ctx.cast(CastKind::Implicit, type);
            return true;
        }

        /// No other valid integer conversions.
        return false;
    }

    /// Nil is convertible to any optional type.
    if (from.is_nil and isa<OptionalType>(to)) {
        from = ctx.cast(CastKind::LValueToRValue, from);
        from = ctx.cast(CastKind::Implicit, type);
        return true;
    }

    /// Any type is convertible to the optional type of that type.
    if (auto opt = dyn_cast<OptionalType>(to); opt and from == opt->elem) {
        from = ctx.cast(CastKind::LValueToRValue, from);
        from = ctx.cast(CastKind::OptionalWrap, opt);
        return true;
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

bool src::Sema::TryConvert(ConversionSequence& seq, Expr* e, Type to) {
    ConversionContext<false> ctx(*this, seq, &e);
    bool ok = ConvertImpl(ctx, e->type, to);
    seq.score = ok ? ctx.score : Candidate::InvalidScore;
    return ok;
}

auto src::Sema::Unwrap(Expr* e, bool keep_lvalues) -> Expr* {
    Assert(e->sema.ok, "Unwrap() called on broken or unanalysed expression");

    /// Unwrap active optionals.
    if (auto opt = Optionals.GetActiveOptionalType(e)) {
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
    if (auto opt = Optionals.GetActiveOptionalType(e)) return opt->elem.strip_refs_and_pointers;
    return e->type.strip_refs_and_pointers;
}

/// ===========================================================================
///  Optional tracking.
/// ===========================================================================
src::Sema::OptionalState::ScopeGuard::ScopeGuard(src::Sema& S)
    : S(S),
      previous(std::exchange(S.Optionals.guard, this)) {}

src::Sema::OptionalState::ScopeGuard::~ScopeGuard() {
    for (auto&& [e, old_state] : changes) S.Optionals.tracked[e] = std::move(old_state);
    S.Optionals.guard = previous;
}

src::Sema::OptionalState::ActivationGuard::ActivationGuard(Sema& S, Expr* expr)
    : S(S), expr(expr) {
    S.Optionals.Activate(expr);
}

src::Sema::OptionalState::ActivationGuard::~ActivationGuard() {
    S.Optionals.Deactivate(expr);
}

auto src::Sema::OptionalState::GetObjectPath(MemberAccessExpr* m) -> std::pair<LocalDecl*, Path> {
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

void src::Sema::OptionalState::ChangeState(Expr* e, auto cb) {
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

void src::Sema::OptionalState::Activate(Expr* e) { // clang-format off
    ChangeState(e, utils::overloaded {
        [&] (LocalDecl* var) { tracked[var].active = true; },
        [&] (LocalDecl* var, Path path) {
            /// Activating a field only makes sense if the thing
            /// itself is active, so if we get here, it must be
            /// active.
            tracked[var].active = true;
            tracked[var].active_fields.push_back(std::move(path));
        }
    });
} // clang-format on

void src::Sema::OptionalState::Deactivate(Expr* e) { // clang-format off
    ChangeState(e, utils::overloaded {
        [&] (LocalDecl* var) { tracked[var].active = false; },
        [&] (LocalDecl* var, Path path) {
            /// Delete all paths that *start with* this path, as nested
            /// objects are now part of a different object.
            llvm::erase_if(tracked[var].active_fields, [&](Path& p) {
                return utils::starts_with(p, path);
            });
        }
    });
} // clang-format on

auto src::Sema::OptionalState::GetActiveOptionalType(Expr* e) -> OptionalType* {
    if (not e) return nullptr;
    e = e->ignore_paren_refs;

    if (auto var = dyn_cast<LocalDecl>(e)) {
        auto obj = tracked.find(var);
        if (obj == tracked.end() or not obj->second.active) return nullptr;
        return dyn_cast<OptionalType>(var->type.desugared);
    }

    if (auto m = dyn_cast<MemberAccessExpr>(e)) {
        if (auto [var, path] = GetObjectPath(m); var) {
            Assert(not path.empty());
            auto obj = tracked.find(var);
            if (obj == tracked.end() or not obj->second.active) return nullptr;
            if (
                rgs::any_of(
                    obj->second.active_fields,
                    [&](Path& p) { return utils::starts_with(p, path); }
                )
            ) {
                auto s = cast<StructType>(var->type.desugared);
                FieldDecl* f{};
                for (auto idx : path) {
                    f = s->all_fields[idx];
                    s = dyn_cast<StructType>(f->type.desugared);
                }

                return dyn_cast<OptionalType>(f->type.desugared);
            }
        }
    }

    return nullptr;
}

auto src::Sema::OptionalState::MatchNilTest(Expr* test) -> Expr* {
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

/// ===========================================================================
///  Overload Resolution
/// ===========================================================================
/// Refer to [expr.overload] in the manual for a detailed description of
/// the algorithm implemented below.
auto src::Sema::PerformOverloadResolution(
    Location where,
    ArrayRef<ProcDecl*> overloads,
    MutableArrayRef<Expr*> args,
    bool required
) -> ProcDecl* {
    /// First, analyse all arguments.
    for (auto& a : args)
        if (not Analyse(a))
            return nullptr;

    /// 1.
    SmallVector<Candidate> candidates;
    candidates.reserve(overloads.size());
    for (auto p : overloads) candidates.emplace_back(p);

    /// 2.
    for (auto& ci : candidates) {
        auto& params = ci.type->param_types;
        ci.arg_convs.resize(args.size());

        /// 2a.
        if (params.size() > args.size()) {
            ci.s = Candidate::Status::ArgumentCountMismatch;
            continue;
        }

        /// 2b.
        if (not ci.type->variadic and params.size() < args.size()) {
            ci.s = Candidate::Status::ArgumentCountMismatch;
            continue;
        }

        /// 2c/2d.
        for (usz j = 0; j < params.size(); j++) {
            if (not TryConvert(ci.arg_convs[j], args[j], params[j])) {
                ci.s = Candidate::Status::ArgumentTypeMismatch;
                ci.mismatch_index = j;
                break;
            } else {
                ci.score += ci.arg_convs[j].score;
            }
        }
    }

    /// 3/4.
    auto min = utils::UniqueMin(
        candidates,
        [](auto& c) { return c.s == Candidate::Status::Viable; },
        &Candidate::score
    );

    if (min == candidates.end()) {
        if (required) ReportOverloadResolutionFailure(where, candidates, args);
        return nullptr;
    }

    /// 5/6/7.
    auto proc = min->proc;
    auto& params = cast<ProcType>(proc->type)->param_types;
    for (usz i = 0; i < args.size(); i++) {
        if (i < params.size()) ApplyConversionSequence(args[i], std::move(min->arg_convs[i]));
        else {
            if (args[i]->type == Type::OverloadSet) {
                Error(args[i], "Cannot pass an overload set as a variadic argument");
                return nullptr;
            }
        }
        InsertLValueToRValueConversion(args[i]);
    }

    return proc;
}

void src::Sema::ReportOverloadResolutionFailure(
    Location where,
    ArrayRef<Candidate> overloads,
    ArrayRef<Expr*> args
) {
    using enum utils::Colour;
    utils::Colours C{ctx->use_colours};
    Error(where, "Overload resolution failed");

    /// Print all argument types.
    fmt::print(stderr, "  {}{}Arguments:\n", C(Bold), C(White));
    for (auto [i, e] : llvm::enumerate(args))
        fmt::print(stderr, "    {}{}{}. {}\n", C(Bold), C(White), i + 1, e->type.str(mod->context->use_colours, true));

    /// Print overloads and why each one was invalid.
    fmt::print(stderr, "\n  {}{}Overloads:\n", C(Bold), C(White));
    for (auto [i, o] : llvm::enumerate(overloads)) {
        if (i != 0) fmt::print("\n");
        fmt::print(
            stderr,
            "    {}{}{}. {}{}{}\n",
            C(Bold),
            C(White),
            i + 1,
            o.proc->type.str(mod->context->use_colours, true),
            C(White),
            C(Reset)
        );

        if (o.proc->location.seekable(mod->context)) {
            auto lc = o.proc->location.seek_line_column(mod->context);
            fmt::print(
                stderr,
                "       at {}:{}:{}\n",
                mod->context->file(o.proc->location.file_id)->path().string(),
                lc.line,
                lc.col
            );
        }

        fmt::print(stderr, "       ");
        switch (o.s) {
            case Candidate::Status::ArgumentCountMismatch:
                fmt::print(stderr, "Requires {} arguments\n", o.type->param_types.size());
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
/// local variable declarations register a destructor call. For the sake
/// of brevity, we shall refer to such expressions as *protected*.
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
/// variable. Even without the `print` call, the destructor of `x` would
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
/// executed, whenever the *rest* of the scope after that expression
/// is exited; this includes cases where the same scope is reëntered
/// further up before the protected expression.
///
/// These examples show that not all branches are created equal; in
/// fact, we can categorise branches according to two properties: a
/// branch can involve upwards and downwards movement, sometimes even
/// both, but it can only move forwards or backwards.
///
/// To elaborate, a branch involves a source (a GotoExpr) and a target
/// (a LabelExpr); either the source or target are in the same scope
/// or they aren’t. If they are, then the branch moves either forwards
/// or backwards, and there is no upwards or downwards movement.
///
/// If they aren’t, then the source and the target have some nca (nearest
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
/// In order to handle such branches in a same manner, it helps to
/// operate based on the assumption that control flow never actually
/// ‘jumps’ around randomly, but instead always flows from a parent
/// scope to a direct child or vice versa: in this case, we do not
/// simply jump from the source to the target, but rather, we move
/// up one scope from the source, to the NCA block, then backwards
/// in that block to the block containing the target, and then down
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
/// combine some of the logic. First, note that a cross jump needs to move
/// from the source up to the NCA, then forward or backward in the NCA,
/// and then down from the NCA to the target. Thus, a cross jump can be
/// decomposed into combinations of the other three.
///
/// Furthermore, upwards jumps involve leaving scopes; this means that we
/// conceptually move to the beginning of a scope, and then up to its parent;
/// downwards scopes do something similar, except that they move from the
/// start of a scope to somewhere in the middle of it in order to enter a nested
/// scope.
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
/// While rather deranged, it shows that both protected expressions and scopes
/// need not occur directly at the top-level of another scope, but may instead
/// be nested arbitrarily in other expressions; this means that we can’t simply
/// iterate over each expression in the parent scope to check if e.g. `int y`
/// precedes or follows `{ goto back; }`.
///
/// To enable this, we introduce the concept of a *full expression* (FE): an
/// expression is a full expression, iff its parent is a block expression
/// (= scope). The *parent full expression* `FE(e)` of an expression `e` is the
/// closest full expression that contains `e` (or `e` itself if it is a full
/// expression).
///
/// This allows us to compare expressions with one another wrt where they
/// occur in a scope by examining the relative order of their parent full
/// expressions: here, the FE containing `back:` is second in its parent,
/// and the FE containing `{ goto back; }` third. Thus, this is an example
/// of backwards movement in the same scope.
///
/// Full expressions also allow us to deal with nested protected expressions:
/// in this case, the jump would unwind over the *protected subexpression*
/// `int y`, which means that we need to generate a destructor call here before
/// performing the jump.
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
/// A formal elaboration of the algorithm is found below. The entry point for the
/// algorithm is Validate-Jump. It makes use of two procedures: Unwind, which unwinds
/// from an expression to some parent scope, and Unwind-Local, which unwinds within
/// a scope.
///
/// ALGORITHM Validate-Jump (in Goto, in Label)
///     contract Goto and Label must be full expressions.
///     where Goto  := the branch transferring control flow.
///           Label := the target that we are branching to.
///
///     1. Let Source be the parent scope of the full expression that contains Goto; let Target
///        be the parent scope of the full expression that contains Label; let NCA be the nearest
///        common ancestor scope of Source and Target.
///
///     2. If NCA != Target (that is, the target is a child of the NCA), then the jump involves
///        downward movement. Otherwise, go to step 3. Invoke Unwind(Label, NCA). If the returned
///        list is not empty (that is, if we crossed any protected expressions), the program is
///        ill-formed.
///
///     3. Let To-Unwind = Unwind(Goto, NCA).
///
///     4. Let FE(Goto) and FE(Label) be the parent full expressions of Goto and Label, respectively,
///        in NCA. If FE(Goto) < FE(Label), let First be FE(Goto) and Second be FE(Label); otherwise,
///        let First be FE(Label) and Second be FE(Goto).
///
///     5. Let Rest = Unwind-Local(Second, First). If FE(Goto) < FE(Label), and Rest is not empty, the
///        program is ill-formed. Append Rest to To-Unwind.
///
///     6. During codegen, when emitting the branch, first handle all expressions in To-Unwind (e.g.
///        invoke destructors of variables and execute deferred expressions).
///
/// PROCEDURE Unwind (in E, in To) yields Expr[]
///     contract To is a proper ancestor of the parent scope of E.
///     where E  := the *full expression* we are unwinding from;
///           To := the scope (= block) we are unwinding to.
///
///     1. Let S be the parent scope of E. Let To-Unwind be an empty Expr[].
///
///     2. If S = To, return To-Unwind.
///
///     3. Let First be the first full expression of S. Append the result of invoking
///        Unwind-Local(E, First) to To-Unwind.
///
///     4. Set E to the parent full expression of S, and S to the parent scope of S. Go
///        to step 2.
///
/// PROCEDURE Unwind-Local (in FE, in To) yields Expr[]
///     contract FE and To are in the same scope, and To precedes FE.
///     where FE := the *full expression* we are unwinding from;
///           To := the *full expression* we are unwinding to.
///
///     1. Let To-Unwind be an empty Expr[].
///
///     2. Add any protected subexpressions of FE to To-Unwind, in reverse tree order.
///
///     3. If FE = To, return To-Unwind.
///
///     4. Go to step 2.
///
/// Unwind-Local as above, but stores expressions into a vector instead
/// of creating a new one or issues an error if the vector is nullptr.
///
auto src::Sema::UnwindLocal(UnwindContext ctx, BlockExpr* S, Expr* FE, Expr* To) -> bool {
    auto FEIter = rgs::find(S->exprs, FE);
    auto ToIter = rgs::find(S->exprs, To);
    Assert(FEIter != S->exprs.end());
    Assert(ToIter != S->exprs.end());

    /// We want to include the contents of the expression that we’re unwinding
    /// from only if we’re actually unwinding, in which case the UnwindContext
    /// will not be an expression.
    if (not ctx.is<Expr*>()) ++FEIter;

    /// Handle protected subexpressions.
    for (auto E : rgs::subrange(ToIter, FEIter) | vws::reverse) {
        if (auto expr = ctx.dyn_cast<Expr*>()) {
            /// TODO: `protected_children` is horrible jank; get rid of it somehow.
            if (not E->protected_children.empty()) {
                Error(expr, "Jump is ill-formed");
                Diag::Note(
                    mod->context,
                    E->protected_children[0]->location,
                    "Because it would bypass {} here",
                    isa<DeferExpr>(E->protected_children[0])
                        ? "deferred expression"s
                        : "variable declaration"s
                );
                return false;
            }
        } else {
            auto prot = ctx.get<SmallVectorImpl<Expr*>*>();
            for (auto P : E->protected_children | vws::reverse) prot->push_back(P);
        }
    }
    return true;
}

/// \brief Unwind from E in S up to, but not including, To, if S != To.
///
/// This is `Unwind` as described in the big comment above, with the same
/// modifications as `UnwindLocal`.
///
/// \return The parent full expression of E in To, or nullptr if there was an error.
auto src::Sema::Unwind(UnwindContext prot, BlockExpr* S, Expr* E, BlockExpr* To) -> Expr* {
    /// 1/2.
    while (S != To) {
        /// 3.
        if (
            not S->exprs.empty() and
            not UnwindLocal(prot, S, E, S->exprs.front())
        ) return nullptr;

        /// 4.
        E = S->parent_full_expression;
        S = S->parent;
    }

    return E;
}

/// Unwind from uw in S up to and including To.
void src::Sema::UnwindUpTo(BlockExpr* S, BlockExpr* To, UnwindExpr* uw) {
    auto E = uw->parent_full_expression;
    for (;;) {
        if (
            not S->exprs.empty() and
            not UnwindLocal(&uw->unwind, S, E, S->exprs.front())
        ) return;

        if (S == To) break;
        E = S->parent_full_expression;
        S = S->parent;
    }
}

void src::Sema::ValidateDirectBr(GotoExpr* g, BlockExpr* source) {
    if (not g->sema.ok) return;
    auto label = g->target;
    auto target = label->parent;

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

    /// 1.
    BlockExpr* nca = BlockExpr::NCAInFunction(source, target);
    Assert(nca, "Goto and label blocks must have a common ancestor");

    /// 2.
    Expr* label_nca = label;
    if (nca != target) {
        label_nca = Unwind(g, label->parent, label, nca);
        if (not label_nca) return;
    }

    /// 3.
    Expr* goto_nca = Unwind(&g->unwind, source, g, nca);

    /// 4/5.
    auto goto_it = rgs::find(nca->exprs, goto_nca);
    auto label_it = rgs::find(nca->exprs, label_nca);
    Assert(goto_it != nca->exprs.end());
    Assert(label_it != nca->exprs.end());
    auto unwind_to = std::min(goto_it, label_it);
    auto unwind_from = std::max(goto_it, label_it);
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
    Expr* target
) -> ConstructExpr* {
    /// Validate initialisers.
    if (not rgs::all_of(init_args, [&](auto*& e) { return Analyse(e); }))
        return nullptr;

    /// Helper to print the argument types.
    auto FormatArgTypes = [&](auto args) {
        utils::Colours C{true};
        return fmt::join(
            vws::transform(args, [&](auto* e) { return e->type.str(mod->context->use_colours, true); }),
            fmt::format("{}, ", C(utils::Colour::Red))
        );
    };

    /// Helper to emit an error that construction is not possible.
    auto InvalidArgs = [&] -> ConstructExpr* {
        Error(
            loc,
            "Cannot construct '{}' from arguments {}",
            type,
            FormatArgTypes(init_args)
        );

        return nullptr;
    };

    /// Helper to emit a trivial copy.
    auto TrivialCopy = [&](Type ty) -> ConstructExpr* {
        if (not Convert(init_args[0], ty)) {
            Error(
                init_args[0],
                "Cannot convert '{}' to '{}'",
                init_args[0]->type,
                ty
            );
            return nullptr;
        }

        return ConstructExpr::CreateTrivialCopy(mod, init_args[0]);
    };

    /// Initialisation is dependent on the type.
    switch (auto ty = type.desugared; ty->kind) {
        default: Unreachable("Invalid type for variable declaration");
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            Unreachable("Should have been desugared");

        /// These take zero or one argument.
        case Expr::Kind::BuiltinType:
        case Expr::Kind::IntType: {
            if (init_args.empty()) return ConstructExpr::CreateZeroinit(mod);
            if (init_args.size() == 1) return TrivialCopy(type);
            return InvalidArgs();
        }

        /// References must always be initialised.
        case Expr::Kind::ReferenceType: {
            if (init_args.empty()) {
                Error(loc, "Reference must be initialised");
                return nullptr;
            }

            if (init_args.size() == 1) return TrivialCopy(type);
            return InvalidArgs();
        }

        /// This is the complicated one.
        case Expr::Kind::StructType: {
            auto s = cast<StructType>(ty);

            /// If there are no arguments, and no constructor, zero-initialise the struct.
            if (init_args.empty() and s->initialisers.empty()) return ConstructExpr::CreateZeroinit(mod);

            /// Otherwise, perform overload resolution.
            auto ctor = PerformOverloadResolution(loc, s->initialisers, init_args, true);
            if (not ctor) return nullptr;
            return ConstructExpr::CreateInitialiserCall(mod, ctor, init_args);
        }

        case Expr::Kind::ArrayType: {
            auto [base, total_size, _] = ty.strip_arrays;
            auto no_args = init_args.empty();
            auto arr = no_args ? nullptr : dyn_cast<ArrayLitExpr>(init_args[0]->ignore_parens);

            /// No args or a single empty array literal.
            if (no_args or (arr and arr->elements.empty())) {
                /// An array of trivial types is itself trivial.
                if (base.trivial) return ConstructExpr::CreateZeroinit(mod);

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
            if (arr) {
                auto array_type = cast<ArrayType>(ty);
                auto size = array_type->dimension().getZExtValue();
                auto elem = array_type->elem.desugared;

                /// Array literal must not be larger than the array type.
                if (size < arr->elements.size()) {
                    Error(
                        loc,
                        "Cannot initialise array '{}' from larger literal containing '{}' elements",
                        type,
                        size
                    );
                    return nullptr;
                }

                /// Construct each element.
                SmallVector<Expr*, 16> ctors;
                for (auto e : arr->elements)
                    if (auto ctor = Construct(e->location, elem, e))
                        ctors.push_back(ctor);

                /// If the array literal is too small, that’s fine, so long
                /// as the element type is default-constructible.
                if (auto rem = size - arr->elements.size()) {
                    rem *= elem.strip_arrays.total_dimension;
                    if (base.trivial) {
                        ctors.push_back(ConstructExpr::CreateArrayZeroinit(mod, rem));
                    } else if (auto ctor = base.default_constructor) {
                        ctors.push_back(ConstructExpr::CreateArrayInitialiserCall(mod, {}, ctor, rem));
                    } else {
                        Error(
                            loc,
                            "Cannot create array '{}' from literal containing '{}' "
                            "elements as '{}' is not default-constructible",
                            type,
                            arr->elements.size(),
                            base
                        );
                        return nullptr;
                    }
                }

                /// Create a constructor that calls all of these constructors.
                return ConstructExpr::CreateArrayListInit(mod, ctors);
            }

            /// Otherwise, attempt to broadcast a single value. If the
            /// value is simply convertible to the target type, then
            /// just copy it.
            if (base.trivial) {
                ConversionSequence seq;
                if (not TryConvert(seq, init_args[0], base)) {
                    Error(
                        loc,
                        "Cannot initialise an object of type '{}' with a value of type '{}'",
                        type,
                        init_args[0]->type
                    );
                    return nullptr;
                }

                ApplyConversionSequence(init_args[0], std::move(seq));
                return ConstructExpr::CreateArrayBroadcast(mod, init_args[0], total_size);
            }

            /// If not, see if there is a constructor with that type.
            if (auto s = dyn_cast<StructType>(base)) {
                auto ctor = PerformOverloadResolution(loc, s->initialisers, init_args, true);
                if (not ctor) return nullptr;
                return ConstructExpr::CreateArrayInitialiserCall(mod, init_args, ctor, total_size);
            }

            /// Otherwise, we can’t do anything here.
            return InvalidArgs();
        }

        case Expr::Kind::SliceType: {
            if (init_args.empty()) return ConstructExpr::CreateZeroinit(mod);
            if (init_args.size() == 1) return TrivialCopy(type);

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
                if (type == init_args[0]->type)
                    return ConstructExpr::CreateTrivialCopy(mod, init_args[0]);

                /// Otherwise, attempt to convert it to the wrapped type.
                Optionals.Activate(target);
                return TrivialCopy(opt->elem);
            }

            return InvalidArgs();
        }

        case Expr::Kind::ProcType: Todo();
        case Expr::Kind::ScopedPointerType: Todo();
        case Expr::Kind::ClosureType: Todo();
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
    defer { e->sema.set_done(); };
    switch (e->kind) {
        /// Marked as type checked in the constructor.
        case Expr::Kind::BuiltinType:
        case Expr::Kind::EmptyExpr:
        case Expr::Kind::IntType:
        case Expr::Kind::ModuleRefExpr:
        case Expr::Kind::Nil:
        case Expr::Kind::OpaqueType:
        case Expr::Kind::ScopedType:
        case Expr::Kind::SugaredType:
            Unreachable();

        /// Only generated by sema and always type checked.
        case Expr::Kind::ConstExpr:
        case Expr::Kind::ConstructExpr:
        case Expr::Kind::ImplicitThisExpr:
            Unreachable();

        /// Handled by the code that type checks structs.
        case Expr::Kind::FieldDecl:
            Unreachable("FieldDecl should be handled when analysing StructType");

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
        case Expr::Kind::BoolLiteralExpr:
            cast<BoolLitExpr>(e)->stored_type = BuiltinType::Bool(mod, e->location);
            break;

        /// Integers are of type int.
        case Expr::Kind::IntegerLiteralExpr: {
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

        /// Array literals are weird because we can’t really do much with
        /// them until we get to the context that they’re used in.
        case Expr::Kind::ArrayLiteralExpr: {
            auto a = cast<ArrayLitExpr>(e);
            for (auto& elem : a->elements)
                if (not Analyse(elem))
                    return e->sema.set_errored();
        } break;

        /// String literals are u8 slices.
        case Expr::Kind::StringLiteralExpr: {
            auto str = cast<StrLitExpr>(e);
            auto loc = str->location;

            /// Unlike in C++, string literals are *not* lvalues; rather a
            /// new string slice is constructed every time a string literal
            /// is used.
            str->stored_type = new (mod) SliceType(Type::I8, loc);
            AnalyseAsType(str->stored_type);
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

            for (auto& param : type->param_types) {
                if (not AnalyseAsType(param) or not MakeDeclType(param)) e->sema.set_errored();
                else if (param == Type::Unknown) {
                    e->sema.set_errored();
                    Error(param.ptr, "Type inference is not permitted here");
                }
            }

            if (not AnalyseAsType(type->ret_type)) e->sema.set_errored();
        } break;

        /// Structures.
        case Expr::Kind::StructType: {
            auto s = cast<StructType>(e);
            usz padding_count = 0;
            u32 index = 0;

            /// Create a padding field at the given location to serve as padding.
            const auto CreatePaddingField = [&](Size padding, auto insert_before) {
                auto ty = ArrayType::GetByteArray(mod, isz(padding.bytes()));
                auto f = new (mod) FieldDecl(
                    fmt::format("#padding{}", padding_count++),
                    ty,
                    {},
                    s->stored_size,
                    index++,
                    true
                );

                s->all_fields.insert(insert_before, f);
                s->stored_size += padding;
            };

            /// Helper to align a field to its alignment.
            const auto AlignField = [&](Align alignment, usz& it) {
                /// Check if we need to insert padding.
                auto pad = s->stored_size.align_padding(alignment);
                if (pad.bits() == 0) return;
                CreatePaddingField(pad, s->all_fields.begin() + it);

                /// Move iterator forward past the padding to the actual element.
                it++;
            };

            /// Members must all be valid decls.
            ///
            /// We iterate using an index to avoid iterator invalidation, as we
            /// may have to insert padding fields as we go.
            for (usz i = 0; i < s->all_fields.size(); i++) {
                if (
                    auto& f = s->all_fields[i];
                    not AnalyseAsType(f->stored_type) or
                    not MakeDeclType(f->stored_type)
                ) {
                    e->sema.set_errored();
                    continue;
                }

                /// Set offset of each field and determine the size and alignment
                /// of the struct.
                auto align = s->all_fields[i]->type.align(ctx);
                AlignField(align, i);
                s->all_fields[i]->offset = s->stored_size;
                s->all_fields[i]->index = index++;
                s->stored_alignment = std::max(s->stored_alignment, align);
                s->stored_size += s->all_fields[i]->type.size(ctx);
            }

            /// Size must be a multiple of the alignment.
            auto pad = s->stored_size.align_padding(s->stored_alignment);
            if (pad.bits() != 0) CreatePaddingField(pad, s->all_fields.end());

            /// Analyse initialiser types first in case they call each other since
            /// we will have to perform overload resolution in that case.
            for (auto& i : s->initialisers) {
                auto ty = cast<ProcType>(i->type);
                ty->smp_parent = s;
                ty->smp_kind = SpecialMemberKind::Constructor;
                if (not AnalyseProcedureType(i)) e->sema.set_errored();
            }

            /// At this point, the type is complete.
            if (not e->sema.errored) e->sema.set_done();

            /// Member functions are isolated from everything above.
            tempset with_stack = decltype(with_stack){};

            /// Analyse deleter, if there is one.
            if (s->deleter) {
                auto ty = cast<ProcType>(s->deleter->type);
                ty->smp_parent = s;
                ty->smp_kind = SpecialMemberKind::Destructor;

                /// Analyse the destructor first since initialisers may call it.
                with_stack.push_back(new (mod) ImplicitThisExpr(s->deleter, s, {}));
                Expr* p = s->deleter;
                if (Analyse(p)) {
                    Assert(p == s->deleter, "Analysis must not reassign procedure");
                    Assert(s->deleter->ret_type == Type::Void, "Deleter must return void");
                }
            }

            /// Analyse initialisers.
            for (auto& i : s->initialisers) {
                with_stack.clear();
                with_stack.push_back(new (mod) ImplicitThisExpr(i, s, {}));
                Expr* p = i;
                if (Analyse(p)) {
                    Assert(p == i, "Analysis must not reassign procedure");
                    Assert(i->ret_type == Type::Void, "Initialiser must return void");
                }
            }

            /// Analyse member functions.
            for (auto& [_, procs] : s->member_procs) {
                for (auto m : procs) {
                    Expr* p = m;
                    if (Analyse(p)) Assert(p == m, "Analysis must not reassign procedure");
                }
            }
        } break;

        /// Defer expressions have nothing to typecheck really, so
        /// we just check the operand and leave it at that. Even
        /// nested `defer defer` expressions, albeit degenerate, are
        /// accepted.
        case Expr::Kind::DeferExpr: {
            protected_subexpressions.push_back(e);
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
            if (Analyse(a->cond) and EnsureCondition(a->cond)) {
                /// If this asserts that an optional is not nil, mark it as active.
                if (auto o = Optionals.MatchNilTest(a->cond)) Optionals.Activate(o);
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
                mod->strtab.intern(a->cond->location.text(mod->context)),
                a->location
            );

            a->file_str = new (mod) StrLitExpr(
                mod->strtab.intern(mod->context->file(a->location.file_id)->path().string()),
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
            OptionalState::ScopeGuard _{*this};
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
                tempset protected_subexpressions = SmallVector<Expr*, 1>{};
                tempset needs_link_to_full_expr = SmallVector<Expr*>{};

                if (not Analyse(expr)) {
                    if (i == last) e->sema.set_errored();
                    continue;
                }

                if (i == last) {
                    b->stored_type = expr->type;
                    b->is_lvalue = expr->is_lvalue;
                }

                /// Update links.
                expr->protected_children = std::move(protected_subexpressions);
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
        case Expr::Kind::InvokeExpr: return AnalyseInvoke(e);

        /// Builtins are handled out of line.
        case Expr::Kind::InvokeBuiltinExpr: return AnalyseInvokeBuiltin(e);

        /// Cast expression.
        case Expr::Kind::CastExpr: {
            auto m = cast<CastExpr>(e);
            Analyse(m->operand);
            switch (m->cast_kind) {
                /// Only generated by sema. Converts an lvalue to an rvalue.
                case CastKind::LValueToRValue:
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
                    auto fields = s->fields();
                    auto f = rgs::find(fields, m->member, [](auto& f) { return f->name; });
                    if (f != fields.end()) {
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

            /// Currently, we only allow looking up names in a module.
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

            return Error(sa, "LHS of operator '::' must be a module", sa->object->type);
        }

        /// Perform name lookup in scope.
        case Expr::Kind::DeclRefExpr:
            AnalyseDeclRefExpr<false>(e);
            break;

        /// Determine the static chain offset from a variable reference to
        /// its declaration. Type and lvalueness is already set by the ctor.
        case Expr::Kind::LocalRefExpr: {
            auto var = cast<LocalRefExpr>(e);
            if (var->parent != var->decl->parent) {
                /// Synthesised variables (e.g. loop variables) should never be captured.
                if (not var->decl->is_legal_to_capture) return Error(
                    var,
                    "Cannot capture synthesised variable '{}'",
                    var->decl->name
                );

                var->decl->set_captured();
            }
        } break;

        /// Parameter declaration.
        case Expr::Kind::ParamDecl: {
            protected_subexpressions.push_back(e);
            auto var = cast<ParamDecl>(e);
            if (not AnalyseAsType(var->stored_type)) return e->sema.set_errored();
            if (not MakeDeclType(var->stored_type)) return e->sema.set_errored();
            var->ctor = ConstructExpr::CreateMoveParam(mod);

            /// Add the variable to the current scope.
            if (not var->sema.errored) {
                if (var->with) with_stack.push_back(var);
                curr_scope->declare(var->name, var);
                var->is_lvalue = true;
            }
        } break;

        /// Variable declaration.
        case Expr::Kind::LocalDecl: {
            protected_subexpressions.push_back(e);
            auto var = cast<LocalDecl>(e);
            if (not AnalyseAsType(var->stored_type)) return e->sema.set_errored();

            /// Parameters should be ParamDecls instead.
            if (var->local_kind == LocalKind::Parameter) Unreachable();

            /// Synthesised variables are just lvalues that point somewhere else.
            else if (var->local_kind == LocalKind::Synthesised) {
                if (not MakeDeclType(var->stored_type)) return e->sema.set_errored();
                var->ctor = ConstructExpr::CreateUninitialised(mod);
            }

            /// This is a regular variable.
            else {
                /// Infer the type from the initialiser.
                if (var->stored_type == Type::Unknown) {
                    if (var->init_args.empty()) return Error(var, "Type inference requires an initialiser");
                    if (var->init_args.size() != 1) Todo();
                    if (not Analyse(var->init_args[0])) return e->sema.set_errored();
                    InsertLValueToRValueConversion(var->init_args[0]);
                    var->stored_type = var->init_args[0]->type;
                }

                /// Type must be valid for a variable.
                if (not MakeDeclType(var->stored_type)) return e->sema.set_errored();

                /// If the initialiser is an array literal, set this as
                /// the result object of that literal.
                if (not var->init_args.empty()) {
                    auto arr = dyn_cast<ArrayLitExpr>(var->init_args[0]);
                    if (arr) arr->result_object = var;
                }

                /// Type must be constructible from the initialiser args.
                var->ctor = Construct(var->location, var->type, var->init_args, var);
                if (not var->ctor) e->sema.set_errored();
                var->init_args.clear();
            }

            /// Add the variable to the current scope.
            if (not var->sema.errored) {
                curr_scope->declare(var->name, var);
                var->is_lvalue = true;
            }
        } break;

        /// If expressions.
        case Expr::Kind::IfExpr: {
            auto i = cast<IfExpr>(e);

            /// If the condition has an error, the type of the if expression
            /// itself can still be determined as it is independent of the
            /// condition.
            if (Analyse(i->cond)) EnsureCondition(i->cond);

            /// Analyse the branches.
            ///
            /// If the condition tests whether an optional is not nil, set
            /// the active state of the optional to true in the branch where
            /// it isn’t.
            if (auto o = Optionals.MatchNilTest(i->cond)) {
                {
                    OptionalState::ActivationGuard _{*this, o};
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
                    OptionalState::ActivationGuard _{*this, local->decl};
                    if (i->else_ and not Analyse(i->else_)) return e->sema.set_errored();
                }

                if (not Analyse(i->then)) return e->sema.set_errored();
                if (i->then->type == Type::NoReturn) Optionals.Activate(local->decl);
            } // clang-format on

            /// Otherwise, there is nothing to infer.
            else if (not Analyse(i->then) or (i->else_ and not Analyse(i->else_)))
                return e->sema.set_errored();

            /// Type is void, unless one of the conditions below applies.
            i->stored_type = Type::Void;

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

            /// Make sure the range is something we can iterate over; currently,
            /// that’s only arrays and slices.
            if (not Analyse(f->range)) return e->sema.set_errored();
            UnwrapInPlace(f->range, true);
            if (not isa<ArrayType, SliceType>(f->range->type.strip_refs_and_pointers)) return Error(
                f->range,
                "Type '{}' is not iterable",
                f->range->type
            );

            /// Slices can just be loaded whole.
            /// TODO: Figure out what to do in general w/ rvalue arrays, objects, etc
            ///       wrt variables, iterations, subscripting, and member access—preferably
            ///       in a way that doesn’t require the backend to deal w/ all of that again.
            if (isa<SliceType>(f->range->type)) InsertLValueToRValueConversion(f->range);
            else if (not f->range->is_lvalue) Diag::ICE(
                mod->context,
                f->range->location,
                "Sorry, iterating over rvalue arrays is currently not supported"
            );

            /// The loop variable is an lvalue of the element type of the range.
            auto s = cast<SingleElementTypeBase>(f->range->type);
            f->iter->stored_type = s->elem;

            /// Now check the loop variable. We can’t analyse the body if this fails
            /// since it is probably going to use it.
            Expr* expr = f->iter;
            if (not Analyse(expr)) return e->sema.set_errored();
            Assert(expr == f->iter, "Must not wrap loop variable");

            /// Finally, check the body.
            loop_stack.push_back(f);
            defer { loop_stack.pop_back(); };
            expr = f->body;
            Analyse(expr);
            Assert(expr == f->body, "Body of for-in expression must be a block");
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

            /// Handle arrays and slices.
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

            Error(
                s,
                "Cannot perform subscripting on type '{}'",
                s->object->type
            );
        } break;

        /// Binary operators are complicated.
        case Expr::Kind::BinaryExpr: {
            auto b = cast<BinaryExpr>(e);
            if (not Analyse(b->lhs) or not Analyse(b->rhs))
                return e->sema.set_errored();
            switch (b->op) {
                default: Unreachable("Invalid binary operator");

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

                    /// Both types must be integers.
                    if (
                        not b->lhs->type.is_int(false) or
                        not b->rhs->type.is_int(false)
                    ) return Error( //
                        b,
                        "Operands of '{}' must be integers, but got '{}' and '{}'",
                        Spelling(b->op),
                        b->lhs->type,
                        b->rhs->type
                    );

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
                        Spelling(b->op),
                        b->lhs->type,
                        b->rhs->type
                    );

                    /// The type of a comparison is bool.
                    b->stored_type = BuiltinType::Bool(mod, b->location);
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
                case Tk::ShiftRightLogicalEq:
                case Tk::Assign: {
                    /// These operators never perform reference reassignment, which
                    /// means the LHS must not be of reference type.
                    UnwrapInPlace(b->lhs, true);
                    if (not b->lhs->is_lvalue) return Error(
                        b,
                        "Left-hand side of `=` must be an lvalue"
                    );

                    /// Compound assignment.
                    if (b->op != Tk::Assign) {
                        InsertLValueToRValueConversion(b->rhs);

                        /// Both types must be integers.
                        if (
                            not b->lhs->type.is_int(false) or
                            not b->rhs->type.is_int(false)
                        ) return Error( //
                            b,
                            "Operands of '{}' must be integers, but got '{}' and '{}'",
                            Spelling(b->op),
                            b->lhs->type,
                            b->rhs->type
                        );
                    }

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
                        Optionals.Activate(b->lhs);

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

bool src::Sema::AnalyseAsType(Type& e) {
    Analyse(e.ptr);
    if (e->sema.ok and not isa<TypeBase>(e.ptr)) Error(e->location, "Not a type");
    return e->sema.ok;
}

template <bool allow_undefined>
bool src::Sema::AnalyseDeclRefExpr(Expr*& e) {
    auto d = cast<DeclRefExpr>(e);

    /// Resolve a DeclRefExpr in place.
    auto ResolveInPlace = [&](Expr* decl) {
        d->stored_type = decl->type;
        d->is_lvalue = false;
    };

    /// Some DeclRefExprs may already be resolved to a node. If this one
    /// isn’t, find the nearest declaration with the given name.
    if (not d->decl) {
        auto* const decls = [&] -> BlockExpr::Symbols* {
            /// Try to find a declared symbol with that name.
            if (auto syms = d->scope->find(d->name, false)) return syms;

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

    /// Integer-to-integer conversions are fine.
    if (c->operand->type.is_int(true) and e->type.is_int(true)) {
        InsertLValueToRValueConversion(c->operand);
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

bool src::Sema::AnalyseInvoke(Expr*& e) {
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
        auto callee = PerformOverloadResolution(invoke->location, o->overloads, invoke->args, true);
        if (not callee) return e->sema.set_errored();
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

        /// Callee must be an rvalue.
        InsertLValueToRValueConversion(callee);

        /// Analyse the arguments.
        for (auto& arg : invoke->args)
            if (not Analyse(arg))
                e->sema.set_errored();

        /// Make sure the types match.
        for (usz i = 0; i < invoke->args.size(); i++) {
            if (i < ptype->param_types.size()) {
                if (not Convert(invoke->args[i], ptype->param_types[i])) {
                    Error(
                        invoke->args[i],
                        "Argument type '{}' is not convertible to parameter type '{}'",
                        invoke->args[i]->type,
                        ptype->param_types[i]
                    );
                    e->sema.set_errored();
                }
            }

            /// Variadic arguments only undergo lvalue-to-rvalue conversion.
            else { InsertLValueToRValueConversion(invoke->args[i]); }
        }

        /// Make sure there are as many arguments as parameters.
        if (
            invoke->args.size() < ptype->param_types.size() or
            (invoke->args.size() != ptype->param_types.size() and not ptype->variadic)
        ) {
            Error(
                e,
                "Expected {} arguments, but got {}",
                ptype->param_types.size(),
                invoke->args.size()
            );
        }

        /// The type of the expression is the return type of the
        /// callee. Invoke expressions are never lvalues.
        invoke->stored_type = ptype->ret_type;
        return not e->sema.errored;
    };

    /// Handle member function calls.
    if (invoke->callee->type == Type::MemberProc) {
        auto m = cast<MemberAccessExpr>(invoke->callee);
        if (auto o = dyn_cast<OverloadSetExpr>(m->field)) return ResolveOverloadSet(o, m->field);
        return PerformSimpleCall(m->field);
    }

    /// Perform overload resolution, if need be.
    if (auto o = dyn_cast<OverloadSetExpr>(invoke->callee))
        return ResolveOverloadSet(o, invoke->callee);

    /// If the callee is of function type, and not a type itself,
    /// then this is a function call.
    if (isa<ProcType, ClosureType>(invoke->callee->type) and not isa<TypeBase>(invoke->callee))
        return PerformSimpleCall(invoke->callee);

    /// Otherwise, if the callee is a type, then this is a declaration.
    if (isa<TypeBase>(invoke->callee)) {
        /// TODO: naked: decl. parens: literal!
        Type ty{invoke->callee};
        if (not MakeDeclType(ty)) e->sema.set_errored();

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
                curr_proc,
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

    /// Otherwise, no idea what this is supposed to be.
    else {
        e->sema.set_errored();
        return Error(invoke->callee, "Expected procedure or type");
    }
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

            /// Mark var as destroyed.
            cast<LocalRefExpr>(invoke->args[0])->decl->set_deleted_or_moved();

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
    if (not proc->body) return;
    Assert(not proc->sema.analysed);
    tempset curr_scope = proc->body;
    tempset unwind_entries = decltype(unwind_entries){};

    /// Protected subexpressions never cross a procedure boundary.
    /// FIXME: Shouldn’t this be a tempset too?
    /// FIXME: Comment above is irrelevant since this system will be yeeted soon.
    defer { protected_subexpressions.clear(); };

    /// Assign a name to the procedure if it doesn’t have one.
    if (proc->name.empty())
        proc->name = fmt::format("__srcc_lambda_{}", lambda_counter++);

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

        /// Regular module.
        for (auto& p : mod->context->import_paths) {
            auto mod_path = p / i.linkage_name;
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
        for (auto p = f->parent; p; p = p->parent)
            if (not p->captured_locals.empty())
                return true;
        return false;
    };

    /// Only after analysing all functions can we determine whether
    /// any of them take static chain pointers or not.
    for (auto f : mod->functions)
        if (TakesChainPointer(f))
            cast<ProcType>(f->type)->static_chain_parent = f->parent;
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
