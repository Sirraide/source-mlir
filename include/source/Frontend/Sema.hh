#ifndef SOURCE_FRONTEND_SEMA_HH
#define SOURCE_FRONTEND_SEMA_HH

#include <source/Frontend/AST.hh>
#include <source/Support/Buffer.hh>

namespace src {
template <typename T>
struct make_formattable {
    using type = T;
};

template <typename T>
requires std::is_reference_v<T>
struct make_formattable<T> {
    using type = make_formattable<std::remove_reference_t<T>>::type;
};

template <std::derived_from<Expr> T>
struct make_formattable<T*> {
    using type = std::string;
};

template <>
struct make_formattable<Type> {
    using type = std::string;
};

template <>
struct make_formattable<const Type> {
    using type = std::string;
};

template <typename T>
using make_formattable_t = make_formattable<T>::type;

class Sema {
    Module* mod{};
    ProcDecl* curr_proc{};
    BlockExpr* curr_scope{};

    /// Loops that we’re currently analysing.
    SmallVector<Loop*, 10> loop_stack;

    /// The defer expression whose contents we are currently analysing.
    DeferExpr* curr_defer{};

    /// Expressions that require unwinding once everything else has been checked.
    struct Unwind {
        BlockExpr* in_scope;
        UnwindExpr* expr;

        /// Unused if this is a goto.
        BlockExpr* to_scope{};
    };

    /// Conversion sequence to convert from one type to another.
    ///
    /// Each conversion in the sequence entails performing one
    /// of the following actions; at most one CastExpr may be
    /// created for each conversion sequence.
    ///
    ///     - Creating a CastExpr.
    ///     - Creating a ConstExpr with a certain value.
    ///     - Calling a constructor.
    struct ConversionSequence {
        struct BuildCast {
            CastKind kind;
            Type to;
        };

        struct BuildConstExpr {
            /// Empty so we don’t store 27 EvalResults for a large
            /// conversion sequence; since we can only construct
            /// one ConstExpr anyway, a ConversionSequence only
            /// needs to store a single EvalResult.
        };

        struct CallConstructor {
            ProcDecl* ctor;
            Buffer<Expr*> args;
        };

        /// Convert an overload set to a DeclRefExpr to a procedure.
        struct OverloadSetToProc {
            ProcDecl* proc;
        };

        using Entry = std::variant<BuildCast, BuildConstExpr, CallConstructor, OverloadSetToProc>;
        SmallVector<Entry> entries;
        std::optional<EvalResult> constant;
        int score{};

        static void ApplyCast(Sema& s, Expr*& e, CastKind kind, Type to);
        static void ApplyConstExpr(Sema& s, Expr*& e, EvalResult res);
        static void ApplyConstructor(Sema& s, Expr*& e, ProcDecl* ctor, ArrayRef<Expr*> args);
        static void ApplyOverloadSetToProc(Sema& s, Expr*& e, ProcDecl* proc);
    };

    /// Helper for checking conversions.
    template <bool perform_conversion>
    struct ConversionContext {
        Sema& S;
        ConversionSequence* seq;
        Expr** e;
        int score{};
        bool has_expr{};

        ConversionContext(Sema& s, ConversionSequence& seq, Expr** e = nullptr)
        requires (not perform_conversion)
            : S(s), seq(&seq), e(e) {
            if (e) has_expr = true;
        }

        ConversionContext(Sema& s, Expr*& e)
        requires perform_conversion
            : S(s), seq(nullptr), e(&e) {
            has_expr = true;
        }

        /// Get the current expression; this can only be valid if the conversion
        /// sequence is empty or if the last entry is a ConstExpr; otherwise, the
        /// expression has already been converted to something else; note that there
        /// may also be no expression at all in some cases if we’re just checking
        /// whether two types are convertible with one another.
        readonly(Expr*, expr, return has_expr ? *e : nullptr);

        /// Whether the current expression is an lvalue.
        readonly(bool, is_lvalue, return expr and expr->is_lvalue);

        /// Emit a cast.
        Type cast(CastKind k, Type to);

        /// Emit lvalue-to-rvalue conversion.
        void lvalue_to_rvalue();

        /// Emit a conversion from an overload set to a procedure.
        Type overload_set_to_proc(ProcDecl* proc);

        /// Replace the expression with a constant.
        Type replace_with_constant(EvalResult&& res);

        /// Attempt to evaluate this as a constant expression.
        bool try_evaluate(EvalResult& out);
    };

    /// Overload candidate.
    struct Candidate {
        enum struct Status {
            Viable,
            ArgumentCountMismatch,
            ArgumentTypeMismatch,
            NoViableArgOverload,
        };

        enum { InvalidScore = -1 };

        ProcDecl* proc;
        SmallVector<ConversionSequence> arg_convs;
        Status s = Status::Viable;
        int score = InvalidScore;

        LLVM_READONLY readonly_const(ProcType*, type, return cast<ProcType>(proc->type));
        usz mismatch_index{};
    };

    /// Machinery for tracking the state of optionals.
    class OptionalState {
        using Path = SmallVector<u32, 4>;
        struct State {
            /// Whether the variable itself is active.
            bool active : 1;

            /// List of field paths that are active, by field index.
            SmallVector<Path, 1> active_fields;
        };

    public:
        /// Guard for entering a scope.
        class ScopeGuard {
            friend OptionalState;
            Sema& S;
            ScopeGuard* previous;

            /// Entities of optional type whose active state has changed in
            /// this scope, as well as the value of that state in the previous
            /// scope.
            DenseMap<LocalDecl*, State> changes;

        public:
            ScopeGuard(Sema& S);
            ~ScopeGuard();
        };

        /// Guard for temporarily making a value active. This always
        /// resets the active state to what it was before the guard,
        /// irrespective of whether it is changed after the guard was
        /// created.
        struct ActivationGuard {
            friend OptionalState;
            Sema& S;
            Expr* expr;

        public:
            ActivationGuard(Sema& S, Expr* expr);
            ~ActivationGuard();
        };

    private:
        /// All entities that are currently tracked, and whether they
        /// are active. Key is the root object of the entity, which
        /// is always a local variable (this is because tracking only
        /// makes sense for lvalues since rvalues don’t survive long
        /// enough to be tracked in the first place).
        DenseMap<LocalDecl*, State> tracked;

        /// Current guard.
        ScopeGuard* guard{};


        /// Used to implement Activate() and Deactivate().
        void ChangeState(Expr* e, auto cb);

        /// Get the path to an entity of optional type.
        auto GetObjectPath(MemberAccessExpr* e) -> std::pair<LocalDecl*, Path>;

    public:
        /// Mark an optional as active until the end of the current scope.
        ///
        /// \param e The expression to activate. May be null.
        void Activate(Expr* e);

        /// Mark an optional as inactive until the end of the current scope.
        ///
        /// \param e The expression to deactivate. May be null.
        void Deactivate(Expr* e);

        /// If this is an active entity of optional type, return the optional
        /// type, else return null.
        auto GetActiveOptionalType(Expr* e) -> OptionalType*;

        /// Test if an expression checks whether an entity of optional type
        /// is nil, and if so, return a pointer to that entity.
        auto MatchNilTest(Expr* test) -> Expr*;
    } Optionals;

    /// Expressions that still need to be checked for unwinding.
    SmallVector<Unwind> unwind_entries;

    /// Expressions that need to know what the current full expression is.
    SmallVector<Expr*> needs_link_to_full_expr{};

    /// Expressions eligible for `.x` access.
    SmallVector<Expr*> with_stack;

    readonly(Context*, ctx, return mod->context);

    /// Number of anonymous procedures.
    usz lambda_counter = 0;

    /// Whether we’re currently the direct child of a block.
    bool at_block_level = false;

    /// Whether to print unsupported C++ imports.
    bool debug_cxx = false;

public:
    Sema(Module* mod) : mod(mod) {}

    /// Use Context::has_error to check for errors.
    static void Analyse(Module* mod, bool debug_cxx = false) {
        Sema s{mod};
        s.debug_cxx = debug_cxx;
        s.AnalyseModule();
    }

    /// Analyse the given expression and issue an error if it is not a type.
    bool AnalyseAsType(Type& e, bool diag_if_not_type = true);

private:
    bool Analyse(Expr*& e);

    template <bool allow_undefined>
    bool AnalyseDeclRefExpr(Expr*& e);

    void AnalyseExplicitCast(Expr*& e, bool is_hard);
    bool AnalyseInvoke(Expr*& e, bool direct_child_of_block = false);
    bool AnalyseInvokeBuiltin(Expr*& e);
    void AnalyseModule();
    void AnalyseProcedure(ProcDecl* proc);
    bool AnalyseProcedureType(ProcDecl* proc);
    void AnalyseRecord(RecordType* r);

    /// Apply a conversion sequence to an expression.
    void ApplyConversionSequence(Expr*& e, std::same_as<ConversionSequence> auto&& seq);

    /// Determine whether a parameter should be passed by value and check its type.
    /// \return False on error.
    bool ClassifyParameter(ParamInfo* info);

    /// Get a constructor for a type and a set of arguments.
    ///
    /// \param loc Location to use for errors.
    /// \param ty The type to construct.
    /// \param args Arguments with which to construct a \p ty.
    /// \param target Expression that is marked as active if this creates an active optional.
    /// \return The constructor for `into`.
    auto Construct(
        Location loc,
        Type ty,
        MutableArrayRef<Expr*> args,
        Expr* target = nullptr
    ) -> ConstructExpr*;

    /// Convert an expression to a type, inserting implicit conversions
    /// as needed. This  *will* perform lvalue-to-rvalue conversion if
    /// the type conversion requires it and also in any case unless \p
    /// lvalue is true.
    bool Convert(Expr*& e, Type to, bool lvalue = false);

    /// Implements Convert() and TryConvert().
    template <bool perform_conversion>
    bool ConvertImpl(
        ConversionContext<perform_conversion>& ctx,
        Type from, /// Required for recursive calls in non-conversion mode.
        Type to
    );

    /// Ensure that an expression is valid as the condition of an if expression,
    /// while loop, etc.
    bool EnsureCondition(Expr*& e);

    /// Create an implicit dereference, but do not overwrite the original expression.
    [[nodiscard]] auto CreateImplicitDereference(Expr* e, isz depth) -> Expr*;

    /// Create a diagnostic at a location.
    template <typename... Args>
    Diag EmitError(Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        if (e->sema.errored) return Diag();
        e->sema.set_errored();
        return Diag::Error(mod->context, e->location, fmt, MakeFormattable(std::forward<Args>(args))...);
    }

    /// Create a diagnostic and mark an expression as errored.
    template <typename... Args>
    Diag EmitError(Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        return Diag::Error(mod->context, loc, fmt, MakeFormattable(std::forward<Args>(args))...);
    }

    /// Same as EmitError(), but returns false for convenience.
    template <typename... Args>
    bool Error(Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        EmitError(loc, fmt, MakeFormattable(std::forward<Args>(args))...);
        return false;
    }

    /// Same as EmitError(), but returns false for convenience.
    template <typename... Args>
    bool Error(Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        EmitError(e, fmt, std::forward<Args>(args)...);
        return false;
    }

    /// Evaluate a constant expression.
    bool Evaluate(Expr* e, EvalResult& out, bool must_succeed = true);

    /// Evaluate an integral constant expression and replace it with the result.
    bool EvaluateAsIntegerInPlace(Expr*& e, bool must_succeed = true);

    /// Evaluate a constant expression as an overload set. This must always
    /// yield an overload set, so if this fails, it’s an ICE. Only call this
    /// if the type of the expression is actually OverloadSet.
    auto EvaluateAsOverloadSet(Expr* e) -> OverloadSetExpr*;

    /// Perform any final operations (after type conversion) required to
    /// pass an expression an an argument to a call.
    ///
    /// \param arg The argument to finalise.
    /// \param param The parameter to which the argument is being passed, or
    ///        nullptr if this is a variadic argument.
    /// \return False on error.
    bool FinaliseInvokeArgument(Expr*& arg, const ParamInfo* param);

    /// Dereference a reference, yielding an lvalue.
    ///
    /// This automatically handles dereferencing both references that
    /// are themselves lvalues and rvalues.
    void InsertImplicitDereference(Expr*& e, isz depth);

    /// Perform lvalue-to-rvalue conversion.
    ///
    /// Notably, this does *not* change the type of the expression; unlike
    /// in C++, expressions of reference type can be rvalues or lvalues.
    void InsertLValueToRValueConversion(Expr*& e);

    /// Check if this is an 'in' parameter.
    bool IsInParameter(Expr* e);

    template <bool in_array = false>
    bool MakeDeclType(Type& e);

    template <typename T>
    auto MakeFormattable(T&& t) -> make_formattable_t<T> {
        using Type = std::remove_cvref_t<T>;
        if constexpr (std::is_pointer_v<Type> and std::derived_from<std::remove_pointer_t<Type>, Expr>) {
            return std::forward<T>(t)->type.str(mod->context->use_colours, true);
        } else if constexpr (std::is_same_v<std::remove_cvref_t<Type>, src::Type>) {
            return std::forward<T>(t).str(mod->context->use_colours, true);
        } else {
            return std::forward<T>(t);
        }
    }

    /// Resolve overload set.
    ///
    /// \param where Location for issuing diagnostics.
    /// \param overloads The overloads to resolve.
    /// \param args The arguments to the call.
    /// \param required Whether resolution must succeed. Diagnostics are suppressed
    ///        if this is false.
    /// \return The resolved overload, or nullptr if resolution failed.
    auto PerformOverloadResolution(
        Location where,
        ArrayRef<ProcDecl*> overloads,
        MutableArrayRef<Expr*> args,
        bool required
    ) -> ProcDecl*;

    /// Print diagnostic for overload resolution failure.
    void ReportOverloadResolutionFailure(
        Location where,
        ArrayRef<Candidate> overloads,
        ArrayRef<Expr*> args
    );

    /// Like Convert(), but does not perform the conversion, and does not
    /// issue any diagnostics on conversion failure.

    /// This  *will* perform lvalue-to-rvalue conversion if
    /// the type conversion requires it and also in any case unless \p
    /// lvalue is true.
    bool TryConvert(ConversionSequence& out, Expr* e, Type to, bool lvalue = false);

    /// Strip references and optionals (if they’re active) from the expression
    /// to yield the underlying value.
    [[nodiscard]] Expr* Unwrap(Expr* e, bool keep_lvalues = false);

    /// Unwrap an expression and replace it with the unwrapped expression.
    void UnwrapInPlace(Expr*& e, bool keep_lvalues = false);

    /// Get the unwrapped type of an expression.
    Type UnwrappedType(Expr* e);

    /// Unwinder.
    ///
    /// Used for stack unwinding as part of direct branches (goto, break
    /// continue, return).
    ///
    /// If this is a small vector, store unwound expressions in it. If it
    /// is an expression, instead emit an error and mark that expression as
    /// errored.
    using UnwindContext = llvm::PointerUnion<SmallVectorImpl<Expr*>*, Expr*>;
    bool UnwindLocal(UnwindContext ctx, BlockExpr* S, Expr* FE, Expr* To);
    auto Unwind(UnwindContext ctx, BlockExpr* S, Expr* E, BlockExpr* To) -> Expr*;
    void UnwindUpTo(BlockExpr* parent, BlockExpr* to, UnwindExpr* uw);
    void ValidateDirectBr(GotoExpr* g, BlockExpr* source);
};
} // namespace src

#endif // SOURCE_FRONTEND_SEMA_HH
