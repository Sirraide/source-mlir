#ifndef SOURCE_FRONTEND_SEMA_HH
#define SOURCE_FRONTEND_SEMA_HH

#include <source/Frontend/AST.hh>

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
struct make_formattable<Expr::TypeHandle> {
    using type = std::string;
};

template <typename T>
using make_formattable_t = make_formattable<T>::type;

class Sema {
    Module* mod{};
    ProcDecl* curr_proc{};
    BlockExpr* curr_scope{};

    /// Loops that we’re currently analysing.
    SmallVector<WhileExpr*, 10> loop_stack;

    /// The defer expression whose contents we are currently analysing.
    DeferExpr* curr_defer{};

    /// Unwinding is performed once everything else has been checked.
    struct Unwind {
        BlockExpr* in_scope;
        UnwindExpr* expr;

        /// Unused if this is a goto.
        BlockExpr* to_scope{};
    };

    /// Overload candidate.
    struct Candidate {
        enum struct Status {
            Viable,
            ArgumentCountMismatch,
            ArgumentTypeMismatch,
            NoViableArgOverload,
        };

        ProcDecl* proc;
        Status s = Status::Viable;
        int score{};

        LLVM_READONLY readonly_const(ProcType*, type, return cast<ProcType>(proc->type));
        usz mismatch_index{};
    };

    SmallVector<Unwind> unwind_entries;

    /// Expressions that need to know what the current full expression is.
    SmallVector<Expr*> needs_link_to_full_expr{};

    /// Set when encountering a protected expression.
    SmallVector<Expr*, 1> protected_subexpressions{};

    /// Expressions eligible for `.x` access.
    SmallVector<Expr*> with_stack;

    /// Locals of optional type whose active state has changed in this scope,
    /// as well as the value of that state in the previous scope.
    DenseMap<LocalDecl*, bool> active_optionals;

    /// Number of anonymous procedures.
    usz lambda_counter = 0;

public:
    /// Use Context::has_error to check for errors.
    static void Analyse(Module* mod) {
        Sema s{mod};
        s.AnalyseModule();
    }

private:
    Sema(Module* mod) : mod(mod) {}

    bool Analyse(Expr*& e);

    /// Analyse the given expression and issue an error if it is not a type.
    bool AnalyseAsType(Expr*& e);

    template <bool allow_undefined>
    bool AnalyseDeclRefExpr(Expr*& e);

    bool AnalyseInvokeBuiltin(Expr*& e);
    void AnalyseExplicitCast(Expr*& e, bool is_hard);
    void AnalyseProcedure(ProcDecl* proc);
    bool AnalyseProcedureType(ProcDecl* proc);
    void AnalyseModule();

    /// Handle variable initialisation. This assumes that the type of the
    /// variable is known and valid.
    bool AnalyseVariableInitialisation(LocalDecl* var);

    /// Convert an expression to a type, inserting implicit conversions
    /// as needed. This  *will* perform lvalue-to-rvalue conversion if
    /// the type conversion requires it and also in any case unless \p
    /// lvalue is true.
    bool Convert(Expr*& e, Expr* to, bool lvalue = false);

    /// Implements Convert() and TryConvert().
    template <bool perform_conversion>
    int ConvertImpl(
        std::conditional_t<perform_conversion, Expr*&, Expr*> e,
        Expr* from, /// Required for recursive calls in non-conversion mode.
        Expr* to
    );

    /// Ensure that an expression is valid as the condition of an if expression,
    /// while loop, etc.
    bool EnsureCondition(Expr*& e);

    /// Create an implicit dereference, but do not overwrite the original expression.
    [[nodiscard]] auto CreateImplicitDereference(Expr* e, isz depth) -> Expr*;

    /// Returns false for convenience.
    template <typename... Args>
    bool Error(Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        Diag::Error(mod->context, loc, fmt, MakeFormattable(std::forward<Args>(args))...);
        return false;
    }

    /// Returns false for convenience.
    template <typename... Args>
    bool Error(Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        if (e->sema.errored) return false;
        Diag::Error(mod->context, e->location, fmt, MakeFormattable(std::forward<Args>(args))...);
        e->sema.set_errored();
        return false;
    }

    /// Evaluate a constant expression.
    bool Evaluate(Expr* e, EvalResult& out, bool must_succeed = true);

    /// Evaluate an integral constant expression and replace it with the result.
    bool EvaluateAsIntegerInPlace(Expr*& e, bool must_succeed = true);

    void InsertImplicitCast(Expr*& e, Expr* to);

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

    template <bool in_array = false>
    bool MakeDeclType(Expr*& e);

    template <typename T>
    auto MakeFormattable(T&& t) -> make_formattable_t<T> {
        using Type = std::remove_cvref_t<T>;
        if constexpr (std::is_pointer_v<Type> and std::derived_from<std::remove_pointer_t<Type>, Expr>) {
            return std::forward<T>(t)->type.str(true);
        } else if constexpr (std::is_same_v<Type, Expr::TypeHandle>) {
            return std::forward<T>(t).str(true);
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

    /// Like Convert(), but does not perform the conversion, does not
    /// issue any diagnostics, and returns a score suitable for overload
    /// resolution.
    int TryConvert(Expr* e, Expr* to);

    /// Strip references and optionals (if they’re active) from the expression
    /// to yield the underlying value.
    [[nodiscard]] Expr* Unwrap(Expr* e, bool keep_lvalues = false);

    /// Unwrap an expression and replace it with the unwrapped expression.
    void UnwrapInPlace(Expr*& e, bool keep_lvalues = false);

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
