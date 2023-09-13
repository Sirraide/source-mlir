#ifndef SOURCE_INCLUDE_FRONTEND_AST_HH
#define SOURCE_INCLUDE_FRONTEND_AST_HH

#include <source/Core.hh>
#include <source/Frontend/Lexer.hh>
#include <source/Support/Result.hh>

namespace src {

class Expr;
class Type;
class StringLiteralExpr;
class FunctionDecl;
class Scope;
class ParamDecl;
class VarDecl;

/// Types used in some declarations before `Type` is defined.
namespace detail {
extern Expr* UnknownTy;
extern Expr* StringLiteralTy;
}

/// ===========================================================================
///  Enums
/// ===========================================================================
enum struct Linkage {
    Local,      ///< Local variable.
    Internal,   ///< Not exported and defined.
    Imported,   ///< Imported from another module or library.
    Exported,   ///< Exported and defined.
    Reexported, ///< Imported and exported, and thus not defined.
};

enum struct Mangling {
    None,   ///< Do not mangle.
    Source, ///< Use Source mangling.
};

enum struct CallingConv {
    C,        ///< C calling convention.
    Internal, ///< Fast calling convention for internal functions.
};

/// ===========================================================================
///  Constant Expression Evaluator
/// ===========================================================================
class EvalResult {
    friend class Expr;

    Type* _type;
    std::variant<i64, f64, Type*, StringLiteralExpr*, std::nullptr_t> data;

public:
    template <typename... Ts>
    bool is() { return (std::holds_alternative<Ts>(data) or ...); }

    auto type() const -> Type* { return _type; }
    auto as_bool() const -> bool { return bool(std::get<i64>(data)); }
    auto as_float() const -> f64 { return std::get<f64>(data); }
    auto as_int() const -> i64 { return std::get<i64>(data); }
    auto as_type() const -> Type* { return std::get<Type*>(data); }
    auto as_string() -> StringLiteralExpr* { return std::get<StringLiteralExpr*>(data); }
};

/// ===========================================================================
///  Base
/// ===========================================================================
/// Base class for all AST nodes and types.
class Expr {
public:
    /// LLVM RTTI Discriminator.
    enum Kind {
        /// Special expressions.
        EK_Root,
        EK_ExprBundle,
        EK_StaticExpr,
        EK_DeclGroup,
        EK_Type,

        /// Untyped expressions.
        EK_ForInExpr,
        EK_ForInfinite,
        EK_ForCStyle,
        EK_WhileExpr,
        EK_RepeatUntilExpr,
        EK_RepeatExpr,
        EK_DiscardedExpr,
        EK_ReturnExpr,
        EK_DeferExpr,
        EK_BreakContinueExpr,
        EK_AssertExpr,
        EK_WithExpr,
        EK_UnreachableExpr,
        EK_CaseExpr,
        EK_AsmExpr,

        /// Typed Expressions.
        EK_ParenExpr,
        EK_InvokeExpr,
        EK_IfExpr,
        EK_MatchExpr,
        EK_BlockExpr,
        EK_NameRefExpr,
        EK_MemberAccessExpr,
        EK_ScopeAccessExpr,
        EK_ParamRefExpr,
        EK_UnaryExpr,
        EK_BinaryExpr,
        EK_VectorReduceExpr,
        EK_ClosureExpr,
        EK_TupleLiteralExpr,
        EK_ArrayLiteralExpr,
        EK_StringLiteralExpr,
        EK_NullLiteralExpr,
        EK_IntegerLiteralExpr,
        EK_BuiltinLiteralExpr,
        EK_CastExpr,
        EK_TryExpr,

        /// Object Declarations.
        EK_FunctionDecl,
        EK_VarDecl,

        /// Declarations.
        EK_ParamDecl,
        EK_MemberDecl,
        EK_VariantClauseDecl,
        EK_EnumeratorDecl,

        /// Named types.
        EK_EnumType,
        EK_OpaqueType,
        EK_TupleType,
        EK_StructType,

        /// Templates.
        EK_StructTemplate,

        /// Other types.
        EK_BuiltinType,
        EK_AlternativeType,
        EK_FFIType,
        EK_ScopedPointerType,
        EK_ReferenceType,
        EK_OptionalType,
        EK_SliceType,
        EK_InclusiveRangeType,
        EK_ExclusiveRangeType,
        EK_ClosureType,
        EK_ValueType,
        EK_VariableType,
        EK_DeclTypeDecayType,
        EK_TypeofType,
        EK_ArrayType,
        EK_VectorType,
        EK_FunctionType,
        EK_VariantClauseType,
        EK_IntegerType,
        EK_AlignedType,
    };

private:
    property_r(const Kind, kind);

    /// The location of this expression.
    property_rw(Location, location);

protected:
    /// Whether this expression has been type-checked.
    bool _type_checked = false;

    Expr(Kind k, Location _location = {}) : kind_field(k), location_field(_location) {}

    /// Copying or moving AST nodes is nonsense.
    Expr(const Expr&) = delete;
    Expr(Expr&&) = delete;
    Expr& operator=(const Expr&) = delete;
    Expr& operator=(Expr&&) = delete;

public:
    virtual ~Expr() = default;

    /// Make sure we never just allocate an Expr.
    void* operator new(size_t) = delete;

    /// Creating an Expr requires a Module.
    /// Note: This is implemented in module.cc.
    /// TODO: Use __builtin_operator_new(): https://clang.llvm.org/docs/LanguageExtensions.html#builtin-operator-new-and-builtin-operator-delete
    void* operator new(size_t, Module*);

    /// Evaluate this expression.
    ///
    /// \param ctx The Source context. Needed for `::size` etc.
    /// \param res Out parameter for the result of the evaluation.
    /// \param diag What kind of diagnostic to issue on failure.
    /// \return Whether evaluation was successful.
    bool evaluate(Context* ctx, EvalResult& res, Diag::Kind diag);

    /// Check if this expression could be a type.
    bool is_maybe_a_type() const;

    /// Mark an expression as type-checked.
    void mark_type_checked() { _type_checked = true; }

    /// Get the type of this expression.
    ///
    /// If this is a type, then this function just returns
    /// a reference to this object. If this expression is
    /// untyped, this returns void.
    readonly_decl(Type*, type);

    /// Check if this expression has been type-checked.
    auto type_checked() const -> bool { return _type_checked; }
};

/// ===========================================================================
///  Helpers
/// ===========================================================================
/// A list of expressions
using ExprList = SmallVector<Expr*, 4>;

/// ===========================================================================
///  Special Expressions
/// ===========================================================================
/// A group of declarations.
class DeclGroupExpr : public Expr {
    /// The declarations in this group.
    ExprList decls;

public:
    DeclGroupExpr() : Expr(EK_DeclGroup) {}

    /// Add a declaration to this group.
    void add(Expr* e) { decls.push_back(e); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_DeclGroup; }
};

/// A bundle of expressions.
///
/// This has no syntactic equivalent and is used to store a group
/// of expressions in a slot that can only hold one. Semantically,
/// this is equivalent to a block expressions, except that it does
/// not introduce a new scope.
class ExprBundle : public Expr {
    /// The expressions in this bundle.
    ExprList exprs;

public:
    ExprBundle() : Expr(EK_ExprBundle) {}
    ExprBundle(ExprList expressions) : Expr(EK_ExprBundle), exprs(std::move(expressions)) {}
    ExprBundle(Expr* expr) : Expr(EK_ExprBundle) { exprs.push_back(expr); }

    /// Add an expression to this bundle.
    void add(Expr* e) { exprs.push_back(e); }

    /// Get an iterator to the first expression.
    auto begin() const -> decltype(exprs.begin()) { return exprs.begin(); }
    auto begin() -> decltype(exprs.begin()) { return exprs.begin(); }

    /// Get an iterator to the end of the expressions.
    auto end() const -> decltype(exprs.end()) { return exprs.end(); }
    auto end() -> decltype(exprs.end()) { return exprs.end(); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ExprBundle; }
};

/// Expression to be evaluated during sema that generates expressions.
///
/// This is used to represent constructs that are introduced by the
/// `static` keyword, such as `static if` and `static for`.
///
/// A `static if` selects between two expressions, and loops such as
/// `static for` generate a list of expressions; these expressions are
/// treated like a block expressions, except for when they’re in the
/// body of another block expression, a `match` expression, and a few
/// other places where they are spliced into the surrounding context.
///
/// A static expression may generate no expressions at all, which is
/// the case e.g. for a loop whose condition is always false, as well
/// as for `static assert` expressions.
///
/// If the type of a static expression is queried, it will return the
/// type of the last generated expression, just like a block expression,
/// or `void` if no expressions were generated.
class StaticExpr : public Expr {
    /// The expression to evaluate.
    property_r(Expr*, expr);

    /// Generated or selected expressions.
    property_r(ExprList, selected_exprs);

public:
    StaticExpr(Expr* expr, Location loc)
        : Expr(EK_StaticExpr), expr_field(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_StaticExpr; }
};

/// ===========================================================================
///  Untyped Expressions
/// ===========================================================================
/// Ranged for-in loop.
class ForInExpr : public Expr {
    /// The loop variable (optional).
    property_r(Expr*, var);

    /// The enumerator variable (also optional).
    property_r(Expr*, enum_var);

    /// The loop range.
    property_r(Expr*, range);

    /// The loop body.
    property_r(Expr*, body);

public:
    ForInExpr(Expr* var, Expr* enum_var, Expr* range, Expr* body, Location loc = {})
        : Expr(EK_ForInExpr, loc),
          var_field(var),
          enum_var_field(enum_var),
          range_field(range),
          body_field(body) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ForInExpr; }
};

/// Infinite loop.
class ForInfiniteExpr : public Expr {
    /// The loop body.
    property_r(Expr*, body);

public:
    ForInfiniteExpr(Expr* body, Location loc = {})
        : Expr(EK_ForInfinite, loc), body_field(body) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ForInfinite; }
};

/// C-style for loop.
class ForCStyleExpr : public Expr {
    /// The loop init.
    property_r(Expr*, init);

    /// The loop condition.
    property_r(Expr*, cond);

    /// The loop increment.
    property_r(Expr*, inc);

    /// The loop body.
    property_r(Expr*, body);

public:
    ForCStyleExpr(Expr* init, Expr* cond, Expr* inc, Expr* body, Location loc = {})
        : Expr(EK_ForCStyle, loc),
          init_field(init),
          cond_field(cond),
          inc_field(inc),
          body_field(body) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ForCStyle; }
};

/// While loop.
class WhileExpr : public Expr {
    /// The loop body.
    Expr* _body;

    /// The loop condition.
    Expr* _cond;

public:
    WhileExpr(Expr* cond, Expr* body, Location loc = {})
        : Expr(EK_WhileExpr, loc), _body(body), _cond(cond) {}

    /// Get the loop body.
    auto body() const -> Expr* { return _body; }

    /// Get the loop condition.
    auto cond() const -> Expr* { return _cond; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_WhileExpr; }
};

/// Repeat-until loop.
class RepeatUntilExpr : public Expr {
    /// The loop body.
    Expr* _body;

    /// The loop condition.
    Expr* _cond;

public:
    RepeatUntilExpr(Expr* cond, Expr* body)
        : Expr(EK_RepeatUntilExpr), _body(body), _cond(cond) {}

    /// Get the loop body.
    auto body() const -> Expr* { return _body; }

    /// Get the loop condition.
    auto cond() const -> Expr* { return _cond; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_RepeatUntilExpr; }
};

/// Repeat loop.
class RepeatExpr : public Expr {
    /// The loop body.
    Expr* _body;

    /// The repetition count.
    Expr* _count;

public:
    RepeatExpr(Expr* count, Expr* body)
        : Expr(EK_RepeatExpr), _body(body), _count(count) {}

    /// Get the loop body.
    auto body() const -> Expr* { return _body; }

    /// Get the repetition count.
    auto count() const -> Expr* { return _count; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_RepeatExpr; }
};

/// Discarded value expression.
class DiscardedExpr : public Expr {
    /// The expression whose value is discarded.
    Expr* _expr;

public:
    DiscardedExpr(Expr* expr) : Expr(EK_DiscardedExpr), _expr(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_DiscardedExpr; }
};

/// Return expression.
class ReturnExpr : public Expr {
    /// The expression to return, if any.
    property_r(Expr*, expr);

public:
    ReturnExpr(Expr* expr, Location loc)
        : Expr(EK_ReturnExpr, loc), expr_field(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ReturnExpr; }
};

/// Deferred expression.
class DeferExpr : public Expr {
    /// The expression to defer.
    property_r(Expr*, expr);

public:
    DeferExpr(Expr* expr, Location loc)
        : Expr(EK_DeferExpr, loc), expr_field(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_DeferExpr; }
};

/// Break or continue expression.
class BreakContinueExpr : public Expr {
    /// The target label, if any.
    property_r(Expr*, label);
    property_r(std::string, label_name);
    property_r(bool, is_break);
    readonly(bool, is_continue, return not is_break);

public:
    BreakContinueExpr(std::string label, bool is_break, Location loc)
        : Expr(EK_BreakContinueExpr, loc),
          label_name_field(std::move(label)),
          is_break_field(is_break) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_BreakContinueExpr; }
};

/// Assert expression.
///
/// This is also used for static assertions.
class AssertExpr : public Expr {
    /// The assertion condition.
    Expr* _cond;

    /// The assertion message.
    Expr* _msg;

public:
    AssertExpr(Expr* cond, Expr* msg)
        : Expr(EK_AssertExpr, cond->location), _cond(cond), _msg(msg) {}

    /// Get the assertion condition.
    auto cond() const -> Expr* { return _cond; }

    /// Get the assertion message.
    auto msg() const -> Expr* { return _msg; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_AssertExpr; }
};

/// With expression.
///
/// With expressions can be scoped or unscoped. Scoped with expressions
/// have a body, and the members of the controlling expression are scoped
/// to that body. Unscoped with expressions have no body, and the members
/// of the controlling expression are accessible within the rest of the
/// scope that contains the with expression
///
/// Examples:
/// ```
///     with a { ... } /// Scoped `with` expression.
///     with a;        /// Unscoped `with` expression.
/// ```
class WithExpr : public Expr {
    /// The controlling expression.
    property_r(Expr*, expr);

    /// The body, if any.
    property_r(Expr*, body);

    /// Associated scope.
    ///
    /// If this is scoped, the body’s scope, otherwise,
    /// the parent scope containing the with expression.
    property_r(Scope*, scope);

public:
    WithExpr(Expr* expr, Expr* body, Scope* scope, Location loc = {})
        : Expr(EK_WithExpr, loc),
          expr_field(expr),
          body_field(body),
          scope_field(scope) {}

    /// Get whether this is a scoped with expression.
    bool is_scoped() const { return body != nullptr; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_WithExpr; }
};

/// Unreachable expression.
class UnreachableExpr : public Expr {
public:
    UnreachableExpr(Location loc) : Expr(EK_UnreachableExpr, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_UnreachableExpr; }
};

/// Inline assembly.
class AsmExpr : public Expr {
    /// Currently just a string.
    SmallVector<std::string> _asm;

public:
    AsmExpr(SmallVector<std::string> asm_) : Expr(EK_AsmExpr), _asm(std::move(asm_)) {}

    /// Get the assembly string.
    auto asm_string() const -> const SmallVector<std::string>& { return _asm; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_AsmExpr; }
};

/// ===========================================================================
///  Typed Expressions
/// ===========================================================================
/// Base class for expressions that are primarily values.
class TypedExpr : public Expr {
    /// Non-type expressions have a type.
    ///
    /// Since a lot of expressions may end up being types, we
    /// unfortunately can’t just use `Type` here.
    property_rw(Expr*, type);

protected:
    TypedExpr(Expr::Kind k, Expr* type, Location loc = {}) : Expr(k, loc), type_field(type) {}

public:
    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= EK_InvokeExpr and e->kind <= EK_EnumeratorDecl;
    }
};

/// Expression surrounded by parens.
class ParenExpr : public TypedExpr {
    /// The expression inside the parens.
    property_r(Expr*, expr);

public:
    ParenExpr(Expr* expr, Location loc)
        : TypedExpr(EK_ParenExpr, detail::UnknownTy, loc), expr_field(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ParenExpr; }
};

/// Invoke Expression.
///
/// This is user for anything that syntactically looks like function
/// application, that is, for procedure calls, qualified and member
/// function calls, as well as struct literals, template instantiations,
/// and declarations before Sema.
class InvokeExpr : public TypedExpr {
    /// The invoked expression.
    property_r(Expr*, invokee);

    /// The arguments.
    property_r(ExprList, args);

    /// Any assignment after an invoke expression is bound to
    /// the expression for easy access to constructor parameters
    /// etc.
    property_r(Expr*, assignment);

    /// Whether this invocation uses parentheses.
    property_r(bool, has_parens);

public:
    InvokeExpr(
        Expr* invokee,
        ExprList args,
        bool has_parens,
        Location loc,
        Expr* assignment = nullptr
    ) : TypedExpr(EK_InvokeExpr, detail::UnknownTy, loc),
        invokee_field(invokee),
        args_field(std::move(args)),
        assignment_field(assignment),
        has_parens_field(has_parens) {}

    /// Add an argument.
    void add_arg(Expr* e) { args.push_back(e); }

    /// Get an iterator to the first argument.
    auto begin() const { return args.begin(); }
    auto begin() { return args.begin(); }

    /// Get an iterator to the last argument.
    auto end() const { return args.end(); }
    auto end() { return args.end(); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_InvokeExpr; }
};

/// If expression
class IfExpr : public TypedExpr {
    /// The condition.
    Expr* _cond;

    /// The then branch.
    Expr* _then;

    /// The else branch.
    Expr* _else;

    /// True if this is a `static if` expression.
    bool _is_static;

public:
    IfExpr(Expr* cond, Expr* then, Expr* else_, Location loc = {});

    /// Get the condition.
    auto cond() const -> Expr* { return _cond; }

    /// Get the else branch.
    auto else_() const -> Expr* { return _else; }

    /// Get whether this is a `static if` expression.
    auto is_static() const -> bool { return _is_static; }

    /// Get the then branch.
    auto then() const -> Expr* { return _then; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_IfExpr; }
};

/// Case of a match expression.
class CaseExpr : public Expr {
    /// Controlling expression.
    property_r(Expr*, control);

    /// Body.
    property_r(Expr*, body);

    /// Whether the case is allowed to fall through.
    property_r(bool, fallthrough);

public:
    CaseExpr(Expr* control, Expr* body, bool fallthrough, Location loc = {})
        : Expr(EK_CaseExpr, loc),
          control_field(control),
          body_field(body),
          fallthrough_field(fallthrough) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_CaseExpr; }
};

/// Match expression.
class MatchExpr : public TypedExpr {
public:
    enum struct Kind {
        Unknown,       ///< Not yet determined.
        Boolean,       ///< No controlling expr. First expr that is true is taken.
        Operator,      ///< Controlling expr is compared to each case using an operator.
        Variant,       ///< Match over variant clauses.
        Type,          ///< Match over types.
        VariantOrType, ///< Not yet determined, but either Variant or Type.
    };

private:
    Kind _kind;

    /// The controlling expression, if any.
    Expr* _control;

    /// The match cases.
    SmallVector<CaseExpr*> _cases;

    /// The default case, if any.
    CaseExpr* _default;

    /// The match operator, if any.
    Tk _op;

    /// Operator location.
    mutable Location _op_loc;

public:
    MatchExpr(
        Expr* control,
        Tk op,
        SmallVector<CaseExpr*> cases,
        CaseExpr* default_,
        Location loc
    ) : TypedExpr(EK_MatchExpr, detail::UnknownTy, loc),
        _kind(Kind::Unknown),
        _control(control),
        _cases(std::move(cases)),
        _default(default_),
        _op(op),
        _op_loc(loc) {}

    /// Get the match cases.
    auto cases() const -> const SmallVector<CaseExpr*>& { return _cases; }

    /// Get the controlling expression, if any.
    auto control() const -> Expr* { return _control; }

    /// Get the body of the default case, if any.
    auto default_case() const -> Expr* { return _default; }

    /// Get the kind of match.
    auto match_kind() const -> Kind { return _kind; }

    /// Get the match operator, if any.
    auto op() const -> Tk { return _op; }

    /// Get the location of the match operator, if any.
    auto op_location() const -> Location& { return _op_loc; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_MatchExpr; }
};

/// Block expression.
class BlockExpr : public TypedExpr {
    /// The block body.
    property_r(ExprList, exprs);

    /// Whether this is an implicit or explicit block.
    property_r(bool, is_implicit);

public:
    BlockExpr(ExprList body, bool implicit, Location loc) : TypedExpr(EK_BlockExpr, detail::UnknownTy, loc),
                                                            exprs_field(std::move(body)),
                                                            is_implicit_field(implicit) {}

    /// Add an expression to the block.
    void add(Expr* e) { exprs.push_back(e); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_BlockExpr; }
};

/// A (qualified) name that refers to some declaration.
///
/// This is also used for
class NameRefExpr : public TypedExpr {
    /// The name.
    property_r(std::string, name);

    /// The scope in which the name will be looked up.
    property_r(Scope*, scope);

    /// The declaration this refers to.
    property_r(Decl*, decl);

    /// Distance to the declaration in the static chain; this is
    /// only relevant for local variables.
    property_rw(u32, chain_distance);

    /// Whether this is a local lookup (i.e. whether there is a
    /// leading '.').
    property_r(bool, is_local);

public:
    /// Get the location of the referenced declaration.
    readonly_decl(Location, decl_location);

    NameRefExpr(std::string name, Scope* scope, bool is_local, Location loc)
        : TypedExpr(EK_NameRefExpr, detail::UnknownTy, loc),
          name_field(std::move(name)),
          scope_field(scope),
          chain_distance_field(0),
          is_local_field(is_local) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_NameRefExpr; }
};

/// Member access expression.
///
/// Catch-all expression for anything that *looks* like member access,
/// that is, `A.B`. This includes member access, scope access, enum
/// access, module access, etc.
class MemberAccessExpr : public TypedExpr {
public:
    enum struct Kind {
        Unknown,      ///< Not yet determined.
        RecordMember, ///< Access to a record member.
        ScopeAccess,  ///< Access to a declaration in a scope or module.
        EnumMember,   ///< Access to an enumerator.
    };

private:
    /// The kind of member access.
    property_r(Kind, kind);

    /// The base expression.
    property_r(Expr*, base);

    /// The member that is accessed, or, if this is improper member access,
    /// i.e. it is actually a type etc., the type etc. this refers to.
    property_r(Expr*, member);

    /// The member name.
    property_r(std::string, name);

public:
    MemberAccessExpr(Expr* base, std::string name, Location loc)
        : TypedExpr(EK_MemberAccessExpr, detail::UnknownTy, loc),
          kind_field(Kind::Unknown),
          base_field(base),
          member_field(nullptr),
          name_field(std::move(name)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_MemberAccessExpr; }
};

/// Scope access expression.
///
/// E.g. `A::B::C`.
class ScopeAccessExpr : public TypedExpr {
    /// The name sequence.
    property_r(SmallVector<std::string>, name_list);

public:
    ScopeAccessExpr(SmallVector<std::string> name_list, Location loc)
        : TypedExpr(EK_ScopeAccessExpr, detail::UnknownTy, loc),
          name_list_field(std::move(name_list)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ScopeAccessExpr; }
};

/// Reference to a parameter value.
///
/// This references the value passed to a function parameter on entry
/// to the function. This is not generated by the parser and only used
/// in some parts of sema when synthesising code.
class ParamRefExpr : public TypedExpr {
    /// The parameter this refers to.
    ParamDecl* _param;

public:
    ParamRefExpr(ParamDecl* param);

    /// Get the parameter this refers to.
    auto param() const -> ParamDecl* { return _param; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ParamRefExpr; }
};

/// A unary expression.
///
/// Note that this may actually represent part of a declaration
/// due to ambiguities in the grammar.
class UnaryExpr : public TypedExpr {
    /// The operand.
    property_r(Expr*, operand);

    /// The operator.
    property_r(Tk, op);

    /// Whether this is a postfix expression.
    property_r(const bool, is_postfix);

    /// Whether this is a prefix expression.
    readonly(bool, is_prefix, return not is_postfix);

public:
    UnaryExpr(Tk op, Expr* operand, bool is_postfix, Location loc = {})
        : TypedExpr(EK_UnaryExpr, detail::UnknownTy, loc),
          operand_field(operand),
          op_field(op),
          is_postfix_field(is_postfix) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_UnaryExpr; }
};

/// A binary expression.
///
/// Note that this may actually represent (part of) a declaration
/// due to ambiguities in the grammar.
class BinaryExpr : public TypedExpr {
    /// The left operand.
    property_r(Expr*, lhs);

    /// The right operand.
    property_r(Expr*, rhs);

    /// The operator.
    property_r(Tk, op);

public:
    BinaryExpr(Tk op, Expr* left, Expr* right, Location loc)
        : TypedExpr(EK_BinaryExpr, detail::UnknownTy, loc),
          lhs_field(left),
          rhs_field(right),
          op_field(op) {}

    /// Whether this is an assignment or compound assigment expression.
    bool is_assignment() const;

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_BinaryExpr; }
};

/// Vector reduction operation.
class VectorReduceExpr : public TypedExpr {
    /// The reduction operation.
    property_r(Tk, op);

    /// The operand.
    property_r(Expr*, operand);

public:
    VectorReduceExpr(Tk op, Expr* operand, Location loc)
        : TypedExpr(EK_VectorReduceExpr, detail::UnknownTy, loc),
          op_field(op),
          operand_field(operand) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_VectorReduceExpr; }
};

/// Kind of a cast expression.
enum struct CastKind {
    Soft,
    Hard,
    Implicit,
    LValueToRValue,
};

/// Cast expression.
///
/// The type to cast to is the type of the expression.
class CastExpr : public TypedExpr {
    /// The operand.
    property_r(Expr*, operand);

    /// What kind of cast this is.
    property_r(CastKind, cast_kind);

public:
    CastExpr(Expr* operand, Expr* type, CastKind k, Location loc)
        : TypedExpr(EK_CastExpr, type, loc),
          operand_field(operand),
          cast_kind_field(k) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_CastExpr; }
};

/// Try expression.
class TryExpr : public TypedExpr {
    /// The operand.
    property_r(Expr*, operand);

public:
    TryExpr(Expr* operand, Location loc = {})
        : TypedExpr(EK_TryExpr, detail::UnknownTy, loc), operand_field(operand) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_TryExpr; }
};

/// Tuple literal.
class TupleLiteralExpr : public TypedExpr {
    /// The tuple elements.
    property_r(ExprList, elements);

public:
    TupleLiteralExpr(ExprList elems, Location loc)
        : TypedExpr(EK_TupleLiteralExpr, detail::UnknownTy, loc),
          elements_field(std::move(elems)) {}

    /// Build a tuple type from the element values.
    auto build_tuple_type(Context* ctx) const -> Type*;

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_TupleLiteralExpr; }
};

/// Array literal.
class ArrayLiteralExpr : public TypedExpr {
    /// The array elements.
    property_r(ExprList, elements);

public:
    ArrayLiteralExpr(ExprList elems, Location loc)
        : TypedExpr(EK_ArrayLiteralExpr, detail::UnknownTy, loc),
          elements_field(std::move(elems)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ArrayLiteralExpr; }
};

/// Closure expression.
///
/// This represents an instance of a closure, which is a function
/// relative to another function.
class ClosureExpr : public TypedExpr {
    /// The function this is a closure of.
    FunctionDecl* _function;

    /// The function in which the closure was created.
    FunctionDecl* _parent;

public:
    ClosureExpr(FunctionDecl* function, FunctionDecl* parent)
        : TypedExpr(EK_ClosureExpr, detail::UnknownTy),
          _function(function),
          _parent(parent) {}

    /// Get the function this is a closure of.
    auto function() const -> FunctionDecl* { return _function; }

    /// Check if this is a trivial closure, i.e. one with no enviroment.
    auto is_trivial() const -> bool;

    /// Get the function in which the closure was created.
    auto parent() const -> FunctionDecl* { return _parent; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ClosureExpr; }
};

/// String literal.
class StringLiteralExpr : public TypedExpr {
    /// String literal index in the context.
    u32 _index;

public:
    StringLiteralExpr(u32 index, Location loc)
        : TypedExpr(EK_StringLiteralExpr, detail::StringLiteralTy, loc), _index(index) {}

    /// Get the string literal index.
    auto index() const -> u32 { return _index; }

    /// Get the value of this string literal.
    auto value(Context* ctx) const -> std::string_view;

    /// RTTI.
    static bool classof(const Expr* e) { return EK_StringLiteralExpr; }
};

/// Null literal.
class NullLiteralExpr : public TypedExpr {
public:
    NullLiteralExpr(Expr* type) : TypedExpr(EK_NullLiteralExpr, type) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_NullLiteralExpr; }
};

/// Integer literal.
///
/// This is NOT used for booleans.
class IntegerLiteralExpr : public TypedExpr {
    /// The value.
    i64 _value;

public:
    IntegerLiteralExpr(i64 value, Location loc, Expr* type = detail::UnknownTy)
        : TypedExpr(EK_IntegerLiteralExpr, type, loc), _value(value) {}

    /// Get the value.
    auto value() const -> i64 { return _value; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_IntegerLiteralExpr; }
};

/// Special literals (true, false, null).
class BuiltinLiteralExpr : public TypedExpr {
    /// The literal kind.
    property_r(Tk, kind);

public:
    BuiltinLiteralExpr(Tk kind, Location loc, Expr* type = detail::UnknownTy)
        : TypedExpr(EK_BuiltinLiteralExpr, type, loc), kind_field(kind) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_BuiltinLiteralExpr; }
};

/// ===========================================================================
///  Declarations
/// ===========================================================================
/// Base class for declarations.
class Decl : public TypedExpr {
    /// Every declaration has a name.
    property_r(std::string, name);

protected:
    Decl(Expr::Kind k, std::string name, Expr* type, Location loc = {})
        : TypedExpr(k, type, loc), name_field(std::move(name)) {}

public:
    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= EK_FunctionDecl and e->kind <= EK_EnumeratorDecl;
    }
};

/// Base class for declarations that have linkage.
class ObjectDecl : public Decl {
    /// Object declarations have a mangled name.
    std::string _mangled_name;

    /// Linkage of this declaration.
    property_rw(Linkage, linkage);

    /// Mangling scheme of this declaration.
    property_r(Mangling, mangling);

protected:
    ObjectDecl(
        Expr::Kind k,
        std::string name,
        Expr* type,
        Linkage linkage = Linkage::Internal,
        Mangling mangling = Mangling::Source,
        Location loc = {}
    ) : Decl(k, std::move(name), type, loc),
        linkage_field(linkage),
        mangling_field(mangling) {}

public:
    /// Check if this declaration is a definition.
    auto is_definition() const -> bool { return not is_imported(); }

    /// Check if this declaration is exported.
    auto is_exported() const -> bool { return linkage == Linkage::Exported or linkage == Linkage::Reexported; }

    /// Check if this declaration is imported.
    auto is_imported() const -> bool { return linkage == Linkage::Imported or linkage == Linkage::Reexported; }

    /// Get the mangled name.
    auto mangled_name(Context& ctx) const -> const std::string&;
    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= EK_FunctionDecl and e->kind <= EK_VarDecl;
    }
};

/// Function declaration.
class FunctionDecl : public ObjectDecl {
    /// Module that owns this function.
    property_r(Module*, module);

    /// The parent function. Null if this is the top-level function.
    property_r(FunctionDecl*, parent);

    /// The body of the function.
    property_rw(Expr*, body);

    /// The main scope of this function.
    property_rw(Scope*, scope);

    /// Parameter declarations.
    property_r(SmallVector<ParamDecl*>, params);

    /// Local variable declarations.
    property_r(SmallVector<VarDecl*>, locals);

    /// If the local variables are packed into a block for static
    /// chain passing, the type of that block.
    Type* _vars_block_type;

    /// Index of the static chain pointer of our parent function
    /// in our variable block, if any, or InvalidIndex otherwise.
    u32 _parent_chain_index = InvalidIndex;

    /// LLVM function.
    property_r(llvm::Function*, llvm);

public:
    FunctionDecl(
        std::string name,
        Expr* type,
        Module* mod,
        FunctionDecl* parent,
        SmallVector<ParamDecl*> param_decls,
        Linkage linkage = Linkage::Internal,
        Mangling mangling = Mangling::Source,
        Location loc = {}
    ) : ObjectDecl(EK_FunctionDecl, std::move(name), type, linkage, mangling, loc),
        module_field(mod),
        parent_field(parent),
        body_field(body),
        params_field(std::move(param_decls)),
        _vars_block_type(nullptr) {
        mod->add_function(this);
    }

    static inline constexpr u32 InvalidIndex = -1u;

    /// Add a local variable to this function.
    void add_variable(VarDecl* v) { locals.push_back(v); }

    /// Get a reference to this function.
    auto ref() const -> NameRefExpr*;

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_FunctionDecl; }
};

/// Local or global variable declaration.
class VarDecl : public ObjectDecl {
    /// Function that owns this declaration.
    property_r(FunctionDecl*, owner);

    /// Initialiser. If the type of this variable has constructors, these
    /// are the constructor arguments before Sema, and the constructor call
    /// after.
    property_r(Expr*, init);

    /// Get the module that owns this declaration.
    readonly(Module*, module, return owner->module);
public:
    VarDecl(
        std::string name,
        FunctionDecl* owner,
        Expr* type,
        Expr* init,
        Linkage linkage,
        Mangling mangling,
        Location loc = {}
    ) : ObjectDecl(EK_VarDecl, std::move(name), type, linkage, mangling, loc),
        owner_field(owner),
        init_field(init) {
        owner->add_variable(this);
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_VarDecl; }
};

/// Parameter declaration.
class ParamDecl : public Decl {
    /// Whether this is a static (= compile-time) parameter.
    bool _static : 1;

    /// Whether this is a with-parameter.
    bool _with : 1;

    /// The default value of this parameter, or, if the type of this
    /// parameter has constructors, the constructor arguments.
    Expr* _default;

public:
    ParamDecl(
        std::string name,
        Expr* type,
        bool is_static,
        bool is_with,
        Location loc = {}
    ) : Decl(EK_ParamDecl, std::move(name), type, loc),
        _static(is_static),
        _with(is_with),
        _default(nullptr) {}

    /// Get the default value.
    auto default_value() const -> Expr* { return _default; }

    /// Set the default value.
    auto default_value(Expr* e) -> void { _default = e; }

    /// Check if this is a static parameter.
    auto is_static() const -> bool { return _static; }

    /// Check if this is a with-parameter.
    auto is_with() const -> bool { return _with; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ParamDecl; }
};

/// Struct member declaration.
class MemberDecl : public Decl {
private:
    /// Offset of the member, in bits; used by `offsetof`.
    property_rw(u32, offset);

    /// Index of the member. Members may be non-consecutive due to padding.
    property_rw(u32, index);

    /// Check if this member is padding.
    readonly(bool, is_padding, return index == PaddingIndex);

public:
    MemberDecl(std::string name, Expr* type)
        : Decl(EK_MemberDecl, std::move(name), type),
          index_field(InvalidIndex) {}

    friend class RecordType;

    /// The index of a padding member.
    static inline constexpr u32 PaddingIndex = -2u;

    /// The index of an invalid member.
    static inline constexpr u32 InvalidIndex = -1u;

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_MemberDecl; }
};

/// Struct variant clause decl.
///
/// The type of a variant clause is void, iff
/// that clause is the void variant clause.
class VariantClauseDecl : public Decl {
    /// Index of the variant clause.
    property_r(u32, index);

    /// Whether this variant is dynamic.
    property_r(bool, dynamic);

public:
    VariantClauseDecl(std::string name, Expr* type, bool dynamic, Location loc)
        : Decl(EK_VariantClauseDecl, std::move(name), type, loc),
          dynamic_field(dynamic) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_VariantClauseDecl; }
};

/// Enumeration member declaration.
class EnumeratorDecl : public Decl {
    /// The value of this enumerator.
    property_r(Expr*, init);

public:
    EnumeratorDecl(std::string name, Expr* init, Location loc)
        : Decl(EK_EnumeratorDecl, std::move(name), detail::UnknownTy, loc) {}

    /// Get the value of this enumerator.
    readonly_decl(i64, value);

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_EnumeratorDecl; }
};

/// ===========================================================================
///  Types
/// ===========================================================================
class Type : public Expr {
    /// Builtin types are allocated here.
    static Type BuiltinTypes[18];

protected:
    Type(Kind k, Location loc = {}) : Expr(k, loc) {}

public:
    /// Builtin types.
    static Type* Unknown;
    static Type* UnknownValue;
    static Type* Void;
    static Type* TypeTy;
    static Type* Template;
    static Type* NoReturn;
    static Type* Int;
    static Type* F32;
    static Type* F64;

    /// Cached because we need it them all the time.
    static Type* VoidPtr;
    static Type* VoidRef;
    static Type* StringLiteral;
    static Type* Bool;
    static Type* I8;
    static Type* I16;
    static Type* I32;
    static Type* I64;

    /// FFI types.
    static Type* FFIChar;
    static Type* FFIChar8;
    static Type* FFIChar16;
    static Type* FFIChar32;
    static Type* FFIWChar;
    static Type* FFIShort;
    static Type* FFIInt;
    static Type* FFILong;
    static Type* FFILongLong;
    static Type* FFILongDouble;
    static Type* FFISize;

    /// Initialise types.
    static void Initialise();

    /// Get a string representation of this type.
    auto str() const -> std::string;

    /// Get the type stripped of qualifiers.
    auto unqualified() const -> Type*;

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= EK_EnumType && e->kind <= EK_AlignedType;
    }
};

/// Builtin type.
class BuiltinType : public Type {
    friend void Type::Initialise();

    /// No-one should ever have to create these.
    BuiltinType() : Type(EK_BuiltinType) {}

public:
    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_BuiltinType; }
};

/// FFI type.
///
/// This represents a type used for C FFI. Note that, unlike C, since
/// we don’t have unsigned numbers, both e.g. `int` and `unsigned int`
/// are represented by the same FFI type.
///
/// For portability, FFI types are preserved in module descriptions,
/// as well as in mangled names, and conversions between FFI types and
/// other integer types are only allowed if they are guaranteed to be
/// safe by the standard.
class FFIType : public Type {
    friend void Type::Initialise();

    /// No-one should ever have to create these.
    FFIType() : Type(EK_FFIType) {}

public:
    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_FFIType; }
};

/// Alternative type.
///
/// Used in type queries and restrictions, e.g. `int is 3 | 4`,
/// `some_struct is var1 | var2`.
class AlternativeType: public Type {
    /// The alternatives.
    property_r(SmallVector<Expr*>, alternatives);

public:
    AlternativeType(SmallVector<Expr*> alternatives, Location loc = {})
        : Type(EK_AlternativeType, loc), alternatives_field(std::move(alternatives)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_AlternativeType; }
};

/// Base class for types that only have a single element type.
class SingleElementTypeBase : public Type {
    /// The element type.
    Expr* _elem_type;

protected:
    SingleElementTypeBase(Kind k, Expr* elem_type, Location loc = {})
        : Type(k, loc), _elem_type(elem_type) {}

public:
    /// Get the element type.
    auto elem() const -> Expr* { return _elem_type; }
};

/// Scoped pointer type.
class ScopedPointerType : public SingleElementTypeBase {
public:
    ScopedPointerType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_ScopedPointerType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ScopedPointerType; }
};

/// Reference type.
class ReferenceType : public SingleElementTypeBase {
public:
    ReferenceType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_ReferenceType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ReferenceType; }
};

/// Optional type.
class OptionalType : public SingleElementTypeBase {
public:
    OptionalType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_OptionalType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_OptionalType; }
};

/// Slice type.
class SliceType : public SingleElementTypeBase {
public:
    SliceType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_SliceType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_SliceType; }
};

/// Inclusive range type.
class InclusiveRangeType : public SingleElementTypeBase {
public:
    InclusiveRangeType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_InclusiveRangeType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_InclusiveRangeType; }
};

/// Exclusive range type.
class ExclusiveRangeType : public SingleElementTypeBase {
public:
    ExclusiveRangeType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_ExclusiveRangeType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ExclusiveRangeType; }
};

/// Closure type.
class ClosureType : public SingleElementTypeBase {
public:
    ClosureType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_ClosureType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ClosureType; }
};

/// Value type.
///
/// This more or less corresponds to `const` in C++.
class ValueType : public SingleElementTypeBase {
public:
    ValueType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_ValueType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ValueType; }
};

/// Variable type.
///
/// The opposite of a value type. Typically a no-op,
/// but can be used to remove `val`-ness in casts.
class VariableType : public SingleElementTypeBase {
public:
    VariableType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_VariableType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_VariableType; }
};

/// Decltype decay type.
///
/// This is used to convert a type into a type suitable for use in
/// a declaration. We can’t just do this at parse time because it
/// might depend on a `typeof` or other late type information.
class DecltypeDecayType : public SingleElementTypeBase {
public:
    DecltypeDecayType(Expr* elem_type, Location loc = {}) : SingleElementTypeBase(EK_DeclTypeDecayType, elem_type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_DeclTypeDecayType; }
};

/// Typeof type.
class TypeofType : public SingleElementTypeBase {
public:
    TypeofType(Expr* expr, Location loc = {}) : SingleElementTypeBase(EK_TypeofType, expr, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_TypeofType; }
};

/// Array type.
class ArrayType : public SingleElementTypeBase {
    /// The dimension of the array.
    property_r(Expr*, dim);

public:
    ArrayType(Expr* elem_type, Expr* size, Location loc = {})
        : SingleElementTypeBase(EK_ArrayType, elem_type, loc),
          dim_field(size) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_ArrayType; }
};

/// Vector type.
class VectorType : public SingleElementTypeBase {
    /// The dimension of the vector.
    property_r(Expr*, dim);

public:
    VectorType(Expr* elem_type, Expr* size, Location loc = {})
        : SingleElementTypeBase(EK_VectorType, elem_type, loc), dim_field(size) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_VectorType; }
};

/// Integer type.
class IntegerType : public Type {
    /// The size of the integer, in bits.
    property_r(isz, bits);

public:
    IntegerType(isz size, Location loc = {}) : Type(EK_IntegerType, loc), bits_field(size) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_IntegerType; }
};

/// Type with additional alignment requirements.
class AlignedType : public SingleElementTypeBase {
    /// The alignment of the type, in bits.
    u32 _align;

public:
    AlignedType(Type* elem_type, u32 align, Location loc = {})
        : SingleElementTypeBase(EK_AlignedType, elem_type, loc), _align(align) {}

    /// Get the alignment of the typel, in bits.
    auto align() const -> u32 { return _align; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_AlignedType; }
};

/// Function type.
class FunctionType : public Type {
    /// The return type.
    Expr* _return_type;

    /// The parameter types.
    SmallVector<Expr*> _params;

    /// The calling convention.
    CallingConv _calling_conv;

    /// Whether the function is variadic.
    bool _variadic : 1;

    /// Whether this function takes a static chain parameter.
    bool _has_static_chain : 1;

public:
    /// TODO: Add `noundef` to all parameters when lowering to LLVM IR.
    FunctionType(
        SmallVector<Expr*> params,
        Expr* return_type,
        CallingConv calling_conv,
        bool variadic,
        bool has_static_chain
    ) : Type(EK_FunctionType),
        _return_type(return_type),
        _params(std::move(params)),
        _calling_conv(calling_conv),
        _variadic(variadic),
        _has_static_chain(has_static_chain) {}

    /// Get the calling convention.
    auto cc() const -> CallingConv { return _calling_conv; }

    /// Whether this function takes a static chain parameter.
    auto has_static_chain() const -> bool { return _has_static_chain; }

    /// Get the parameter types.
    auto params() const -> const SmallVector<Type*>& { return _params; }

    /// Get the return type.
    auto ret() const -> Type* { return _return_type; }

    /// Whether the function is variadic.
    auto variadic() const -> bool { return _variadic; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_FunctionType; }
};

/// Base class for types that have a name.
class NamedType : public Type {
    /// The module this type belongs to.
    property_r(Module*, module);

    /// The name of the type.
    property_r(std::string, name);

    /// The mangled name of this type is cached here.
    ///
    /// Note: If imported from a C header, the name should be mangled
    /// as if declared in the `.C` module.
    std::string mangled_name_cached{};
    readonly_decl(std::string, mangled_name);

protected:
    NamedType(Kind k, Module* module, std::string name, Location loc = {})
        : Type(k, loc),
          module_field(module),
          name_field(std::move(name)) {}

public:
    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= EK_EnumType and e->kind <= EK_StructTemplate;
    }
};

/// Opaque type.
class OpaqueType : public NamedType {
public:
    OpaqueType(Module* module, std::string name, Location loc)
        : NamedType(EK_OpaqueType, module, std::move(name), loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_OpaqueType; }
};

/// Base class for types that have members.
class RecordType : public NamedType {
public:
    using MemberDecls = SmallVector<MemberDecl*, 4>;

private:
    static bool IsPadding(MemberDecl* m) { return not m->is_padding(); }

protected:
    /// The member variables of this record.
    property_r(MemberDecls, members_and_padding);

    /// The size of this record, in bits.
    property_rw(u32, size);

    /// The alignment of this record, in bits.
    property_rw(u32, align);

    /// Get the size of this record in bits.
    readonly(u32, bits, return size);

    /// Get the the minimum number of bytes required to store this record.
    readonly(u32, bytes, return utils::AlignTo<u32>(align, 8) / 8);

    /// Get all members of this record that aren’t padding.
    readonly(
        decltype(std::views::filter(members_and_padding, IsPadding)),
        members,
        return std::views::filter(members_and_padding, IsPadding)
    );

public:
    RecordType(
        Kind k,
        Module* module,
        std::string name,
        MemberDecls members,
        Location loc = {}
    ) : NamedType(k, module, std::move(name), loc),
        members_and_padding_field(std::move(members)) {}
};

/// Tuple type.
class TupleType : public RecordType {
public:
    TupleType(Module* module, RecordType::MemberDecls members, std::string name)
        : RecordType(EK_TupleType, module, std::move(name), std::move(members)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_TupleType; }
};

/// Struct type.
class StructType : public RecordType {
    /// Our parent struct.
    ///
    /// If this is an instance of a template, then this is the parent
    /// template; otherwise, this is null.
    property_rw(StructType*, parent);

    /// The scope of the struct body.
    property_rw(Scope*, scope);

    /// The variants of this struct.
    property_r(SmallVector<VariantClauseDecl*>, variants);

    /// The member functions of this record.
    property_r(SmallVector<FunctionDecl*>, member_functions);

    /// Initialiser functions for this struct.
    property_r(SmallVector<FunctionDecl*>, inits);

    /// Default initialiser.
    property_r(FunctionDecl*, default_init);

    /// Type of the variant storage for this type.
    property_rw(Type*, variant_storage_type);

    /// Type of the variant index for this type.
    property_rw(Type*, variant_index_type);

    /// Field index of the variant index.
    property_rw(u32, variant_index_field);

    /// Field index of the variant storage.
    property_rw(u32, variant_storage_field);

    /// Whether this is a packed struct.
    property_rw(bool, packed);

    /// Whether this type has a void variant.
    readonly_decl(bool, has_void_variant);

public:
    StructType(
        Module* module,
        std::string name,
        Scope* scope,
        RecordType::MemberDecls members,
        SmallVector<FunctionDecl*> member_functions,
        SmallVector<VariantClauseDecl*> variants,
        bool packed,
        Location loc = {}
    ) : RecordType(EK_StructType, module, std::move(name), std::move(members), loc),
        parent_field(nullptr),
        scope_field(scope),
        variants_field(std::move(variants)),
        member_functions_field(std::move(member_functions)),
        packed_field(packed) {}

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind == EK_StructType or e->kind == EK_StructTemplate;
    }
};

/// ===========================================================================
///  Templates
/// ===========================================================================
class StructTemplate : public StructType {
public:
    /// Instances of this template.
    struct Instance {
        SmallVector<EvalResult> argument_values;
        StructType* type;
    };

private:
    /// The template parameters.
    property_r(SmallVector<ParamDecl*>, template_params);

    /// The instances of this template.
    property_r(SmallVector<Instance>, instances);

public:
    StructTemplate(
        Module* module,
        std::string name,
        Scope* scope,
        RecordType::MemberDecls members,
        SmallVector<FunctionDecl*> member_functions,
        SmallVector<VariantClauseDecl*> variants,
        SmallVector<ParamDecl*> template_params,
        bool packed,
        Location loc = {}
    ) : StructType(module, std::move(name), scope, std::move(members), std::move(member_functions), std::move(variants), packed, loc),
        template_params_field(std::move(template_params)) {}

    /// Instantiate this template with the given arguments.
    auto instantiate(SmallVector<EvalResult> args) -> StructType*;

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_StructTemplate; }
};

/// Enumeration type.
class EnumType : public NamedType {
    /// Underlying integetr type.
    property_r(Expr*, underlying_type);

    /// The enumerators that are part of this enumeration.
    property_r(SmallVector<EnumeratorDecl*>, enumerators);

public:
    EnumType(
        Module* module,
        std::string name,
        SmallVector<EnumeratorDecl*> enumerators,
        Expr* type,
        Location loc = {}
    ) : NamedType(EK_EnumType, module, std::move(name), loc),
        underlying_type_field(type),
        enumerators_field(std::move(enumerators)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == EK_EnumType; }
};

/// ===========================================================================
///  Scopes
/// ===========================================================================
class Scope {
    /// The parent scope.
    property_r(Scope*, parent);

    /// The module this scope belongs to.
    property_r(Module*, module);

    /// Symbols in this scope.
    property_r(StringMap<Expr*>, symbol_table);

    /// Whether this scope is a function scope.
    property_r(bool, is_function);


public:
    /// Get the nearest parent scope that is a function scope.
    readonly_decl(Scope*, enclosing_function_scope);

    Scope(const Scope& other) = delete;
    Scope& operator=(const Scope& other) = delete;

    /// Create a new scope.
    explicit Scope(Scope* parent, Module* mod);

    /// Declare a symbol in this scope.
    auto declare(std::string name, Expr* value) -> Result<Expr*>;

    /// Mark this scope as a function scope. This cannot be undone.
    void set_function_scope() {
        Assert(not is_function, "Scope already marked as function scope");
        is_function_field = true;
    }

    /// Visit each symbol with the given name.
    template <typename Func>
    void visit(const std::string& name, Func f, bool this_scope_only) {
        for (auto& [_, expr] : symbol_table) {
            if (auto bundle = cast<ExprBundle>(expr)) {
                for (auto e : *bundle)
                    f(e);
            } else {
                f(expr);
            }
        }

        if (parent and not this_scope_only) parent->visit(name, f, false);
    }
};
} // namespace src
#endif // SOURCE_INCLUDE_FRONTEND_AST_HH
