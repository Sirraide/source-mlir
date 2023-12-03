#ifndef SOURCE_FRONTEND_AST_HH
#define SOURCE_FRONTEND_AST_HH

#include <source/Core.hh>
#include <source/Frontend/Lexer.hh>
#include <source/Support/Result.hh>

namespace src {
class Expr;
class Type;
class FunctionDecl;
class LocalDecl;
class StructType;
class ProcType;
class BlockExpr;
class Nil;

namespace detail {
extern Expr* const UnknownType;
}

class EvalResult {
    std::variant<std::monostate, APInt, Expr*, std::nullptr_t> value{};

public:
    Expr* type = detail::UnknownType;

    EvalResult() : value(std::monostate{}) {}
    EvalResult(std::nullptr_t); /// Nil.
    EvalResult(APInt value, Expr* type) : value(std::move(value)), type(type) {}
    EvalResult(Expr* type) : value(type), type(type) {}

    auto as_int() -> APInt& { return std::get<APInt>(value); }
    auto as_type() -> Expr* { return std::get<Expr*>(value); }

    bool is_int() const { return std::holds_alternative<APInt>(value); }
    bool is_type() const { return std::holds_alternative<Expr*>(value); }
};

/// ===========================================================================
///  Enums
/// ===========================================================================
enum struct Linkage {
    Local,       ///< Local variable.
    Internal,    ///< Not exported and defined.
    Imported,    ///< Imported from another module or library.
    Exported,    ///< Exported and defined.
    Reexported,  ///< Imported and exported, and thus not defined.
    LinkOnceODR, ///< Merge definitions across different TUs. Used mainly for compiler-generated code.
};

enum struct Mangling {
    None,   ///< Do not mangle.
    Source, ///< Use Source mangling.
};

enum struct Builtin {
    Destroy,
    New,
};

/// ===========================================================================
///  Special Expressions
/// ===========================================================================
class Expr {
public:
    enum struct Kind {
        /// Type [begin]
        BuiltinType,
        FFIType,
        StructType,
        IntType,
        Nil,
        ProcType,

        /// SingleElementType [begin]
        ReferenceType,
        ScopedPointerType,
        SliceType,
        ArrayType,
        OptionalType,
        SugaredType,
        ScopedType,
        ClosureType,
        /// SingleElementType [end]
        /// Type [end]

        /// Loop [begin]
        WhileExpr,
        ForInExpr,
        /// Loop [end]

        AssertExpr,
        DeferExpr,
        ExportExpr,
        LabelExpr,
        EmptyExpr,
        ModuleRefExpr,
        OverloadSetExpr,

        /// UnwindExpr [begin]
        ReturnExpr,
        GotoExpr,
        LoopControlExpr,
        /// UnwindExpr [end]

        /// TypedExpr [begin]
        BlockExpr,
        ImplicitThisExpr,
        InvokeExpr,
        InvokeBuiltinExpr,
        ConstExpr,
        CastExpr,
        MemberAccessExpr,
        ScopeAccessExpr,
        UnaryPrefixExpr,
        IfExpr,
        BinaryExpr,
        DeclRefExpr,
        LocalRefExpr,
        ParenExpr,
        SubscriptExpr,
        BoolLiteralExpr,
        IntegerLiteralExpr,
        StringLiteralExpr,

        /// Decl [begin]
        LocalDecl,

        /// ObjectDecl [begin]
        ProcDecl,
    };

    class SemaState {
        enum struct St {
            NotAnalysed,
            InProgress,
            Errored,
            Ok,
        };

        St state = St::NotAnalysed;

    public:
        readonly_const(bool, analysed, return state == St::Errored or state == St::Ok);
        readonly_const(bool, errored, return state == St::Errored);
        readonly_const(bool, in_progress, return state == St::InProgress);
        readonly_const(bool, ok, return state == St::Ok);

        void set_done() {
            if (state != St::Errored) state = St::Ok;
        }

        /// Returns false for convenience.
        bool set_errored() {
            state = St::Errored;
            return false;
        }

        void set_in_progress() {
            if (state == St::NotAnalysed) state = St::InProgress;
        }

        /// Reset the state.
        void unset() { state = St::NotAnalysed; }
    };

    /// Helper type that makes accessing properties of types easier.
    class TypeHandle {
        Expr* ptr;

    public:
        TypeHandle(Expr* ptr) : ptr(ptr) {}
        TypeHandle(std::nullptr_t) = delete;

        /// Get the alignment of this type, in *bytes*.
        auto align(Context* ctx) -> Align;

        /// Get the procedure type from a closure or proc.
        readonly_decl(ProcType*, callable);

        /// Check whether this type is default constructible.
        readonly_decl(bool, default_constructible);

        /// Get the type stripped of any sugar.
        readonly_decl(TypeHandle, desugared);

        /// Get the mangled name of this type.
        ///
        /// Context may be null if this is a struct type.
        readonly_decl(std::string, mangled_name);

        /// Check if this is any integer type.
        bool is_int(bool bool_is_int);

        /// Check if this is 'nil'.
        readonly_decl(bool, is_nil);

        /// Check if this is the builtin 'noreturn' type.
        readonly_decl(bool, is_noreturn);

        /// Get the number of nested reference levels in this type.
        readonly_decl(isz, ref_depth);

        /// Get the size of this type.
        auto size(Context* ctx) -> Size;

        /// Get a string representation of this type.
        auto str(bool use_colour) const -> std::string;

        /// Strip all references from this type.
        readonly_decl(TypeHandle, strip_refs);

        /// Strip all references and pointers from this type.
        readonly_decl(TypeHandle, strip_refs_and_pointers);

        /// Check if this type logically yields a value, i.e. is not
        /// void or noreturn.
        readonly_decl(bool, yields_value);

        /// Access the underlying type pointer.
        operator Expr*() const { return ptr; };
        Expr* operator->() { return ptr; }
    };

    /// The kind of this expression.
    const Kind kind;

    /// The location of this expression.
    Location location;

    /// State of semantic analysis
    SemaState sema{};

    /// The MLIR value of this expression.
    SOURCE_MLIR_VALUE_MEMBER(mlir);

    /// Protected subexpressions (i.e. defers and local variables).
    /// Only applicable to full expressions at the block level. This
    /// may also contain the expression itself if it is protected.
    SmallVector<Expr*, 1> protected_children{};

    /// Whether this expression has already been emitted.
    bool emitted : 1 = false;

    /// Check if this is an lvalue.
    bool is_lvalue : 1 = false;

public:
    Expr(Kind k, Location loc) : kind(k), location(loc) {}
    virtual ~Expr() = default;

    /// Only allow allocating nodes in a module.
    void* operator new(size_t) = delete;
    void* operator new(size_t sz, Module* mod) noexcept {
        return utils::AllocateAndRegister<Expr>(sz, mod->exprs);
    }

    /// Get this expression as a type handle; this is different
    /// from `type` which returns a handle to the type of this
    /// expression.
    readonly(TypeHandle, as_type, return TypeHandle(this));

    /// Strip lvalue-to-rvalue conversion. This only removes one
    /// level of lvalue-to-rvalue conversion, not lvalue-ref-to-lvalue
    /// conversion.
    readonly_decl(Expr*, ignore_lv2rv);

    /// Strip parentheses, implicit casts, and DeclRefExprs.
    readonly_decl(Expr*, ignore_paren_cast_refs);

    /// Strip parentheses, and DeclRefExprs.
    readonly_decl(Expr*, ignore_paren_refs);

    /// Whether this is an optional that is known to be active in
    /// the current scope.
    readonly_decl(bool, is_active_optional);

    /// Check if this is 'nil'.
    readonly_decl(bool, is_nil);

    /// Get a string representation of the name of the scope of
    /// this expression, if it has one.
    readonly_decl(std::string, scope_name);

    /// Get the type of this expression; returns void if
    /// this expression has no type.
    readonly_decl(TypeHandle, type);

    /// Get the type of this when unwrapped. This will strip an
    /// optional type iff the expression is known to be active.
    readonly_decl(TypeHandle, unwrapped_type);

    /// Print this expression to stdout.
    void print(bool print_children = true) const;
};

class AssertExpr : public Expr {
public:
    /// The condition of this assertion.
    Expr* cond;

    /// The optional message of this assertion.
    Expr* msg;

    /// Condition and filename strings.
    Expr* cond_str{};
    Expr* file_str{};

    AssertExpr(Expr* cond, Expr* msg, Location loc)
        : Expr(Kind::AssertExpr, loc),
          cond(cond),
          msg(msg) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::AssertExpr; }
};

class DeferExpr : public Expr {
public:
    /// The deferred expression.
    Expr* expr;

    DeferExpr(Expr* expr, Location loc)
        : Expr(Kind::DeferExpr, loc),
          expr(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::DeferExpr; }
};

/// Base class for loops.
class Loop : public Expr {
public:
    /// The body of this while loop.
    BlockExpr* body;

    /// For the backend.
    mlir::Block* continue_block{};
    mlir::Block* break_block{};

    Loop(Kind k, BlockExpr* body, Location loc)
        : Expr(k, loc),
          body(body) {}

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::WhileExpr and e->kind <= Kind::ForInExpr;
    }
};

class WhileExpr : public Loop {
public:
    /// The condition of this while loop.
    Expr* cond;

    WhileExpr(Expr* cond, BlockExpr* body, Location loc)
        : Loop(Kind::WhileExpr, body, loc),
          cond(cond) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::WhileExpr; }
};

class ForInExpr : public Loop {
public:
    /// The iteration variable of this for-in loop.
    LocalDecl* iter;

    /// The range that we’re iterating over.
    Expr* range;

    /// Whether we’re iterating in reverse.
    bool reverse;

    ForInExpr(LocalDecl* iter, Expr* range, BlockExpr* body, bool reverse, Location loc)
        : Loop(Kind::ForInExpr, body, loc),
          iter(iter),
          range(range),
          reverse(reverse) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ForInExpr; }
};

/// Base class for expressions that unwind the stack.
class UnwindExpr : public Expr {
public:
    /// Expressions to unwind.
    SmallVector<Expr*> unwind{};

    /// Pointer to parent full expression. Points to this
    /// if this is a full expression.
    Expr* parent_full_expression{};

    UnwindExpr(Kind k, Location loc) : Expr(k, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::ReturnExpr and e->kind <= Kind::LoopControlExpr;
    }
};

class ReturnExpr : public UnwindExpr {
public:
    /// The value being returned.
    Expr* value;

    ReturnExpr(Expr* value, Location loc)
        : UnwindExpr(Kind::ReturnExpr, loc),
          value(value) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ReturnExpr; }
};

class LoopControlExpr : public UnwindExpr {
public:
    /// Label to jump to.
    std::string label;

    /// Resolved expression. This is set to the parent
    /// if there is no label. This is resolved in Sema.
    Loop* target{};

    /// Whether this is a continue or break.
    bool is_continue;
    readonly(bool, is_break, return not is_continue);

    LoopControlExpr(std::string label, bool is_continue, Location loc)
        : UnwindExpr(Kind::LoopControlExpr, loc),
          label(std::move(label)),
          is_continue(is_continue) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LoopControlExpr; }
};

class LabelExpr : public Expr {
public:
    /// The label of this expression.
    std::string label;

    /// The expression labelled by this label.
    Expr* expr;

    /// Parent scope. This is required for forward gotos.
    BlockExpr* parent{};

    /// Block that is represented by this label.
    mlir::Block* block{};

    /// Pointer to parent full expression. Points to this
    /// if this is a full expression.
    Expr* parent_full_expression{};

    /// Whether this label is ever branched to.
    bool used = false;

    LabelExpr(
        ProcDecl* in_procedure,
        std::string label,
        Expr* expr,
        Location loc
    );

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LabelExpr; }
};

class GotoExpr : public UnwindExpr {
public:
    /// The label to jump to.
    std::string label;

    /// The resolved labelled expression.
    LabelExpr* target{};

    GotoExpr(std::string label, Location loc)
        : UnwindExpr(Kind::GotoExpr, loc),
          label(std::move(label)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::GotoExpr; }
};

class EmptyExpr : public Expr {
public:
    EmptyExpr(Location loc)
        : Expr(Kind::EmptyExpr, loc) {
        sema.set_done();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::EmptyExpr; }
};

class ExportExpr : public Expr {
public:
    /// The expression being exported.
    Expr* expr;

    ExportExpr(Expr* expr, Location loc)
        : Expr(Kind::ExportExpr, loc),
          expr(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ExportExpr; }
};

class ModuleRefExpr : public Expr {
public:
    /// The module being referenced.
    Module* module;

    ModuleRefExpr(Module* module, Location loc)
        : Expr(Kind::ModuleRefExpr, loc),
          module(module) {
        sema.set_done();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ModuleRefExpr; }
};

class OverloadSetExpr : public Expr {
public:
    /// The overloads.
    SmallVector<ProcDecl*> overloads;

    OverloadSetExpr(SmallVector<ProcDecl*> overloads, Location loc)
        : Expr(Kind::OverloadSetExpr, loc),
          overloads(std::move(overloads)) {
        sema.set_done(); /// Only constructed by sema.
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::OverloadSetExpr; }
};

/// ===========================================================================
///  Typed Expressions
/// ===========================================================================
enum struct CastKind {
    /// Convert lvalues to rvalues.
    LValueToRValue,

    /// Convert an reference rvalue to an lvalue of the referenced type.
    ReferenceToLValue,

    /// Convert an lvalue to a reference rvalue.
    LValueToReference,

    /// Convert an reference lvalue to an lvalue of the referenced type. Same
    /// as performing LValueToRValue and then ReferenceToLValue.
    LValueRefToLValue,

    /// Test if an optional is nil, yielding an rvalue bool.
    OptionalNilTest,

    /// Unwrap an optional. This yields an lvalue of the contained type
    /// iff the optional is itself an lvalue.
    OptionalUnwrap,

    /// Wrap a value, creating an optional. This always yields an rvalue.
    OptionalWrap,

    /// Convert an array lvalue to a reference rvalue to the first element.
    ArrayToElemRef,

    /// Any other implicit conversion.
    Implicit,

    /// 'as' cast.
    Soft,

    /// 'as!' cast.
    Hard,
};

class TypedExpr : public Expr {
public:
    /// 'type' is already a member of Expr, so don’t use that here.
    Expr* stored_type;

    TypedExpr(Kind k, Expr* type, Location loc)
        : Expr(k, loc), stored_type(type) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind >= Kind::BlockExpr; }
};

enum struct ScopeKind : u8 {
    Block,
    Function,
    Struct,
};

class BlockExpr : public TypedExpr {
public:
    using Symbols = SmallVector<Expr*, 1>;

    /// The expressions that are part of this block.
    SmallVector<Expr*> exprs;

    /// The parent scope.
    BlockExpr* parent;

    /// The module this scope belongs to.
    Module* module;

    /// Symbols in this scope.
    StringMap<Symbols> symbol_table;

    /// Pointer to parent full expression. Points to this
    /// if this is a full expression.
    Expr* parent_full_expression{};

    /// Expressions to unwind at end of scope.
    SmallVector<Expr*> unwind{};

    /// Associated scope op.
    mlir::Operation* scope_op{};

    /// Get the nearest parent scope that is a function scope.
    readonly_decl(BlockExpr*, enclosing_function_scope);

private:
    /// What kind of scope this is.
    ScopeKind scope_kind = ScopeKind::Block;

public:
    /// Whether this is a function scope.
    readonly(bool, is_function, return scope_kind == ScopeKind::Function);

    /// Whether this is a struct scope.
    readonly(bool, is_struct, return scope_kind == ScopeKind::Struct);

    /// Whether this expression was create implicitly.
    bool implicit{};

    BlockExpr(Module* mod, BlockExpr* parent, Location loc = {}, bool implicit = false)
        : TypedExpr(Kind::BlockExpr, detail::UnknownType, loc),
          parent(parent),
          module(mod),
          implicit(implicit) {}

    /// Declare a symbol in this scope.
    void declare(StringRef name, Expr* value) {
        symbol_table[name].push_back(value);
    }

    /// Find a (vector of) symbol(s) in this scope.
    auto find(StringRef name, bool this_scope_only) -> Symbols* {
        if (auto sym = symbol_table.find(name); sym != symbol_table.end())
            return &sym->second;
        if (parent and not this_scope_only) return parent->find(name, false);
        return nullptr;
    }

    /// Mark this scope as a function scope. This cannot be undone.
    void set_function_scope() {
        Assert(scope_kind == ScopeKind::Block);
        scope_kind = ScopeKind::Function;
    }

    /// Mark this scope as a struct scope. This cannot be undone.
    void set_struct_scope() {
        Assert(scope_kind == ScopeKind::Block);
        scope_kind = ScopeKind::Struct;
    }

    /// Find the NCA of two blocks in a function. Returns nullptr
    /// if the blocks are not in the same function.
    static auto NCAInFunction(BlockExpr* a, BlockExpr* b) -> BlockExpr*;

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BlockExpr; }
};

class InvokeExpr : public TypedExpr {
public:
    /// The arguments to the function.
    SmallVector<Expr*> args;

    /// Whether this is a naked invocation.
    bool naked;

    /// The function being invoked.
    Expr* callee;

    /// Bound initialisers arguments. These are any arguments
    /// after a `=` in an invoke expression; they are stored
    /// here to facilitate conversion to a variable declaration
    /// in sema.
    SmallVector<Expr*> init_args;

    InvokeExpr(Expr* callee, SmallVector<Expr*> args, bool naked, SmallVector<Expr*> init, Location loc)
        : TypedExpr(Kind::InvokeExpr, detail::UnknownType, loc),
          args(std::move(args)),
          naked(naked),
          callee(callee),
          init_args(std::move(init)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::InvokeExpr; }
};

class InvokeBuiltinExpr : public TypedExpr {
public:
    /// The arguments to the builtin.
    SmallVector<Expr*> args;

    /// The builtin being invoked.
    Builtin builtin;

    InvokeBuiltinExpr(Builtin builtin, SmallVector<Expr*> args, Location loc)
        : TypedExpr(Kind::InvokeBuiltinExpr, detail::UnknownType, loc),
          args(std::move(args)),
          builtin(builtin) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::InvokeBuiltinExpr; }
};

class ConstExpr : public TypedExpr {
public:
    /// The underlying expression. May be null if this was constructed by sema.
    Expr* expr{};

    /// Cached result.
    EvalResult value;

    ConstExpr(Expr* expr, EvalResult&& cached, Location loc)
        : TypedExpr(Kind::ConstExpr, cached.type, loc),
          expr(expr),
          value(std::move(cached)) {
        sema.set_done();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ConstExpr; }
};

class CastExpr : public TypedExpr {
public:
    /// The kind of this cast.
    CastKind cast_kind;

    /// The expression being cast.
    Expr* operand;

    CastExpr(CastKind kind, Expr* expr, Expr* type, Location loc)
        : TypedExpr(Kind::CastExpr, type, loc),
          cast_kind(kind),
          operand(expr) {}

    /// Whether this is a cast that can perform conversions between
    /// types rather than just e.g. LValueToRValue conversion.
    readonly(
        bool,
        is_converting_cast,
        return cast_kind == CastKind::Implicit or
               cast_kind == CastKind::Soft or
               cast_kind == CastKind::Hard
    );

    /// Whether this is an optional test cast.
    readonly(
        bool,
        is_opt_test,
        return cast_kind == CastKind::OptionalNilTest
    );

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::CastExpr; }
};

class UnaryPrefixExpr : public TypedExpr {
public:
    /// The operand of this unary expression.
    Expr* operand;

    /// The operator of this unary expression.
    Tk op;

    UnaryPrefixExpr(Tk op, Expr* operand, Location loc)
        : TypedExpr(Kind::UnaryPrefixExpr, detail::UnknownType, loc),
          operand(operand),
          op(op) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::UnaryPrefixExpr; }
};

class IfExpr : public TypedExpr {
public:
    /// The condition of this if expression.
    Expr* cond;

    /// The then branch of this if expression.
    Expr* then;

    /// The optional else branch of this if expression.
    Expr* else_;

    IfExpr(Expr* cond, Expr* then, Expr* else_, Location loc)
        : TypedExpr(Kind::IfExpr, detail::UnknownType, loc),
          cond(cond),
          then(then),
          else_(else_) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IfExpr; }
};

class BinaryExpr : public TypedExpr {
public:
    /// The left-hand side of this binary expression.
    Expr* lhs;

    /// The right-hand side of this binary expression.
    Expr* rhs;

    /// The operator of this binary expression.
    Tk op;

    BinaryExpr(Tk op, Expr* lhs, Expr* rhs, Location loc)
        : TypedExpr(Kind::BinaryExpr, detail::UnknownType, loc),
          lhs(lhs),
          rhs(rhs),
          op(op) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BinaryExpr; }
};

class ScopeAccessExpr : public TypedExpr {
public:
    /// The name of the element being accessed.
    std::string element;

    /// The object being accessed.
    Expr* object;

    /// The resolved reference.
    Expr* resolved{};

    ScopeAccessExpr(Expr* object, std::string element, Location loc)
        : TypedExpr(Kind::ScopeAccessExpr, detail::UnknownType, loc),
          element(std::move(element)),
          object(object) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ScopeAccessExpr; }
};

class DeclRefExpr : public TypedExpr {
public:
    /// The name of the declaration this refers to.
    std::string name;

    /// The scope in which this name was found. May be null
    /// iff `ok` returns true (that is, if the scope is no
    /// longer needed).
    BlockExpr* scope;

    /// The declaration this refers to.
    Expr* decl;

    DeclRefExpr(std::string name, BlockExpr* sc, Location loc)
        : TypedExpr(Kind::DeclRefExpr, detail::UnknownType, loc),
          name(std::move(name)),
          scope(sc),
          decl(nullptr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::DeclRefExpr; }
};

/// DeclRefExpr that references a LocalDecl, possibly in
/// a parent function.
class LocalRefExpr : public TypedExpr {
public:
    /// Procedure containing the reference.
    ProcDecl* parent;

    /// The declaration this refers to.
    LocalDecl* decl;

    LocalRefExpr(ProcDecl* parent, LocalDecl* decl, Location loc);

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LocalRefExpr; }
};

class ParenExpr : public TypedExpr {
public:
    /// The expression inside the parentheses.
    Expr* expr;

    ParenExpr(Expr* expr, Location loc)
        : TypedExpr(Kind::ParenExpr, detail::UnknownType, loc),
          expr(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ParenExpr; }
};

class SubscriptExpr : public TypedExpr {
public:
    Expr* object;
    Expr* index;

    SubscriptExpr(Expr* object, Expr* index, Location loc)
        : TypedExpr(Kind::SubscriptExpr, detail::UnknownType, loc),
          object(object),
          index(index) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SubscriptExpr; }
};

class IntLitExpr : public TypedExpr {
public:
    /// The value of this literal.
    APInt value;

    IntLitExpr(APInt value, Location loc)
        : TypedExpr(Kind::IntegerLiteralExpr, detail::UnknownType, loc),
          value(std::move(value)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IntegerLiteralExpr; }
};

class BoolLitExpr : public TypedExpr {
public:
    /// The value of this literal.
    bool value;

    BoolLitExpr(bool value, Location loc)
        : TypedExpr(Kind::BoolLiteralExpr, detail::UnknownType, loc),
          value(value) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BoolLiteralExpr; }
};

class StrLitExpr : public TypedExpr {
public:
    /// The index of this literal in the string table.
    u32 index;

    StrLitExpr(u32 index, Location loc)
        : TypedExpr(Kind::StringLiteralExpr, detail::UnknownType, loc),
          index(index) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::StringLiteralExpr; }
};

/// ===========================================================================
///  Declarations
/// ===========================================================================
class Decl : public TypedExpr {
public:
    /// The name of this declaration.
    std::string name;

    Decl(Kind k, std::string name, Expr* type, Location loc)
        : TypedExpr(k, type, loc), name(std::move(name)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind >= Kind::LocalDecl; }
};

class ObjectDecl : public Decl {
    /// Mangled name.
    std::string stored_mangled_name;

public:
    /// The module this declaration belongs to.
    Module* module;

    /// Linkage of this object.
    Linkage linkage;

    /// Mangling scheme.
    Mangling mangling;

    /// Get the mangled name of this object.
    readonly_decl(StringRef, mangled_name);

    /// Whether this decl is imported or exported.
    readonly(bool, imported, return linkage == Linkage::Imported or linkage == Linkage::Reexported);
    readonly(bool, exported, return linkage == Linkage::Exported or linkage == Linkage::Reexported);

    ObjectDecl(
        Kind k,
        Module* mod,
        std::string name,
        Expr* type,
        Linkage linkage,
        Mangling mangling,
        Location loc
    ) : Decl(k, std::move(name), type, loc),
        module(mod),
        linkage(linkage),
        mangling(mangling) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind >= Kind::ProcDecl; }
};

/// Type of construction of a variable.
class Constructor {
public:
    enum struct Kind {
        Invalid,
        Uninitialised,
        MoveParameter,
        Zeroinit,
        TrivialCopy,
        SliceFromParts,
        InitialiserCall,
    } k;

private:
    /// The initialiser to call, if any.
    ProcDecl* init;

    Constructor(Kind k, ProcDecl* init = nullptr) : k(k), init(init) {}

public:
    readonly_const(Kind, kind, return k);
    readonly_const(ProcDecl*, initialiser, return init);

    static auto InitialiserCall(ProcDecl* init) -> Constructor { return {Kind::InitialiserCall, init}; }
    static auto MoveParameter() -> Constructor { return {Kind::MoveParameter}; }
    static auto SliceFromParts() -> Constructor { return {Kind::SliceFromParts}; }
    static auto TrivialCopy() -> Constructor { return {Kind::TrivialCopy}; }
    static auto Uninitialised() -> Constructor { return {Kind::Uninitialised}; }
    static auto Unset() -> Constructor { return {Kind::Invalid}; }
    static auto Zeroinit() -> Constructor { return {Kind::Zeroinit}; }
};

enum struct LocalKind {
    Variable,    /// Regular stack variable.
    Parameter,   /// Procedure parameter.
    Synthesised, /// Named lvalue that points to an object somewhere else.
};

/// Local variable declaration.
class LocalDecl : public Decl {
    bool is_captured = false;
    bool deleted = false;

public:
    /// The procedure containing this declaration.
    ProcDecl* parent;

    /// The initialiser arguments, if any.
    SmallVector<Expr*> init_args;

    /// Constructor(s) that should be invoked, if any.
    Constructor ctor = Constructor::Unset();

    /// Index in capture list of parent procedure, if any.
    isz capture_index{};

    /// What kind of variable this is.
    LocalKind local_kind;

    /// If this is a variable of optional type, whether a value
    /// is currently present.
    bool has_value = false;

    /// Whether this declaration is captured.
    readonly(bool, captured, return is_captured);

    /// Whether this variable is ever deleted or moved from.
    readonly(bool, deleted_or_moved, return deleted);

    /// Some variables (e.g. the variable of a for-in loop) cannot
    /// be captured since they do not correspond to a stack variable
    /// (and capturing them would also not be desirable as that is
    /// a common bug in loops).
    readonly_const(
        bool,
        is_legal_to_capture,
        return  local_kind != LocalKind::Synthesised
    );

    LocalDecl(
        ProcDecl* parent,
        std::string name,
        Expr* type,
        SmallVector<Expr*> init,
        LocalKind kind,
        Location loc
    ) : Decl(Kind::LocalDecl, std::move(name), type, loc),
        parent(parent),
        init_args(std::move(init)),
        local_kind(kind) {
    }

    /// Mark this declaration as captured.
    void set_captured();

    /// Mark that this declaration is deleted or moved from.
    void set_deleted_or_moved() { deleted = true; }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LocalDecl; }
};

/*
/// Global variable decalration.
class GlobalDecl : public ObjectDecl {
public:

    /// The initialiser.
    Expr* init;

    /// Linkage and mangling are ignored for e.g. struct fields.
    /// TODO: Maybe use a different FieldDecl class for that?
    GlobalDecl(
        ProcDecl* parent,
        std::string name,
        Expr* type,
        Expr* init,
        Linkage linkage,
        Mangling mangling,
        Location loc
    ) : ObjectDecl(Kind::LocalDecl, std::move(name), type, linkage, mangling, loc),
        parent(parent),
        init(init) {
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LocalDecl; }
};
*/

class ProcDecl : public ObjectDecl {
public:
    /// The parent function. Null if this is the top-level function. This
    /// is used for building static chains.
    ProcDecl* parent;

    /// The function parameter decls. Empty if this is a declaration.
    SmallVector<LocalDecl*> params;

    /// Body of the function.
    BlockExpr* body;

    /// Labels are global per procedure.
    StringMap<LabelExpr*> labels;

    /// Captured variables.
    SmallVector<LocalDecl*> captured_locals;

    /// Type of the struct containing the captured variables.
    StructType* captured_locals_type{};

    /// LocalVar holding the captured variables.
    SOURCE_MLIR_VALUE_MEMBER(captured_locals_ptr);

    /// MLIR function.
    mlir::Operation* mlir_func{};

    ProcDecl(
        Module* mod,
        ProcDecl* parent,
        std::string name,
        Expr* type,
        SmallVector<LocalDecl*> params,
        Linkage linkage,
        Mangling mangling,
        Location loc
    );

    /// Add a labelled expression to the function. This is done
    /// at parse time so all labels are available in sema.
    ///
    /// \param label The label to register.
    /// \param expr The expression that the label points at.
    /// \return The expression, or an error.
    auto add_label(std::string label, LabelExpr* expr) {
        if (labels.contains(label)) Diag::Error(
            module->context,
            expr->location,
            "Label '{}' is already defined",
            label
        );

        labels[label] = expr;
    }

    /// Whether this is a nested procedure.
    readonly(bool, nested, return parent and parent->parent != nullptr);

    /// Whether this takes a static chain pointer parameter.
    readonly_decl(bool, takes_static_chain);

    /// Get the return type of this procedure.
    [[gnu::pure]] readonly_decl(TypeHandle, ret_type);

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ProcDecl; }
};

/// Only generated by Sema for accessing the implicit `this` parameter of initialisers.
class ImplicitThisExpr : public TypedExpr {
public:
    /// The initialiser whose `this` this is.
    ProcDecl* init;

    ImplicitThisExpr(ProcDecl* init, Expr* type, Location loc)
        : TypedExpr(Kind::ImplicitThisExpr, type, loc),
          init(init) {
        sema.set_done();
        is_lvalue = true;
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ImplicitThisExpr; }
};

/// ===========================================================================
///  Types
/// ===========================================================================
class BuiltinType;
class IntType;
class ReferenceType;
class FFIType;

enum struct BuiltinTypeKind {
    Unknown,
    Void,
    Int,
    Bool,
    NoReturn,
    OverloadSet,
};

enum struct FFITypeKind {
    CChar,
    CInt,
};

class Type : public Expr {
protected:
    Type(Kind k, Location loc) : Expr(k, loc) {}

public:
    struct DenseMapInfo {
        static auto getEmptyKey() -> Type* { return nullptr; }
        static auto getTombstoneKey() -> Type* { return reinterpret_cast<Type*>(1); }
        static bool isEqual(const Expr* a, const Expr* b) {
            /// Expr::Equal doesn’t handle nullptr or tombstones.
            uptr ap = uptr(a), bp = uptr(b);
            if (ap < 2 or bp < 2) return ap == bp;
            return Type::Equal(const_cast<Expr*>(a), const_cast<Expr*>(b));
        }

        /// Include the element type in the hash if possible.
        static auto getHashValue(const Expr* t) -> usz;
    };

    /// In the parser, prefer to create new instances of these
    /// initially for better location tracking.
    static BuiltinType* const Int;
    static BuiltinType* const Unknown;
    static BuiltinType* const Void;
    static BuiltinType* const Bool;
    static BuiltinType* const NoReturn;
    static BuiltinType* const OverloadSet;
    static ReferenceType* const VoidRef;
    static ReferenceType* const VoidRefRef;
    static IntType* const I8;
    static IntType* const I16;
    static IntType* const I32;
    static IntType* const I64;
    static FFIType* const CChar;
    static FFIType* const CInt;
    static Nil* const Nil;

    /// It is too goddamn easy to forget to dereference at least
    /// one of the expressions when comparing them w/ operator==,
    /// so we disallow that altogether.
    static bool Equal(Expr* a, Expr* b);

    /// Wrapper around Equal() for a tuple.
    template <typename T>
    static bool Equal(T&& t)
    requires requires (T u) { std::get<0>(u); }
    {
        auto&& [a, b] = std::forward<T>(t);
        return Equal(a, b);
    }

    /// Note: an Expr may be a type even if this returns false.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::BuiltinType and e->kind <= Kind::ClosureType;
    }
};

class IntType : public Type {
public:
    /// The size of this integer type, in bits.
    Size size;

    IntType(Size size, Location loc)
        : Type(Kind::IntType, loc), size(size) {
        /// TODO: Maybe allow `i0` so we can set this unconditionally?
        if (size.bits() > 0) sema.set_done();
    }

    static auto Create(Module* mod, Size size, Location loc = {}) -> IntType* {
        return new (mod) IntType(size, loc);
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IntType; }
};

/// This is both the nil type and the nil literal.
class Nil : public Type {
public:
    Nil(Location loc) : Type(Kind::Nil, loc) { sema.set_done(); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::Nil; }
};

class BuiltinType : public Type {
    using K = BuiltinTypeKind;

public:
    /// The kind of this builtin type.
    const BuiltinTypeKind builtin_kind;

private:
    /// Create a new builtin type.
    static auto Create(Module* m, BuiltinTypeKind kind, Location loc) -> BuiltinType* {
        auto bt = new (m) BuiltinType(kind, loc);
        return bt;
    }

public:
    BuiltinType(BuiltinTypeKind kind, Location loc)
        : Type(Kind::BuiltinType, loc), builtin_kind(kind) {
        sema.set_done();
    }

    static auto Unknown(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::Unknown, loc); }
    static auto Void(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::Void, loc); }
    static auto Int(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::Int, loc); }
    static auto Bool(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::Bool, loc); }
    static auto NoReturn(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::NoReturn, loc); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BuiltinType; }
};

class FFIType : public Type {
public:
    /// The kind of this FFI type.
    FFITypeKind ffi_kind;

private:
    /// Create a new FFI type.
    static auto Create(Module* m, FFITypeKind kind, Location loc) -> FFIType* {
        return new (m) FFIType(kind, loc);
    }

public:
    FFIType(FFITypeKind kind, Location loc)
        : Type(Kind::FFIType, loc), ffi_kind(kind) {
        sema.set_done();
    }

    /// Get the underlying type.
    auto underlying(Context* ctx) const -> Type*;

    static auto CChar(Module* m, Location loc = {}) -> FFIType* { return Create(m, FFITypeKind::CChar, loc); }
    static auto CInt(Module* m, Location loc = {}) -> FFIType* { return Create(m, FFITypeKind::CInt, loc); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::FFIType; }
};

class StructType : public Type {
    /// Needs to access mangled name.
    friend Expr::TypeHandle;

    /// Cached so we don’t need to recompute it.
    std::string mangled_name;

public:
    struct Field {
        std::string name;
        Expr* type;
        Size offset{};
        u32 index{};
        bool padding{};
    };

    /// The parent module.
    Module* module;

    /// The fields of this struct.
    SmallVector<Field> all_fields;

    /// Initialisers of this struct.
    SmallVector<ProcDecl*> initialisers;

    /// The name of this struct.
    std::string name;

    /// Scope associated with this struct.
    BlockExpr* scope;

    /// Cached size and alignment, in bits.
    Size stored_size{};
    Align stored_alignment{1};

    /// MLIR type.
    SOURCE_MLIR_TYPE_MEMBER(mlir);

    StructType(
        Module* mod,
        std::string name,
        SmallVector<Field> fields,
        SmallVector<ProcDecl*> initialisers,
        BlockExpr* scope,
        Location loc
    );

    /// Get the non-padding fields of this struct.
    auto fields() {
        return vws::filter(all_fields, [](Field& f) { return not f.padding; });
    }

    /// Get the fields of this struct, including all padding fields.
    auto field_types() {
        return vws::transform(all_fields, &Field::type);
    }

    /// Check if two struct types have the same layout.
    static bool LayoutCompatible(StructType* a, StructType* b);

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::StructType; }
};

class SingleElementTypeBase : public Type {
public:
    /// The element type.
    Expr* elem;

protected:
    SingleElementTypeBase(Kind k, Expr* elem, Location loc)
        : Type(k, loc), elem(elem) {}

public:
    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::ReferenceType and e->kind <= Kind::ClosureType;
    }
};

class ReferenceType : public SingleElementTypeBase {
public:
    ReferenceType(Expr* elem, Location loc)
        : SingleElementTypeBase(Kind::ReferenceType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ReferenceType; }
};

class OptionalType : public SingleElementTypeBase {
public:
    OptionalType(Expr* elem, Location loc)
        : SingleElementTypeBase(Kind::OptionalType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::OptionalType; }
};

class ScopedPointerType : public SingleElementTypeBase {
public:
    ScopedPointerType(Expr* elem, Location loc)
        : SingleElementTypeBase(Kind::ScopedPointerType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ScopedPointerType; }
};

class ArrayType : public SingleElementTypeBase {
public:
    Expr* dim_expr;

    ArrayType(Expr* elem, Expr* dim_expr, Location loc)
        : SingleElementTypeBase(Kind::ArrayType, elem, loc),
          dim_expr(dim_expr) {}

    /// Get the dimension of this array type.
    auto dimension() -> const APInt& {
        Assert(sema.ok);
        auto cexpr = cast<ConstExpr>(dim_expr);
        return cexpr->value.as_int();
    }

    /// Create an i8[X] type of dimension X.
    ///
    /// This is intended for use by the backend only. Do not use
    /// this in the frontend as it offers no location information
    /// whatsoever.
    static auto GetByteArray(Module* mod, isz dim) -> ArrayType* {
        auto lit = new (mod) IntLitExpr(APInt(64, usz(dim)), {});
        lit->stored_type = Type::Int;
        lit->sema.set_done();

        auto cexpr = new (mod) ConstExpr(lit, EvalResult(APInt(64, usz(dim)), Type::Int), {});
        auto arr = new (mod) ArrayType(Type::I8, cexpr, {});
        arr->sema.set_done();
        return arr;
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ArrayType; }
};

class SliceType : public SingleElementTypeBase {
public:
    SliceType(Expr* elem, Location loc)
        : SingleElementTypeBase(Kind::SliceType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SliceType; }
};

/// Like DeclRefExpr, but for types.
class SugaredType : public SingleElementTypeBase {
public:
    /// The name of the type this was looked up as.
    std::string name;

    SugaredType(std::string name, Expr* underlying, Location loc)
        : SingleElementTypeBase(Kind::SugaredType, underlying, loc),
          name(std::move(name)) {
        sema.set_done();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SugaredType; }
};

/// Like ScopeAccessExpr, but for types.
class ScopedType : public SingleElementTypeBase {
public:
    /// Expression being accessed.
    Expr* object;

    /// The name of the type this was looked up as.
    std::string name;

    ScopedType(std::string name, Expr* object, Expr* resolved, Location loc)
        : SingleElementTypeBase(Kind::ScopedType, resolved, loc),
          object(object),
          name(std::move(name)) {
        sema.set_done();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SugaredType; }
};

class ProcType : public Type {
public:
    /// The parameter types.
    SmallVector<Expr*> param_types;

    /// The return type.
    Expr* ret_type;

    /// The procedure whose chain pointer this takes.
    ProcDecl* static_chain_parent{};

    /// Get the type this is an initialiser of, if any.
    StructType* init_of{};

    /// Whether this is an initialiser.
    readonly_const(bool, is_init, return init_of != nullptr);

    /// Whether this type is variadic.
    bool variadic{};

    ProcType(SmallVector<Expr*> param_types, Expr* ret_type, bool variadic, Location loc)
        : Type(Kind::ProcType, loc),
          param_types(std::move(param_types)),
          ret_type(ret_type),
          variadic(variadic) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ProcType; }
};

class ClosureType : public SingleElementTypeBase {
public:
    ClosureType(ProcType* proc_type)
        : SingleElementTypeBase(Kind::ClosureType, proc_type, proc_type->location) {
        Assert(not proc_type->sema.in_progress);
        sema = proc_type->sema;
    }

    readonly(ProcType*, proc_type, return cast<ProcType>(elem));

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ClosureType; }
};

/// ===========================================================================
///  Remaining Expressions
/// ===========================================================================
/// Declared here because it references StructType::Field.
class MemberAccessExpr : public TypedExpr {
public:
    /// The name of the member being accessed.
    std::string member;

    /// The object being accessed.
    Expr* object;

    /// The field being accessed if this is a struct field access.
    StructType::Field* field{};

    MemberAccessExpr(Expr* object, std::string member, Location loc)
        : TypedExpr(Kind::MemberAccessExpr, detail::UnknownType, loc),
          member(std::move(member)),
          object(object) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::MemberAccessExpr; }
};

template <typename To, typename From>
struct THCastImpl {
    static_assert(std::derived_from<To, src::Expr>);
    static bool isPossible(const From t) { return To::classof(static_cast<src::Expr*>(t)); }
    static bool isPossible(const From* t) { return isPossible(*t); }
    static auto doCast(const From t) -> To* { return static_cast<To*>(static_cast<src::Expr*>(t)); }
    static auto doCast(const From* t) -> To* { return doCast(*t); }
    static auto doCastIfPossible(const From* t) -> To* { return doCastIfPossible(*t); }
    static auto doCastIfPossible(const From t) -> To* {
        if (not isPossible(t)) return nullptr;
        return doCast(t);
    }
};
} // namespace src

namespace llvm {

template <typename T>
struct CastInfo<T, src::Expr::TypeHandle> : src::THCastImpl<T, src::Expr::TypeHandle> {};
template <typename T>
struct CastInfo<T, src::Expr::TypeHandle&> : src::THCastImpl<T, src::Expr::TypeHandle&> {};
template <typename T>
struct CastInfo<T, const src::Expr::TypeHandle> : src::THCastImpl<T, const src::Expr::TypeHandle> {};
template <typename T>
struct CastInfo<T, const src::Expr::TypeHandle&> : src::THCastImpl<T, const src::Expr::TypeHandle&> {};
template <typename T>
struct CastInfo<T, src::Expr::TypeHandle*> : src::THCastImpl<T, src::Expr::TypeHandle*> {};
template <typename T>
struct CastInfo<T, const src::Expr::TypeHandle*> : src::THCastImpl<T, const src::Expr::TypeHandle*> {};

} // namespace llvm

#endif // SOURCE_FRONTEND_AST_HH
