#ifndef SOURCE_FRONTEND_AST_HH
#define SOURCE_FRONTEND_AST_HH

#include <mlir/IR/Value.h>
#include <source/Core.hh>
#include <source/Frontend/Lexer.hh>
#include <source/Support/Result.hh>

namespace src {
class Expr;
class Type;
class FunctionDecl;

namespace detail {
extern Expr* const UnknownType;
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

/// ===========================================================================
///  Special Expressions
/// ===========================================================================
class Expr {
public:
    enum struct Kind {
        /// Type [begin]
        BuiltinType,
        FFIType,
        IntType,
        ReferenceType,
        ScopedPointerType,
        SliceType,
        OptionalType,
        ProcType,
        /// Type [end]

        AssertExpr,
        ReturnExpr,
        DeferExpr,
        WhileExpr,
        LoopControlExpr,

        /// TypedExpr [begin]
        BlockExpr,
        InvokeExpr,
        CastExpr,
        MemberAccessExpr,
        UnaryPrefixExpr,
        IfExpr,
        BinaryExpr,
        DeclRefExpr,
        BoolLiteralExpr,
        IntegerLiteralExpr,
        StringLiteralExpr,

        /// Decl [begin]
        ParamDecl,

        /// ObjectDecl [begin]
        ProcDecl,
        VarDecl,
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
        readonly(bool, analysed, return state == St::Errored or state == St::Ok);
        readonly(bool, errored, return state == St::Errored);
        readonly(bool, in_progress, return state == St::InProgress);
        readonly(bool, ok, return state == St::Ok);

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
    };

    /// Helper type that makes accessing properties of types easier.
    class TypeHandle {
        Expr* ptr;

    public:
        TypeHandle(Expr* ptr) : ptr(ptr) {}
        TypeHandle(std::nullptr_t) = delete;

        /// Get the alignment of this type, in bits.
        auto align(Context* ctx) -> isz;

        /// Check if this is any integer type.
        bool is_int(bool bool_is_int);

        /// Check if this is the builtin 'noreturn' type.
        readonly_decl(bool, is_noreturn);

        /// Get the number of nested reference levels in this type.
        readonly_decl(isz, ref_depth);

        /// Get the size of this type, in bits.
        auto size(Context* ctx) -> isz;

        /// Get the size of this type, in bytes.
        auto size_bytes(Context* ctx) {
            auto sz = size(ctx);
            return sz / 8 + (sz % 8 != 0);
        }

        /// Get a string representation of this type.
        auto str(bool use_colour) const -> std::string;

        /// Strip all references from this type.
        readonly_decl(TypeHandle, strip_refs);

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

    /// Whether this expression has already been emitted.
    bool emitted = false;

    /// Check if this is an lvalue.
    bool is_lvalue = false;

    /// The MLIR value of this expression.
    mlir::Value mlir{};
public:
    Expr(Kind k, Location loc) : kind(k), location(loc) {}
    virtual ~Expr() = default;

    /// Only allow allocating nodes in a module.
    void* operator new(size_t) = delete;
    void* operator new(size_t sz, Module* mod) noexcept {
        return utils::AllocateAndRegister<Expr>(sz, mod->exprs);
    }

    /// Get the type of this expression; returns void if
    /// this expression has no type.
    readonly_decl(TypeHandle, type);

    /// Get this expression as a type handle; this is different
    /// from `type` which returns a handle to the type of this
    /// expression.
    readonly(TypeHandle, as_type, return TypeHandle(this));

    /// Print this expression to stdout.
    void print() const;

    /// RTTI.
    static bool classof(const Expr* e) { return true; }
};

class AssertExpr : public Expr {
public:
    /// The condition of this assertion.
    Expr* cond;

    /// The optional message of this assertion.
    Expr* msg;

    /// TODO: Remove this once we allow arbitrary expressions as messages.
    std::string message_string;

    AssertExpr(Expr* cond, Expr* msg, Location loc)
        : Expr(Kind::AssertExpr, loc),
          cond(cond),
          msg(msg) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::AssertExpr; }
};

class ReturnExpr : public Expr {
public:
    /// The value being returned.
    Expr* value;

    ReturnExpr(Expr* value, Location loc)
        : Expr(Kind::ReturnExpr, loc),
          value(value) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ReturnExpr; }
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

class WhileExpr : public Expr {
public:
    /// The condition of this while loop.
    Expr* cond;

    /// The body of this while loop.
    Expr* body;

    /// Optional label.
    std::string label;

    WhileExpr(Expr* cond, Expr* body, std::string label, Location loc)
        : Expr(Kind::WhileExpr, loc),
          cond(cond),
          body(body),
          label(std::move(label)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::WhileExpr; }
};

class LoopControlExpr : public Expr {
public:
    /// Label to jump to.
    std::string label;

    /// Resolved expression. This is set to the parent
    /// if there is no label. This is resolved in Sema.
    Expr* target{};

    /// Whether this is a continue or break.
    bool is_continue;
    readonly(bool, is_break, return not is_continue);

    LoopControlExpr(std::string label, bool is_continue, Location loc)
        : Expr(Kind::LoopControlExpr, loc),
          label(std::move(label)),
          is_continue(is_continue) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LoopControlExpr; }
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

class BlockExpr : public TypedExpr {
public:
    /// The expressions that are part of this block.
    SmallVector<Expr*> exprs;

    /// The scope of this block.
    Scope* scope;

    /// Whether this expression was create implicitly.
    bool implicit;

public:
    BlockExpr(Scope* scope, SmallVector<Expr*> exprs, Location loc, bool implicit = false)
        : TypedExpr(Kind::BlockExpr, detail::UnknownType, loc),
          exprs(std::move(exprs)),
          scope(scope),
          implicit(implicit) {}

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

    /// Initialiser.
    Expr* init;

    InvokeExpr(Expr* callee, SmallVector<Expr*> args, bool naked, Expr* init, Location loc)
        : TypedExpr(Kind::InvokeExpr, detail::UnknownType, loc),
          args(std::move(args)),
          naked(naked),
          callee(callee),
          init(init) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::InvokeExpr; }
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

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::CastExpr; }
};

class MemberAccessExpr : public TypedExpr {
public:
    /// The name of the member being accessed.
    std::string member;

    /// The object being accessed.
    Expr* object;

    MemberAccessExpr(Expr* object, std::string member, Location loc)
        : TypedExpr(Kind::MemberAccessExpr, detail::UnknownType, loc),
          member(std::move(member)),
          object(object) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::MemberAccessExpr; }
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

class DeclRefExpr : public TypedExpr {
public:
    /// The name of the declaration this refers to.
    std::string name;

    /// The scope in which this name was found.
    Scope* scope;

    /// The declaration this refers to.
    Expr* decl;

    DeclRefExpr(std::string name, Scope* sc, Location loc)
        : TypedExpr(Kind::DeclRefExpr, detail::UnknownType, loc),
          name(std::move(name)),
          scope(sc),
          decl(nullptr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::DeclRefExpr; }
};

class IntLitExpr : public TypedExpr {
public:
    /// The value of this literal.
    isz value;

    IntLitExpr(isz value, Location loc)
        : TypedExpr(Kind::IntegerLiteralExpr, detail::UnknownType, loc),
          value(value) {}

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
    static bool classof(const Expr* e) { return e->kind >= Kind::ParamDecl; }
};

class ObjectDecl : public Decl {
public:
    /// Linkage of this object.
    Linkage linkage;

    /// Mangling scheme.
    Mangling mangling;

    /// Whether this decl is imported or exported.
    readonly(bool, imported, return linkage == Linkage::Imported or linkage == Linkage::Reexported);
    readonly(bool, exported, return linkage == Linkage::Exported or linkage == Linkage::Reexported);

    ObjectDecl(Kind k, std::string name, Expr* type, Linkage linkage, Mangling mangling, Location loc)
        : Decl(k, std::move(name), type, loc),
          linkage(linkage),
          mangling(mangling) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind >= Kind::ProcDecl; }
};

class ParamDecl : public Decl {
public:
    ParamDecl(std::string name, Expr* type, Location loc)
        : Decl(Kind::ParamDecl, std::move(name), type, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ParamDecl; }
};

class VarDecl : public ObjectDecl {
public:
    /// The initialiser.
    Expr* init;

    VarDecl(
        Module* mod,
        std::string name,
        Expr* type,
        Expr* init,
        Linkage linkage,
        Mangling mangling,
        Location loc
    ) : ObjectDecl(Kind::VarDecl, std::move(name), type, linkage, mangling, loc),
        init(init) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::VarDecl; }
};

class ProcDecl : public ObjectDecl {
public:
    /// The module this procedure belongs to.
    Module* module;

    /// The function parameter decls.
    SmallVector<ParamDecl*> params;

    /// Body of the function.
    BlockExpr* body;

    /// Labels are global per procedure.
    StringMap<Expr*> labels;

    ProcDecl(
        Module* mod,
        std::string name,
        Expr* type,
        SmallVector<ParamDecl*> params,
        BlockExpr* body,
        Linkage linkage,
        Mangling mangling,
        Location loc
    ) : ObjectDecl(Kind::ProcDecl, std::move(name), type, linkage, mangling, loc),
        module(mod),
        params(std::move(params)),
        body(body) {
        mod->add_function(this);
    }

    /// Add a labelled expression to the function. This is done
    /// at parse time so all labels are available in sema.
    ///
    /// \param label The label to register.
    /// \param expr The expression that the label points at.
    /// \return The expression, or an error.
    auto add_label(std::string label, Expr* expr) -> Result<Expr*> {
        if (labels.contains(label)) return Diag::Error(
            module->context,
            expr->location,
            "Label '{}' is already defined",
            label
        );

        labels[label] = expr;
        return expr;
    }

    /// Get the return type of this procedure.
    [[gnu::pure]] readonly_decl(TypeHandle, ret_type);

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ProcDecl; }
};

/// ===========================================================================
///  Types
/// ===========================================================================
class BuiltinType;
class FFIType;

enum struct BuiltinTypeKind {
    Unknown,
    Void,
    Int,
    Bool,
    NoReturn,
};

enum struct FFITypeKind {
    CChar,
    CInt,
};

class Type : public Expr {
protected:
    Type(Kind k, Location loc) : Expr(k, loc) {}

public:
    /// Prefer to create new instances of these initially
    /// for better location tracking.
    static BuiltinType* const Int;
    static BuiltinType* const Unknown;
    static BuiltinType* const Void;
    static BuiltinType* const Bool;
    static BuiltinType* const NoReturn;

    /// It is too goddamn easy to forget to dereference at least
    /// one of the expressions when comparing them w/ operator==,
    /// so we disallow that altogether.
    static bool Equal(Expr* a, Expr* b);

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::BuiltinType and e->kind <= Kind::ProcType;
    }
};

class IntType : public Type {
public:
    /// The size of this integer type, in bits.
    isz bits;

    IntType(isz bits, Location loc)
        : Type(Kind::IntType, loc), bits(bits) {}

    static auto Create(Module* mod, isz size, Location loc = {}) -> IntType* {
        return new (mod) IntType(size, loc);
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IntType; }
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
        : Type(Kind::FFIType, loc), ffi_kind(kind) {}

    static auto CChar(Module* m, Location loc = {}) -> FFIType* { return Create(m, FFITypeKind::CChar, loc); }
    static auto CInt(Module* m, Location loc = {}) -> FFIType* { return Create(m, FFITypeKind::CInt, loc); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::FFIType; }
};

class SingleElementTypeBase : public Type {
public:
    /// The element type.
    Expr* elem;

protected:
    SingleElementTypeBase(Kind k, Expr* elem, Location loc)
        : Type(k, loc), elem(elem) {}
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

class SliceType : public SingleElementTypeBase {
public:
    SliceType(Expr* elem, Location loc)
        : SingleElementTypeBase(Kind::SliceType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SliceType; }
};

class ProcType : public Type {
public:
    /// The parameter types.
    SmallVector<Expr*> param_types;

    /// The return type.
    Expr* ret_type;

    ProcType(SmallVector<Expr*> param_types, Expr* ret_type, Location loc)
        : Type(Kind::ProcType, loc),
          param_types(std::move(param_types)),
          ret_type(ret_type) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ProcType; }
};

/// ===========================================================================
///  Scope
/// ===========================================================================
class Scope {
public:
    using Symbols = StringMap<SmallVector<Expr*, 1>>;

    /// The parent scope.
    Scope* parent;

    /// The module this scope belongs to.
    Module* module;

    /// Symbols in this scope.
    Symbols symbol_table;

    /// Whether this scope is a function scope.
    bool is_function;

    /// Get the nearest parent scope that is a function scope.
    readonly_decl(Scope*, enclosing_function_scope);

    Scope(const Scope& other) = delete;
    Scope& operator=(const Scope& other) = delete;

    /// Disallow creating scopes except in the module.
    void* operator new(size_t) = delete;
    void* operator new(size_t, Module*) noexcept;

    /// Create a new scope.
    explicit Scope(Scope* parent, Module* mod)
        : parent{parent}, module{mod} {}

    /// Declare a symbol in this scope.
    void declare(StringRef name, Expr* value) {
        symbol_table[name].push_back(value);
    }

    /// Mark this scope as a function scope. This cannot be undone.
    void set_function_scope() {
        Assert(not is_function, "Scope already marked as function scope");
        is_function = true;
    }

    /// Visit each symbol with the given name.
    template <typename Func>
    void visit(StringRef name, bool this_scope_only, Func f) {
        if (auto sym = symbol_table.find(name); sym != symbol_table.end())
            if (std::invoke(f, sym->second) == utils::StopIteration)
                return;
        if (parent and not this_scope_only) parent->visit(name, false, f);
    }
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
