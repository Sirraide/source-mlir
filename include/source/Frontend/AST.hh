#ifndef SOURCE_FRONTEND_AST_HH
#define SOURCE_FRONTEND_AST_HH

#include <source/Core.hh>
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
        OptionalType,
        ProcType,
        /// Type [end]

        /// TypedExpr [begin]
        BlockExpr,
        InvokeExpr,
        MemberAccessExpr,
        DeclRefExpr,
        IntegerLiteralExpr,
        StringLiteralExpr,

        /// Decl [begin]
        ParamDecl,

        /// ObjectDecl [begin]
        ProcDecl,
    };

private:
    /// The kind of this expression.
    property_r(const Kind, kind);

    /// The location of this expression.
    property_r(Location, location);

public:
    Expr(Kind k, Location loc) : kind_field(k), location_field(loc) {}

    /// Only allow allocating nodes in a module.
    void* operator new(size_t) = delete;
    void* operator new(size_t sz, Module* mod) noexcept {
        /// We don’t know the alignment of the type, so we assume
        /// that it’s not larger than max_align_t.
        return utils::AllocateAndRegister<Expr, alignof(max_align_t)>(sz, mod->exprs);
    }

    /// Get the type of this expression; returns void if
    /// this expression has no type.
    readonly_decl(Expr*, type);

    /// Print this expression to stdout.
    void print() const;

    /// Get a string representation of this expression
    /// as a type. If this is not a type, get the string
    /// representation of the type of this expression.
    auto type_str(bool use_colour) const -> std::string;

    /// RTTI.
    static bool classof(const Expr* e) { return true; }
};

/// ===========================================================================
///  Typed Expressions
/// ===========================================================================
class TypedExpr : public Expr {
    /// The type of this expression.
    property_rw(Expr*, type);

public:
    TypedExpr(Kind k, Expr* type, Location loc)
        : Expr(k, loc), type_field(type) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind >= Kind::BlockExpr; }
};

class BlockExpr : public TypedExpr {
    /// The expressions that are part of this block.
    property_r(SmallVector<Expr*>, exprs);

    /// Whether this expression was create implicitly.
    property_r(bool, implicit);

public:
    BlockExpr(SmallVector<Expr*> exprs, Location loc, bool implicit = false)
        : TypedExpr(Kind::BlockExpr, detail::UnknownType, loc),
          exprs_field(std::move(exprs)),
          implicit_field(implicit) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BlockExpr; }
};

class InvokeExpr : public TypedExpr {
    /// The function being invoked.
    property_rw(Expr*, callee);

    /// The arguments to the function.
    property_r(SmallVector<Expr*>, args);

    /// Initialiser.
    property_rw(Expr*, init);

    /// Whether this is a naked invocation.
    property_r(bool, naked);

public:
    InvokeExpr(Expr* callee, SmallVector<Expr*> args, bool naked, Expr* init, Location loc)
        : TypedExpr(Kind::InvokeExpr, detail::UnknownType, loc),
          callee_field(callee),
          args_field(std::move(args)),
          init_field(init),
          naked_field(naked) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::InvokeExpr; }
};

class MemberAccessExpr : public TypedExpr {
    /// The object being accessed.
    property_rw(Expr*, object);

    /// The name of the member being accessed.
    property_r(std::string, member);

public:
    MemberAccessExpr(Expr* object, std::string member, Location loc)
        : TypedExpr(Kind::MemberAccessExpr, detail::UnknownType, loc),
          object_field(object),
          member_field(std::move(member)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::MemberAccessExpr; }
};

class DeclRefExpr : public TypedExpr {
    /// The name of the declaration this refers to.
    property_r(std::string, name);

    /// The declaration this refers to.
    property_rw(Expr*, decl);

public:
    DeclRefExpr(std::string name, Location loc)
        : TypedExpr(Kind::DeclRefExpr, detail::UnknownType, loc),
          name_field(std::move(name)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::DeclRefExpr; }
};

class IntLitExpr : public TypedExpr {
    /// The value of this literal.
    property_r(isz, value);

public:
    IntLitExpr(isz value, Location loc)
        : TypedExpr(Kind::IntegerLiteralExpr, detail::UnknownType, loc),
          value_field(value) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IntegerLiteralExpr; }
};

class StrLitExpr : public TypedExpr {
    /// The index of this literal in the string table.
    property_r(u32, index);

public:
    StrLitExpr(u32 index, Location loc)
        : TypedExpr(Kind::StringLiteralExpr, detail::UnknownType, loc),
          index_field(index) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::StringLiteralExpr; }
};

/// ===========================================================================
///  Declarations
/// ===========================================================================
class Decl : public TypedExpr {
    /// The name of this declaration.
    property_r(std::string, name);

public:
    Decl(Kind k, std::string name, Expr* type, Location loc)
        : TypedExpr(k, type, loc), name_field(std::move(name)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind >= Kind::ParamDecl; }
};

class ObjectDecl : public Decl {
    /// Linkage of this object.
    property_r(Linkage, linkage);

    /// Mangling scheme.
    property_r(Mangling, mangling);

public:
    ObjectDecl(Kind k, std::string name, Expr* type, Linkage linkage, Mangling mangling, Location loc)
        : Decl(k, std::move(name), type, loc),
          linkage_field(linkage),
          mangling_field(mangling) {}

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

class ProcDecl : public ObjectDecl {
    /// The function parameter decls.
    property_r(SmallVector<ParamDecl*>, params);

    /// Body of the function.
    property_rw(BlockExpr*, body);

public:
    ProcDecl(
        std::string name,
        Expr* type,
        SmallVector<ParamDecl*> params,
        BlockExpr* body,
        Linkage linkage,
        Mangling mangling,
        Location loc
    ) : ObjectDecl(Kind::ProcDecl, std::move(name), type, linkage, mangling, loc),
        params_field(std::move(params)),
        body_field(body) {}

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
    static BuiltinType* const Void;

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::BuiltinType and e->kind <= Kind::ProcType;
    }
};

class IntType : public Type {
    /// The size of this integer type, in bits.
    property_r(isz, bits);

public:
    IntType(isz bits, Location loc)
        : Type(Kind::IntType, loc), bits_field(bits) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IntType; }
};

class BuiltinType : public Type {
    using K = BuiltinTypeKind;

    /// The kind of this builtin type.
    property_r(const BuiltinTypeKind, builtin_kind);

    /// Create a new builtin type.
    static auto Create(Module* m, BuiltinTypeKind kind, Location loc) -> BuiltinType* {
        return new (m) BuiltinType(kind, loc);
    }

public:
    BuiltinType(BuiltinTypeKind kind, Location loc)
        : Type(Kind::BuiltinType, loc), builtin_kind_field(kind) {}

    static auto Unknown(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::Unknown, loc); }
    static auto Void(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::Void, loc); }
    static auto Int(Module* m, Location loc = {}) -> BuiltinType* { return Create(m, K::Int, loc); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BuiltinType; }
};

class FFIType : public Type {
    /// The kind of this FFI type.
    property_r(FFITypeKind, ffi_kind);

    /// Create a new FFI type.
    static auto Create(Module* m, FFITypeKind kind, Location loc) -> FFIType* {
        return new (m) FFIType(kind, loc);
    }

public:
    FFIType(FFITypeKind kind, Location loc)
        : Type(Kind::FFIType, loc), ffi_kind_field(kind) {}

    static auto CChar(Module* m, Location loc = {}) -> FFIType* { return Create(m, FFITypeKind::CChar, loc); }
    static auto CInt(Module* m, Location loc = {}) -> FFIType* { return Create(m, FFITypeKind::CInt, loc); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::FFIType; }
};

class SingleElementTypeBase : public Type {
    /// The element type.
    property_r(Expr*, elem);

protected:
    SingleElementTypeBase(Kind k, Expr* elem, Location loc)
        : Type(k, loc), elem_field(elem) {}
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

class ProcType : public Type {
    /// The parameter types.
    property_r(SmallVector<Expr*>, param_types);

    /// The return type.
    property_r(Expr*, ret_type);

public:
    ProcType(SmallVector<Expr*> param_types, Expr* ret_type, Location loc)
        : Type(Kind::ProcType, loc),
          param_types_field(std::move(param_types)),
          ret_type_field(ret_type) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ProcType; }
};

/// ===========================================================================
///  Scope
/// ===========================================================================
class Scope {
    using Symbols = StringMap<SmallVector<Expr*, 1>>;

    /// The parent scope.
    property_r(Scope*, parent);

    /// The module this scope belongs to.
    property_r(Module*, module);

    /// Symbols in this scope.
    property_r(Symbols, symbol_table);

    /// Whether this scope is a function scope.
    property_r(bool, is_function);

public:
    /// Get the nearest parent scope that is a function scope.
    readonly_decl(Scope*, enclosing_function_scope);

    Scope(const Scope& other) = delete;
    Scope& operator=(const Scope& other) = delete;

    /// Disallow creating scopes except in the module.
    void* operator new(size_t) = delete;
    void* operator new(size_t, Module*) noexcept;

    /// Create a new scope.
    explicit Scope(Scope* parent, Module* mod)
        : parent_field{parent}, module_field{mod} {}

    /// Declare a symbol in this scope.
    auto declare(std::string name, Expr* value);

    /// Mark this scope as a function scope. This cannot be undone.
    void set_function_scope() {
        Assert(not is_function, "Scope already marked as function scope");
        is_function_field = true;
    }

    /// Visit each symbol with the given name.
    template <typename Func>
    void visit(std::string_view name, Func f, bool this_scope_only) {
        if (auto sym = symbol_table.find(name); sym != symbol_table.end())
            for (auto& expr : sym->second)
                std::invoke(f, expr);
        if (parent and not this_scope_only) parent->visit(name, f, false);
    }
};
} // namespace src

#endif // SOURCE_FRONTEND_AST_HH
