#ifndef SOURCE_FRONTEND_AST_HH
#define SOURCE_FRONTEND_AST_HH

#include <source/Core.hh>
#include <source/Frontend/Lexer.hh>
#include <source/Support/Generator.hh>
#include <source/Support/Result.hh>

namespace src {
#define SOURCE_AST_EXPR(name) class name;
#include <source/Frontend/AST.def>

class Assimilator;
class Expr;
class Sema;
class Type;
class TypeBase;
class RecordType;

struct ArrayInfo;

/// ===========================================================================
///  Enums
/// ===========================================================================
enum struct Linkage : u8 {
    Local,       ///< Local variable.
    Internal,    ///< Not exported and defined.
    Imported,    ///< Imported from another module or library.
    Exported,    ///< Exported and defined.
    Reexported,  ///< Imported and exported, and thus not defined.
    LinkOnceODR, ///< Merge definitions across different TUs. Used mainly for compiler-generated code.
};

enum struct CallConv : u8 {
    Native, ///< OS calling convention. Compatible with C.
    Source, ///< Our calling convention.
};

enum struct Mangling : u8 {
    None,   ///< Do not mangle.
    Source, ///< Use Source mangling.
};

enum struct Builtin : u8 {
    Destroy,
    Memcpy,
    New,
};

enum struct LocalKind : u8 {
    Variable,         /// Regular stack variable.
    Parameter,        /// Procedure parameter.
    Synthesised,      /// Named *lvalue* that points to an object somewhere else.
    SynthesisedValue, /// Named *rvalue* that denotes some value.
};

class SemaState {
    enum struct St : u8 {
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

/// ===========================================================================
///  Special Expressions
/// ===========================================================================
class alignas(void*) Expr {
public:
    enum struct Kind : u8 {
#define SOURCE_AST_EXPR(name) name,
#include <source/Frontend/AST.def>
    };

    /// The kind of this expression.
    Kind kind;

    /// State of semantic analysis
    SemaState sema{};

    /// Check if this is an lvalue.
    bool is_lvalue : 1 = false;

    /// The location of this expression.
    Location location;

private:
    friend Context;
    void* operator new(size_t sz) {
        return __builtin_operator_new(sz);
    }

public:
    Expr(Kind k, Location loc) : kind(k), location(loc) {}

    /// This requires invoking the destructor of a derived class.
    void operator delete(Expr* e, std::destroying_delete_t);

    /// Only allow allocating nodes in a module.
    void* operator new(size_t sz, Module* mod) noexcept {
        return utils::AllocateAndRegister<Expr>(sz, mod->exprs);
    }

    /// Strip labels.
    readonly_decl(Expr*, ignore_labels);

    /// Strip lvalue-to-rvalue conversion. This only removes one
    /// level of lvalue-to-rvalue conversion, not lvalue-ref-to-lvalue
    /// conversion.
    readonly_decl(Expr*, ignore_lv2rv);

    /// Strip parentheses.
    readonly_decl(Expr*, ignore_parens);

    /// Strip parentheses, implicit casts, and DeclRefExprs.
    readonly_decl(Expr*, ignore_paren_cast_refs);

    /// Strip parentheses, and DeclRefExprs.
    readonly_decl(Expr*, ignore_paren_refs);

    /// Check if this is 'nil'.
    readonly_decl(bool, is_nil);

    /// Check if this is an expression that we can not branch
    /// over forwards (e.g. DeferExpr, LocalDecl).
    readonly_decl(bool, is_protected);

    /// Whether this is an initialiser/deleter of a struct, an
    /// overload set thereof, or a procedure type that is an smp.
    readonly_decl(bool, is_smp);

    /// Get a string representation of the name of the scope of
    /// this expression, if it has one.
    readonly_decl(std::string, scope_name);

    /// Get the type of this expression; returns void if
    /// this expression has no type.
    readonly_decl(Type, type);

    /// Print this expression to stdout.
    void print(bool print_children = true) const;
};

/// Wrapper around an expression that should be treated as a type.
///
/// This is a value type, so pass it by value.
///
/// Due to the fact that types are expressions, we can’t really separate
/// the two. At the same time, allowing *any* expression to be transparently
/// used as a type has been the source of many bugs, so we use this wrapper
/// to try and make sure to only treat Expr*’s as types where it makes sense
/// to.
class Type {
    Expr* ptr;

    friend Sema;
    friend Assimilator;

    struct NullConstructTag {};
    explicit constexpr Type(NullConstructTag) : ptr(nullptr) {}

public:
    static constinit const Type Int;
    static constinit const Type Unknown;
    static constinit const Type Void;
    static constinit const Type Bool;
    static constinit const Type NoReturn;
    static constinit const Type OverloadSet;
    static constinit const Type MemberProc;
    static constinit const Type ArrayLiteral;
    static constinit const Type VoidRef;
    static constinit const Type VoidRefRef;
    static constinit const Type I8;
    static constinit const Type I16;
    static constinit const Type I32;
    static constinit const Type I64;
    static constinit const Type Nil;

    constexpr explicit Type(Expr* ptr) : ptr(ptr) { Assert(ptr); }
    constexpr Type(TypeBase* ptr);
    Type(std::nullptr_t) = delete;

    /// Get an empty type. This is only meant for deserialisation
    /// and should not be used for anything else.
    static Type UnsafeEmpty() { return Type(NullConstructTag{}); }

    /// Get the alignment of this type, in *bytes*.
    auto align(Context* mod) const -> Align;

    /// Get the procedure type from a closure or proc.
    readonly_decl(ProcType*, callable);

    /// Get the default constructor for this type.
    ///
    /// This only returns a procedure if there is an actual non-trivial
    /// constructor that takes no arguments. For types such as `int`, this
    /// always returns null.
    readonly_decl(ProcDecl*, default_constructor);

    /// Get the type stripped of any sugar.
    readonly_const_decl(Type, desugared);

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
    auto size(Context* mod) const -> Size;

    /// Get a string representation of this type.
    auto str(bool use_colour, bool include_desugared = false) const -> std::string;

    /// Strip all arrays from this type (including sugar) and determine
    /// the overall size of the (multidimensional) array.
    readonly_decl(ArrayInfo, strip_arrays);

    /// Strip all references from this type.
    readonly_decl(Type, strip_refs);

    /// Strip all references and pointers from this type.
    readonly_decl(Type, strip_refs_and_pointers);

    /// Check whether this type is trivial, has no constructors or destructors
    /// at all; that is, it is either a builtin type or a type for which default
    /// construction is zero-initialisation, and there is no destructor.
    readonly_decl(bool, trivial);

    /// Check whether this type is trivial to copy, i.e. whether it can be copied
    /// via a memcpy.
    readonly_decl(bool, trivially_copyable);

    /// Check if this type logically yields a value, i.e. is not
    /// void or noreturn.
    readonly_decl(bool, yields_value);

    /// Check if two types are equal.
    friend bool operator==(Type a, Type b);

    /// Access the underlying type pointer.
    explicit operator Expr*() const { return ptr; };
    Expr* operator->() { return ptr; }
};

struct ArrayInfo {
    /// First nested non-array type (not desugared).
    Type base_type;

    /// Total size of the array as unrolled into a single dimension.
    usz total_dimension;

    /// Number of nested arrays, e.g. 3 for `int[2][3][4]`.
    usz array_depth;
};

class EvalResult {
public:
    using TupleElements = std::vector<EvalResult>;

private:
    std::variant< // clang-format off
        std::monostate,
        APInt,
        Type,
        OverloadSetExpr*,
        TupleElements,
        std::nullptr_t
    > value{}; // clang-format on

public:
    Type type{Type::Unknown};

    EvalResult() : value(std::monostate{}) {}
    EvalResult(std::nullptr_t); /// Nil.
    EvalResult(APInt value, Type type) : value(std::move(value)), type(type) {}
    EvalResult(OverloadSetExpr* os) : value(os), type(Type::OverloadSet) {}
    EvalResult(Type type) : value(type), type(type) {} /// FIXME: should be the `type` type.
    EvalResult(TupleElements elems, TupleType* type);

    auto as_int() -> APInt& { return std::get<APInt>(value); }
    auto as_tuple_elems() -> TupleElements& { return std::get<TupleElements>(value); }
    auto as_type() -> Type { return std::get<Type>(value); }
    auto as_overload_set() -> OverloadSetExpr* { return std::get<OverloadSetExpr*>(value); }

    bool is_int() const { return std::holds_alternative<APInt>(value); }
    bool is_type() const { return std::holds_alternative<Type>(value); }
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

    /// The index variable of this for-in loop.
    LocalDecl* index;

    /// The range that we’re iterating over.
    Expr* range;

    /// Whether we’re iterating in reverse.
    bool reverse;

    ForInExpr(
        LocalDecl* iter,
        LocalDecl* index,
        Expr* range,
        BlockExpr* body,
        bool reverse,
        Location loc
    ) : Loop(Kind::ForInExpr, body, loc),
        iter(iter),
        index(index),
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
    String label;

    /// Resolved expression. This is set to the parent
    /// if there is no label. This is resolved in Sema.
    Loop* target{};

    /// Whether this is a continue or break.
    bool is_continue;
    readonly(bool, is_break, return not is_continue);

    LoopControlExpr(String label, bool is_continue, Location loc)
        : UnwindExpr(Kind::LoopControlExpr, loc),
          label(label),
          is_continue(is_continue) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LoopControlExpr; }
};

class LabelExpr : public Expr {
public:
    /// The label of this expression.
    String label;

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
        String label,
        Expr* expr,
        Location loc
    );

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::LabelExpr; }
};

class GotoExpr : public UnwindExpr {
public:
    /// The label to jump to.
    String label;

    /// The resolved labelled expression.
    LabelExpr* target{};

    GotoExpr(String label, Location loc)
        : UnwindExpr(Kind::GotoExpr, loc),
          label(label) {}

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
          overloads(std::move(overloads)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::OverloadSetExpr; }
};

enum struct ConstructKind {
    /// Do nothing. No arguments.
    Uninitialised,

    /// Fill the entire thing with zeroes. This is just
    /// a memset with the size of the type. No arguments.
    Zeroinit,

    /// Function parameter. This is handled separately and is a
    /// no-op in constructor codegen.
    Parameter,

    /// Trivial memcpy/store. One argument.
    TrivialCopy,

    /// Construct a slice from a pointer and a size. Two arguments.
    SliceFromParts,

    /// Call an initialiser. Procedure + some arguments.
    InitialiserCall,

    /// Call an initialiser on each element of an array. Procedure
    /// + some arguments + array size.
    ArrayInitialiserCall,

    /// Broad cast a value to each element of an array. One argument +
    /// array size.
    ArrayBroadcast,

    /// Zero-initialise (part of) an array. Array size.
    ArrayZeroinit,

    /// Initialise an array from a series of nested constructors.
    ArrayListInit,

    /// Initialise a record from a series of constructors. Note that
    /// padding fields are left uninitialised; no constructors will
    /// be provided for them in this list. The tuple type is added at
    /// the end of the list.
    RecordListInit,
};

/// This expression stores all information required to construct a value.
class ConstructExpr final
    : public Expr
    , llvm::TrailingObjects<ConstructExpr, isz, Expr*, usz> {
    friend TrailingObjects;
    using K = ConstructKind;
    using NumExprs = isz;
    using NumArrayElems = usz;
    enum : NumArrayElems { DefaultParamNoArrayElements = ~usz(0) };

    template <typename T>
    using OT = OverloadToken<T>;

public:
    /// Constructor kind (e.g. zeroinit etc.).
    const ConstructKind ctor_kind;

private:
    ConstructExpr(ConstructKind k)
        : Expr(Kind::ConstructExpr, Location{}), ctor_kind(k) {}

    template <utils::is<ProcDecl*, RecordType*, std::nullptr_t> ProcOrRecordPtrType = std::nullptr_t>
    static auto Create(
        Module* mod,
        ConstructKind k,
        ArrayRef<Expr*> args,
        ProcOrRecordPtrType proc = nullptr,
        usz array_elems = DefaultParamNoArrayElements
    ) -> ConstructExpr* {
        auto num_exprs = args.size() + (proc != nullptr);
        auto raw = utils::AllocateAndRegister<Expr>(
            totalSizeToAlloc<NumExprs, Expr*, NumArrayElems>(
                HasArgumentsSize(k),
                num_exprs,
                array_elems != DefaultParamNoArrayElements
            ),
            mod->exprs
        );

        /// Construct the expression.
        auto e = ::new (raw) ConstructExpr(k);

        /// These have to be initialised first as they are used in the
        /// initialisation of the other ones below.
        if (HasArgumentsSize(k))
            e->getTrailingObjects<NumExprs>()[0] = NumExprs(num_exprs);
        if (array_elems != DefaultParamNoArrayElements)
            e->getTrailingObjects<NumArrayElems>()[0] = array_elems;

        /// Do *not* move these up!
        std::uninitialized_copy(args.begin(), args.end(), e->getTrailingObjects<Expr*>());
        if (proc != nullptr) e->getTrailingObjects<Expr*>()[args.size()] = proc;
        return e;
    }

    /// The `isz` argument denotes how many arguments we have and is
    /// only present if this is not an uninit or zeroinit ctor.
    usz numTrailingObjects(OT<NumExprs>) const {
        return HasArgumentsSize(ctor_kind);
    }

    /// The `usz` argument denotes how many array elements we need to
    /// default-construct.
    usz numTrailingObjects(OT<NumArrayElems>) const {
        return HasArrayElemCount(ctor_kind);
    }

    usz numTrailingObjects(OT<Expr*>) const {
        switch (ctor_kind) {
            case K::Uninitialised:
            case K::Zeroinit:
            case K::Parameter:
            case K::ArrayZeroinit:
                return 0;

            case K::TrivialCopy:
            case K::ArrayBroadcast:
                return 1;

            case K::SliceFromParts:
                return 2;

            case K::InitialiserCall:
            case K::ArrayListInit:
            case K::ArrayInitialiserCall:
            case K::RecordListInit:
                /// This includes the trailing proc/record type.
                return usz(getTrailingObjects<NumExprs>()[0]);
        }

        Unreachable();
    }

    static bool HasArgumentsSize(ConstructKind k) {
        switch (k) {
            case K::Uninitialised:
            case K::Zeroinit:
            case K::Parameter:
            case K::TrivialCopy:
            case K::SliceFromParts:
            case K::ArrayBroadcast:
            case K::ArrayZeroinit:
                return false;

            case K::InitialiserCall:
            case K::ArrayListInit:
            case K::ArrayInitialiserCall:
            case K::RecordListInit:
                return true;
        }

        Unreachable();
    }

    static bool HasArrayElemCount(ConstructKind k) {
        switch (k) {
            case K::Uninitialised:
            case K::Zeroinit:
            case K::Parameter:
            case K::TrivialCopy:
            case K::SliceFromParts:
            case K::InitialiserCall:
            case K::ArrayListInit:
            case K::RecordListInit:
                return false;

            case K::ArrayZeroinit:
            case K::ArrayBroadcast:
            case K::ArrayInitialiserCall:
                return true;
        }

        Unreachable();
    }

    static bool HasTrailingProcedureOrRecordType(ConstructKind k) {
        return k == K::InitialiserCall or
               k == K::ArrayInitialiserCall or
               k == K::RecordListInit;
    }


public:
    /// Whether this is an array constructor.
    auto array_ctor() -> bool {
        switch (ctor_kind) {
            case K::Uninitialised:
            case K::Zeroinit:
            case K::Parameter:
            case K::TrivialCopy:
            case K::SliceFromParts:
            case K::InitialiserCall:
            case K::RecordListInit:
                return false;

            case K::ArrayListInit:
            case K::ArrayZeroinit:
            case K::ArrayBroadcast:
            case K::ArrayInitialiserCall:
                return true;
        }
    }

    /// Get the arguments to this construct expression.
    auto args() -> MutableArrayRef<Expr*> {
        return {
            getTrailingObjects<Expr*>(),
            numTrailingObjects(OT<Expr*>{}) - HasTrailingProcedureOrRecordType(ctor_kind),
        };
    }

    /// Get the arguments and the initialiser. This is meant for printing.
    auto args_and_init() -> MutableArrayRef<Expr*> {
        return {
            getTrailingObjects<Expr*>(),
            numTrailingObjects(OT<Expr*>{}),
        };
    }

    /// Get the number of array elements. Returns 1 if this is not an array ctor.
    auto elems() -> usz {
        if (not HasArrayElemCount(ctor_kind)) return 1;
        return getTrailingObjects<NumArrayElems>()[0];
    }

    /// Get the initialiser to call.
    auto init() -> ProcDecl* {
        if (not HasTrailingProcedureOrRecordType(ctor_kind)) return nullptr;
        return cast<ProcDecl>(getTrailingObjects<Expr*>()[numTrailingObjects(OT<Expr*>{}) - 1]);
    }

    /// Get the record type for RecordListInit.
    auto record_type() -> RecordType* {
        if (not HasTrailingProcedureOrRecordType(ctor_kind)) return nullptr;
        return cast<RecordType>(getTrailingObjects<Expr*>()[numTrailingObjects(OT<Expr*>{}) - 1]);
    }

    static auto CreateUninitialised(Module* m) { return new (m) ConstructExpr(K::Uninitialised); }
    static auto CreateZeroinit(Module* m) { return new (m) ConstructExpr(K::Zeroinit); }
    static auto CreateParam(Module* m) { return new (m) ConstructExpr(K::Parameter); }
    static auto CreateTrivialCopy(Module* m, Expr* expr) { return Create(m, K::TrivialCopy, expr); }
    static auto CreateSliceFromParts(Module* m, Expr* ptr, Expr* size) { return Create(m, K::SliceFromParts, {ptr, size}); }
    static auto CreateArrayListInit(Module* m, ArrayRef<Expr*> args) { return Create(m, K::ArrayListInit, args); }

    static auto CreateInitialiserCall(Module* m, ProcDecl* init, ArrayRef<Expr*> args) {
        return Create(m, K::InitialiserCall, args, init);
    }

    static auto CreateArrayZeroinit(Module* m, usz total_array_size) {
        return Create(m, K::ArrayZeroinit, {}, nullptr, total_array_size);
    }

    static auto CreateArrayBroadcast(Module* m, Expr* value, usz total_array_size) {
        return Create(m, K::ArrayBroadcast, value, nullptr, total_array_size);
    }

    static auto CreateArrayInitialiserCall(
        Module* m,
        ArrayRef<Expr*> exprs,
        ProcDecl* init,
        usz total_array_size
    ) { return Create(m, K::ArrayInitialiserCall, exprs, init, total_array_size); }

    static auto CreateRecordListInit(
        Module* m,
        ArrayRef<Expr*> exprs,
        RecordType* type
    ) { return Create(m, K::RecordListInit, exprs, type); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ConstructExpr; }
};

class AliasExpr : public Expr {
public:
    /// The name of the alias.
    String alias;

    /// The aliased expression.
    Expr* expr;

    AliasExpr(String alias, Expr* expr, Location loc)
        : Expr(Kind::AliasExpr, loc),
          alias(std::move(alias)),
          expr(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::AliasExpr; }
};

class ArrayLitExpr : public Expr {
public:
    /// The elements of this array literal.
    SmallVector<Expr*> elements;

    /// The result object into which this literal is evaluated. If null,
    /// the backend will create a temporary for this to be generated into.
    Expr* result_object{};

    ArrayLitExpr(SmallVector<Expr*> elements, Location loc)
        : Expr(Kind::ArrayLitExpr, loc),
          elements(std::move(elements)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ArrayLitExpr; }
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
    Type stored_type;

    TypedExpr(Kind k, Type type, Location loc)
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
    using Symbols = llvm::TinyPtrVector<Expr*>;

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
        : TypedExpr(Kind::BlockExpr, Type::Unknown, loc),
          parent(parent),
          module(mod),
          implicit(implicit) {}

    /// Declare a symbol in this scope.
    void declare(String name, Expr* value) {
        symbol_table[name.value()].push_back(value);
    }

    /// Find a (vector of) symbol(s) in this scope.
    auto find(String name, bool this_scope_only) -> Symbols* {
        if (auto sym = symbol_table.find(name.value()); sym != symbol_table.end())
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

    /// Bound initialisers arguments. These are any arguments
    /// after a `=` in an invoke expression; they are stored
    /// here to facilitate conversion to a variable declaration
    /// in sema.
    SmallVector<Expr*> init_args;

    /// The function being invoked.
    Expr* callee;

    /// Whether this is a naked invocation.
    bool naked;

    InvokeExpr(Expr* callee, SmallVector<Expr*> args, bool naked, SmallVector<Expr*> init, Location loc)
        : TypedExpr(Kind::InvokeExpr, Type::Unknown, loc),
          args(std::move(args)),
          init_args(std::move(init)),
          callee(callee),
          naked(naked) {}

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
        : TypedExpr(Kind::InvokeBuiltinExpr, Type::Unknown, loc),
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

    CastExpr(CastKind kind, Expr* expr, Type type, Location loc)
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
        : TypedExpr(Kind::UnaryPrefixExpr, Type::Unknown, loc),
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
        : TypedExpr(Kind::IfExpr, Type::Unknown, loc),
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
        : TypedExpr(Kind::BinaryExpr, Type::Unknown, loc),
          lhs(lhs),
          rhs(rhs),
          op(op) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BinaryExpr; }
};

class ScopeAccessExpr : public TypedExpr {
public:
    /// The name of the element being accessed.
    String element;

    /// The object being accessed.
    Expr* object;

    /// The resolved reference.
    Expr* resolved{};

    ScopeAccessExpr(Expr* object, String element, Location loc)
        : TypedExpr(Kind::ScopeAccessExpr, Type::Unknown, loc),
          element(element),
          object(object) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ScopeAccessExpr; }
};

class DeclRefExpr : public TypedExpr {
public:
    /// The name of the declaration this refers to.
    String name;

    /// The scope in which this name was found. May be null
    /// iff `ok` returns true (that is, if the scope is no
    /// longer needed).
    BlockExpr* scope;

    /// The declaration this refers to.
    Expr* decl;

    DeclRefExpr(String name, BlockExpr* sc, Location loc)
        : TypedExpr(Kind::DeclRefExpr, Type::Unknown, loc),
          name(name),
          scope(sc),
          decl(nullptr) {}

    /// Create a resolved DeclRefExpr. It is set to done iff the
    /// referenced proc decl is.
    DeclRefExpr(Decl* referenced, Location loc);

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
        : TypedExpr(Kind::ParenExpr, Type::Unknown, loc),
          expr(expr) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ParenExpr; }
};

class TupleExpr : public TypedExpr {
public:
    /// The elements of this tuple.
    SmallVector<Expr*> elements;

    TupleExpr(SmallVector<Expr*> elements, Location loc)
        : TypedExpr(Kind::TupleExpr, Type::Unknown, loc),
          elements(std::move(elements)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::TupleExpr; }
};

class SubscriptExpr : public TypedExpr {
public:
    Expr* object;
    Expr* index;

    SubscriptExpr(Expr* object, Expr* index, Location loc)
        : TypedExpr(Kind::SubscriptExpr, Type::Unknown, loc),
          object(object),
          index(index) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SubscriptExpr; }
};

class TupleIndexExpr : public TypedExpr {
public:
    Expr* object;
    FieldDecl* field;

    TupleIndexExpr(Expr* object, FieldDecl* field, Location loc);

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::TupleIndexExpr; }
};

class IntLitExpr : public TypedExpr {
public:
    /// The value of this literal.
    APInt value;

    IntLitExpr(APInt value, Location loc)
        : TypedExpr(Kind::IntLitExpr, Type::Unknown, loc),
          value(std::move(value)) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IntLitExpr; }
};

class BoolLitExpr : public TypedExpr {
public:
    /// The value of this literal.
    bool value;

    BoolLitExpr(bool value, Location loc)
        : TypedExpr(Kind::BoolLitExpr, Type::Unknown, loc),
          value(value) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::BoolLitExpr; }
};

class StrLitExpr : public TypedExpr {
public:
    /// The index of this literal in the string table.
    u32 index;

    StrLitExpr(u32 index, Location loc)
        : TypedExpr(Kind::StrLitExpr, Type::Unknown, loc),
          index(index) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::StrLitExpr; }
};

/// ===========================================================================
///  Declarations
/// ===========================================================================
class Decl : public TypedExpr {
public:
    /// The name of this declaration.
    String name;

    Decl(Kind k, String name, Type type, Location loc)
        : TypedExpr(k, type, loc), name(name) {}

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
        String name,
        Type type,
        Linkage linkage,
        Mangling mangling,
        Location loc
    ) : Decl(k, name, type, loc),
        module(mod),
        linkage(linkage),
        mangling(mangling) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind >= Kind::ProcDecl; }
};

/// Struct field decl.
class FieldDecl : public Decl {
public:
    /// Offset to this field in the containing struct.
    Size offset{};

    /// Index of this field in the containing struct.
    u32 index{};

    /// Whether this is a padding field.
    bool padding{};

    FieldDecl(
        String name,
        Type type,
        Location loc,
        Size offset = Size::Bits(0),
        u32 index = 0,
        bool padding = false
    ) : Decl(Kind::FieldDecl, name, type, loc),
        offset(offset),
        index(index),
        padding(padding) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::FieldDecl; }
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
    ConstructExpr* ctor{};

    /// Index in capture list of parent procedure, if any.
    isz capture_index{};

    /// What kind of variable this is.
    LocalKind local_kind;

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
        return local_kind != LocalKind::Synthesised and
               local_kind != LocalKind::SynthesisedValue and
               is_lvalue /// Also handles 'in' parameters.
    );

protected:
    LocalDecl(
        Kind k,
        ProcDecl* parent,
        String name,
        Type type,
        SmallVector<Expr*> init,
        LocalKind kind,
        Location loc
    ) : Decl(k, name, type, loc),
        parent(parent),
        init_args(std::move(init)),
        local_kind(kind) {
    }

public:
    LocalDecl(
        ProcDecl* parent,
        String name,
        Type type,
        SmallVector<Expr*> init,
        LocalKind kind,
        Location loc
    ) : LocalDecl( //
            Kind::LocalDecl,
            parent,
            name,
            type,
            std::move(init),
            kind,
            loc
        ) {}

    /// Mark this declaration as captured.
    void set_captured();

    /// Mark that this declaration is deleted or moved from.
    void set_deleted_or_moved() { deleted = true; }

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind == Kind::LocalDecl or e->kind == Kind::ParamDecl;
    }
};

enum struct Intent : u8 {
    Move = 0b0001,
    Copy = 0b0010,
    In = 0b0100,
    Out = 0b1000,
    CXXByValue = 0b1'0000, /// A C++ value parameter.
    Inout = In | Out,
    Default = Move, ///< Default intent for parameters.

    LLVM_MARK_AS_BITMASK_ENUM(Inout)
};

constexpr auto stringify(Intent i) -> std::string_view {
    switch (i) {
        case Intent::Move: return "move";
        case Intent::Copy: return "copy";
        case Intent::In: return "in";
        case Intent::Out: return "out";
        case Intent::Inout: return "inout";
        case Intent::CXXByValue: return "cxx-by-value";
    }

    Unreachable();
}

/// Parameter information used by both ProcTypes and ParamDecls.
class ParamInfo {
    friend Sema;

public:
    /// How this parameter is passed. See also Sema::ClassifyParameter().
    enum struct Class : u8 {
        AnyRef,    ///< Passed by reference. Temporary materialisation is allowed.
        LValueRef, ///< Must be an lvalue. Passed by reference.
        ByVal,     ///< RValue. Passed by value.
        CopyAsRef, ///< A copy is created on the stack and then passed by reference.
    };

    Type type;
    Intent intent;

private:
    /// Whether Sema is already done with this.
    SemaState sema{};

public:
    /// Computed by sema.
    Class cls{};

    /// Whether this is a `with` parameter. This is only relevant
    /// if the corresponding procedure has a body and can be left
    /// unset otherwise.
    bool with : 1 {};

    ParamInfo(const ParamInfo&) = delete;
    ParamInfo& operator=(const ParamInfo&) = delete;
    ParamInfo(ParamInfo&&) = delete;
    ParamInfo& operator=(ParamInfo&&) = delete;
    ParamInfo(Type type, Intent intent, bool with = false)
        : type{type}, intent{intent}, with{with} {}

    /// Whether this is, in any way, shape, or form, passed by reference.
    readonly_const(bool, passed_by_reference, return cls != Class::ByVal);

    [[nodiscard]] bool operator==(const ParamInfo& other) const {
        return std::tie(type, intent) == std::tie(other.type, other.intent);
    }
};

class ParamDecl : public LocalDecl {
public:
    ParamInfo* info{};
    ParamDecl(
        ProcDecl* parent,
        ParamInfo* info,
        String name,
        Location loc
    ) : LocalDecl( //
            Kind::ParamDecl,
            parent,
            name,
            info->type,
            {},
            LocalKind::Parameter,
            loc
        ),
        info(info) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ParamDecl; }
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
        String name,
        Expr* type,
        Expr* init,
        Linkage linkage,
        Mangling mangling,
        Location loc
    ) : ObjectDecl(Kind::LocalDecl, name type, linkage, mangling, loc),
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
    SmallVector<ParamDecl*> params;

    /// Body of the function.
    BlockExpr* body;

    /// Labels are global per procedure.
    StringMap<LabelExpr*> labels;

    /// Captured variables.
    SmallVector<LocalDecl*> captured_locals;

    /// Type of the struct containing the captured variables.
    StructType* captured_locals_type{};

    /// The parent struct if this is a member function.
    StructType* parent_struct{};

    /// MLIR function.
    mlir::Operation* mlir_func{};

    ProcDecl(
        Module* mod,
        ProcDecl* parent,
        String name,
        Type type,
        SmallVector<ParamDecl*> params,
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
    auto add_label(String label, LabelExpr* expr) {
        if (labels.contains(label.value())) Diag::Error(
            module->context,
            expr->location,
            "Label '{}' is already defined",
            label
        );

        labels[label.value()] = expr;
    }

    /// Whether this is a nested procedure.
    readonly(bool, nested, return parent and parent->parent != nullptr);

    /// Whether this takes a static chain pointer parameter.
    readonly_decl(bool, takes_static_chain);

    /// Get the return type of this procedure.
    [[gnu::pure]] readonly_decl(Type, ret_type);

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ProcDecl; }
};

/// Only generated by Sema for accessing the implicit `this` parameter of initialisers.
class ImplicitThisExpr : public TypedExpr {
public:
    /// The initialiser whose `this` this is.
    ProcDecl* init;

    ImplicitThisExpr(ProcDecl* init, Type type, Location loc)
        : TypedExpr(Kind::ImplicitThisExpr, type, loc),
          init(init) {
        sema.set_done();
        is_lvalue = true;
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ImplicitThisExpr; }
};

class WithExpr : public TypedExpr {
public:
    /// The argument to the with expression.
    Expr* object;

    /// The body of the with expression. May be null.
    BlockExpr* body;

    WithExpr(Expr* object, BlockExpr* body, Location loc)
        : TypedExpr(Kind::WithExpr, Type::Unknown, loc),
          object(object),
          body(body) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::WithExpr; }
};

/// Create a temporary variable and evaluate a ConstructExpr into it. Yields
/// that variable as an lvalue.
class MaterialiseTemporaryExpr : public TypedExpr {
public:
    /// The expression constructing this temporary.
    ConstructExpr* ctor;

    MaterialiseTemporaryExpr(Type type, ConstructExpr* ctor, Location loc)
        : TypedExpr(Kind::MaterialiseTemporaryExpr, type, loc),
          ctor(ctor) {
        sema.set_done();
        is_lvalue = true;
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::MaterialiseTemporaryExpr; }
};

/// ===========================================================================
///  Types
/// ===========================================================================
class BuiltinType;
class IntType;
class ReferenceType;

enum struct BuiltinTypeKind {
    /// Unset. We don’t want to set types to null to avoid crashes, so
    /// this is used instead.
    Unknown,

    /// The void type.
    Void,

    /// The builtin pointer-sized integer type.
    Int,

    /// The builtin boolean type.
    Bool,

    /// Type that indicates that a function never returns. Also used
    /// for the type of control-flot expressions like break, return
    /// and goto.
    NoReturn,

    /// Type of a set of function overloads. This is convertible to
    /// each function type in the set.
    OverloadSet,

    /// Type of a member function expression, e.g. `a.foo`. The only
    /// thing that can reasonably be done with this is to call it.
    MemberProc,

    /// Array literals can be oddly-shaped, so assigning a type to
    /// them doesn’t really make much sense; each array literal has
    /// to be treated as a special case.
    ArrayLiteral,
};

class TypeBase : public Expr {
protected:
    TypeBase(Kind k, Location loc) : Expr(k, loc) {}

public:
    struct DenseMapInfo {
        static auto getEmptyKey() -> TypeBase* { return nullptr; }
        static auto getTombstoneKey() -> TypeBase* { return reinterpret_cast<TypeBase*>(1); }
        static bool isEqual(const Expr* a, const Expr* b) {
            /// Expr::Equal doesn’t handle nullptr or tombstones.
            uptr ap = uptr(a), bp = uptr(b);
            if (ap < 2 or bp < 2) return ap == bp;
            return Type(const_cast<Expr*>(a)) == Type(const_cast<Expr*>(b));
        }

        /// Include the element type in the hash if possible.
        static auto getHashValue(const Expr* t) -> usz;
    };

    /// Note: an Expr may be a type even if this returns false.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::BuiltinType and e->kind <= Kind::ClosureType;
    }
};

constexpr Type::Type(TypeBase* ptr) : ptr(ptr) { Assert(ptr); }

class IntType : public TypeBase {
public:
    const Size size;

    IntType(Size size, Location loc)
        : TypeBase(Kind::IntType, loc), size(size) {
        sema.set_done();
    }

    static auto Create(Module* mod, Size size, Location loc = {}) -> IntType* {
        return new (mod) IntType(size, loc);
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::IntType; }
};

/// This is both the nil type and the nil literal.
class Nil : public TypeBase {
public:
    Nil(Location loc) : TypeBase(Kind::Nil, loc) { sema.set_done(); }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::Nil; }
};

class BuiltinType : public TypeBase {
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
        : TypeBase(Kind::BuiltinType, loc), builtin_kind(kind) {
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

/// Helper class for entities that store a mangled name.
class Named {
    /// Needs to access mangled name.
    friend Type;

    /// Cached so we don’t need to recompute it.
    std::string mangled_name;

    /// Avoid race condition on mangled name.
    std::once_flag name_mangling_flag;

public:
    /// The mangling scheme of this type.
    Mangling mangling;

    /// The parent module.
    Module* module;

    /// The name of this type.
    String name;

protected:
    Named(Module* mod, String name, Mangling mangling)
        : mangling(mangling), module(mod), name(name) {}
};

/// Base class for record types (structs and tuples).
class RecordType : public TypeBase {
public:
    /// The fields of this struct.
    SmallVector<FieldDecl*> all_fields;

    /// Cached size and alignment.
    Size stored_size{};
    Align stored_alignment{1};

protected:
    RecordType(Kind k, SmallVector<FieldDecl*> fields, Location loc)
        : TypeBase(k, loc), all_fields(std::move(fields)) {}

public:
    /// Get the fields of this struct, including all padding fields.
    auto field_types() {
        return vws::transform(all_fields, [](FieldDecl* f) { return f->stored_type; });
    }

    /// Get the non-padding fields of this struct.
    auto non_padding_fields() {
        return vws::filter(all_fields, [](FieldDecl* f) { return not f->padding; });
    }

    /// Get a field by *logical* index (i.e. ignoring non-padding fields).
    auto nth_non_padding(u32 n) -> FieldDecl* {
        for (auto f : non_padding_fields()) {
            if (n == 0) return f;
            n--;
        }

        Unreachable("Invalid field index '{}'", n);
    }

    /// Check if two struct types have the same layout.
    static bool LayoutCompatible(RecordType* a, RecordType* b);

    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::StructType and e->kind <= Kind::TupleType;
    }
};

class StructType
    : public RecordType
    , public Named {
public:
    using MemberProcedures = StringMap<llvm::TinyPtrVector<ProcDecl*>>;

    /// Initialisers of this struct.
    SmallVector<ProcDecl*> initialisers;

    /// Member procedures. Note that we do not distinguish between
    /// ‘static’ and ‘non-static’ member functions.
    MemberProcedures member_procs;

    /// Deleter of this struct.
    ProcDecl* deleter{};

    /// Scope associated with this struct.
    BlockExpr* scope;

    StructType(
        Module* mod,
        String name,
        SmallVector<FieldDecl*> fields,
        SmallVector<ProcDecl*> initialisers,
        MemberProcedures member_procs,
        ProcDecl* deleter,
        BlockExpr* scope,
        Mangling mangling,
        Location loc
    );

    /// Creates an anonymous struct. Intended for the backend only.
    StructType(
        Module* mod,
        SmallVector<FieldDecl*> fields,
        Mangling mangling = Mangling::Source,
        Location loc = {}
    ) : StructType( //
            mod,
            String(),
            std::move(fields),
            {},
            {},
            nullptr,
            nullptr,
            mangling,
            loc
        ) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::StructType; }
};

class TupleType : public RecordType {
public:
    TupleType(SmallVector<FieldDecl*> fields, Location loc)
        : RecordType(Kind::TupleType, std::move(fields), loc) {}

    /// Create a tuple type from a list of types.
    TupleType(Module* mod, auto&& types, Location loc = {})
        : TupleType(GetFields(mod, FWD(types)), loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::TupleType; }

private:
    static auto GetFields(Module* mod, auto&& types) -> SmallVector<FieldDecl*> {
        SmallVector<FieldDecl*> non_padding_fields;
        for (auto&& t : FWD(types)) non_padding_fields.push_back(new (mod) FieldDecl(String(), t, {}));
        return non_padding_fields;
    }
};

class OpaqueType
    : public TypeBase
    , public Named {
public:
    OpaqueType(Module* mod, String name, Mangling mangling, Location loc)
        : TypeBase(Kind::OpaqueType, loc),
          Named(mod, std::move(name), mangling) {
        sema.set_done();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::OpaqueType; }
};

class SingleElementTypeBase : public TypeBase {
public:
    /// The element type.
    Type elem;

protected:
    SingleElementTypeBase(Kind k, Type elem, Location loc)
        : TypeBase(k, loc), elem(elem) {}

public:
    /// RTTI.
    static bool classof(const Expr* e) {
        return e->kind >= Kind::ReferenceType and e->kind <= Kind::ClosureType;
    }
};

class ReferenceType : public SingleElementTypeBase {
public:
    ReferenceType(Type elem, Location loc)
        : SingleElementTypeBase(Kind::ReferenceType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ReferenceType; }
};

class OptionalType : public SingleElementTypeBase {
public:
    OptionalType(Type elem, Location loc)
        : SingleElementTypeBase(Kind::OptionalType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::OptionalType; }
};

class ScopedPointerType : public SingleElementTypeBase {
public:
    ScopedPointerType(Type elem, Location loc)
        : SingleElementTypeBase(Kind::ScopedPointerType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::ScopedPointerType; }
};

class ArrayType : public SingleElementTypeBase {
public:
    Expr* dim_expr;

    ArrayType(Type elem, Expr* dim_expr, Location loc)
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

/*class VectorType : public SingleElementTypeBase {
public:
    Expr* dim_expr;

    VectorType(Type elem, Expr* dim_expr, Location loc)
        : SingleElementTypeBase(Kind::VectorType, elem, loc),
          dim_expr(dim_expr) {}

    /// Get the dimension of this vector type.
    auto dimension() -> const APInt& {
        Assert(sema.ok);
        auto cexpr = cast<ConstExpr>(dim_expr);
        return cexpr->value.as_int();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::VectorType; }
};*/

class SliceType : public SingleElementTypeBase {
public:
    SliceType(Type elem, Location loc)
        : SingleElementTypeBase(Kind::SliceType, elem, loc) {}

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SliceType; }
};

/// Like DeclRefExpr, but for types.
class SugaredType : public SingleElementTypeBase {
public:
    /// The name of the type this was looked up as.
    String name;

    SugaredType(String name, Type underlying, Location loc)
        : SingleElementTypeBase(Kind::SugaredType, underlying, loc),
          name(name) {
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
    String name;

    ScopedType(String name, Expr* object, Type resolved, Location loc)
        : SingleElementTypeBase(Kind::ScopedType, resolved, loc),
          object(object),
          name(name) {
        sema.set_done();
    }

    /// RTTI.
    static bool classof(const Expr* e) { return e->kind == Kind::SugaredType; }
};

enum struct SpecialMemberKind {
    Constructor,
    Destructor,
};

class ProcType : public TypeBase {
public:
    /// The parameter types.
    std::deque<ParamInfo> parameters;

    /// The return type.
    Type ret_type;

    /// The procedure whose chain pointer this takes.
    ProcDecl* static_chain_parent{};

    /// Get the type this is an initialiser/deleter of, if any.
    StructType* smp_parent{};

    /// Get what kind of smf this is.
    SpecialMemberKind smp_kind{};

    /// The calling convention of the procedure.
    CallConv call_conv{};

    /// Whether this type is variadic.
    bool variadic{};

    ProcType(std::deque<ParamInfo> parameters, Type ret_type, CallConv cc, bool variadic, Location loc)
        : TypeBase(Kind::ProcType, loc),
          parameters(std::move(parameters)),
          ret_type(ret_type),
          call_conv(cc),
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

    readonly(ProcType*, proc_type, return cast<ProcType>(static_cast<Expr*>(elem)));

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
    String member;

    /// The object being accessed.
    Expr* object;

    /// The member (function) being accessed if this is a field access.
    Expr* field{};

    MemberAccessExpr(Expr* object, String member, Location loc)
        : TypedExpr(Kind::MemberAccessExpr, Type::Unknown, loc),
          member(member),
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

inline EvalResult::EvalResult(std::nullptr_t) : value(nullptr), type(Type::Nil) {}
inline EvalResult::EvalResult(TupleElements elems, TupleType* type)
    : value(std::move(elems)), type(type) {}

/// Check that we’ve actually defined all the types.
#define SOURCE_AST_EXPR(name) static_assert(sizeof(class name));
#include "source/Frontend/AST.def"
} // namespace src

namespace llvm {

template <typename T>
struct CastInfo<T, src::Type> : src::THCastImpl<T, src::Type> {};
template <typename T>
struct CastInfo<T, src::Type&> : src::THCastImpl<T, src::Type&> {};
template <typename T>
struct CastInfo<T, const src::Type> : src::THCastImpl<T, const src::Type> {};
template <typename T>
struct CastInfo<T, const src::Type&> : src::THCastImpl<T, const src::Type&> {};
template <typename T>
struct CastInfo<T, src::Type*> : src::THCastImpl<T, src::Type*> {};
template <typename T>
struct CastInfo<T, const src::Type*> : src::THCastImpl<T, const src::Type*> {};

} // namespace llvm

#endif // SOURCE_FRONTEND_AST_HH
