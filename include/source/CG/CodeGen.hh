#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <mlir/IR/Builders.h>
#include <source/Core.hh>
#include <source/Frontend/AST.hh>
#include <source/HLIR/HLIRDialect.hh>

namespace src {
class CodeGen {
    Module* const mod;
    Context* const ctx;
    mlir::MLIRContext* const mctx;
    mlir::OpBuilder builder;
    usz anon_structs = 0;
    usz anon_ctors = 0;
    usz anon_dtors = 0;
    bool no_verify;

    /// External procedures that we have already declared.
    llvm::DenseSet<StringRef> procs;

    /// Procedure we’re currently emitting.
    ProcDecl* curr_proc{};

    CodeGen(Module* mod, bool no_verify)
        : mod(mod),
          ctx(mod->context),
          mctx(&ctx->mlir),
          builder(mctx),
          no_verify(no_verify) {}

    CodeGen(const CodeGen&) = delete;
    CodeGen(CodeGen&&) = delete;
    CodeGen& operator=(const CodeGen&) = delete;
    CodeGen& operator=(CodeGen&&) = delete;

public:
    static void Generate(Module* mod, bool no_verify) {
        Assert(not mod->context->has_error(), "Refusing to codegen broken module");
        CodeGen c{mod, no_verify};
        c.GenerateModule();
    }

private:
    /// Create an alloca for a local variable.
    [[nodiscard]] auto AllocateLocalVar(LocalDecl* decl) -> mlir::Value;

    /// Attach a block to the end of a region.
    auto Attach(mlir::Region* region, mlir::Block* block) -> mlir::Block*;

    /// Check if a block is closed.
    bool Closed();
    bool Closed(mlir::Block* block);

    /// Get the (mangled) name of the constructor of a type. Returns
    /// the empty string if there is no constructor.
    auto Constructor(Expr* type) -> StringRef;

    template <typename T, typename... Args>
    auto Create(mlir::Location loc, Args&&... args) -> decltype(builder.create<T>(loc, std::forward<Args>(args)...));

    /// Create a function and execute a callback to populate its body.
    template <typename Callable>
    auto CreateProcedure(mlir::FunctionType type, StringRef name, Callable callable);

    /// Create an external function.
    void CreateExternalProcedure(mlir::FunctionType type, StringRef name);

    /// Get the (mangled) name of the destructor of a type.
    auto Destructor(Expr* type) -> std::optional<StringRef>;

    auto EmitReference(mlir::Location loc, Expr* decl) -> mlir::Value;

    auto EndLifetime(LocalDecl* decl);

    template <typename Op>
    void GenerateBinOp(BinaryExpr* b);

    template <typename Op>
    void GenerateCmpOp(BinaryExpr*, mlir::arith::CmpIPredicate pred);

    void Generate(Expr* expr);
    void GenerateModule();
    void GenerateProcedure(ProcDecl* proc);

    /// Retrieve the static chain pointer for a procedure.
    auto GetStaticChainPointer(ProcDecl* proc) -> mlir::Value;

    /// Perform preprocessing on locals to support nested procedures.
    void InitStaticChain(ProcDecl* proc, hlir::FuncOp f);

    auto Ty(Expr* type, bool for_closure = false) -> mlir::Type;

    /// Generate a set of operations to unwind from expressions that need unwinding.
    auto UnwindValues(ArrayRef<Expr*> exprs) -> SmallVector<mlir::Value>;
};
} // namespace src

#endif // SOURCE_CG_CODEGEN_HH
