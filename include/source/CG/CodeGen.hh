#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <mlir/IR/Builders.h>
#include <source/Core.hh>
#include <source/HLIR/HLIRDialect.hh>

namespace src {
class BinaryExpr;
class CodeGen {
    Module* const mod;
    Context* const ctx;
    mlir::MLIRContext* const mctx;
    mlir::OpBuilder builder;
    bool no_verify;

    using DeferStackEntry = std::variant<Expr*, mlir::func::FuncOp>;
    using DeferStack = SmallVector<DeferStackEntry, 10>;
    SmallVector<DeferStack, 10> defer_stacks;
    SmallVector<mlir::Value, 10> in_scope_allocas;
    ProcDecl* curr_proc{};
    usz defer_procs{};

    CodeGen(Module* mod, bool no_verify)
        : mod(mod),
          ctx(mod->context),
          mctx(&ctx->mlir),
          builder(mctx),
          no_verify(no_verify) {}

public:
    static void Generate(Module* mod, bool no_verify) {
        Assert(not mod->context->has_error(), "Refusing to codegen broken module");
        CodeGen c{mod, no_verify};
        c.GenerateModule();
    }

private:
    /// Call a function that executes deferred expressions.
    void CallCleanupFunc(mlir::func::FuncOp func);

    /// Check if a block is closed.
    bool Closed();
    bool Closed(mlir::Block* block);

    /// Compact entries in a defer stack to allow emitting
    /// it multiple times.
    void CompactDeferStack(DeferStack& stack);

    template <typename T, typename... Args>
    auto Create(mlir::Location loc, Args&&... args) -> decltype(builder.create<T>(loc, std::forward<Args>(args)...));

    /// Emit the entries in a defer stack.
    void EmitDeferStack(mlir::Location loc, DeferStack& stack);

    template <typename Op>
    void GenerateBinOp(BinaryExpr* b);

    template <typename Op>
    void GenerateCmpOp(BinaryExpr*, mlir::arith::CmpIPredicate pred);

    void Generate(Expr* expr);
    void GenerateModule();
    void GenerateProcedure(ProcDecl* proc);

    auto Ty(Expr* type) -> mlir::Type;
};
} // namespace src

#endif // SOURCE_CG_CODEGEN_HH
