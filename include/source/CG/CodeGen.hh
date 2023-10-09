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
    template <typename T, typename ...Args>
    auto Create(mlir::Location loc, Args&& ...args) -> decltype(builder.create<T>(loc, std::forward<Args>(args)...));

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
