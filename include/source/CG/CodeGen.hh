#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <mlir/IR/Builders.h>
#include <source/HLIR/HLIRDialect.hh>
#include <source/Core.hh>

namespace src {
class BinaryExpr;
class CodeGen {
    Module* const mod;
    Context* const ctx;
    mlir::MLIRContext* const mctx;
    mlir::OpBuilder builder;

    CodeGen(Module* mod)
        : mod(mod),
          ctx(mod->context),
          mctx(&ctx->mlir),
          builder(mctx) {}

public:
    static void Generate(Module* mod) {
        Assert(not mod->context->has_error(), "Refusing to codegen broken module");
        CodeGen c{mod};
        c.GenerateModule();
    }

private:
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
