#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <mlir/IR/Builders.h>
#include <source/Core.hh>

namespace src {
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
    void GenerateModule();
    void GenerateProcedure(ProcDecl* proc);

    auto Ty(Expr* type) -> mlir::Type;
};
} // namespace src

#endif // SOURCE_CG_CODEGEN_HH
