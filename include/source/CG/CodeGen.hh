#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <string>

namespace mlir {
class MLIRContext;
}

namespace llvm {
class ThreadPool;
}

namespace src {
class Module;
class Expr;
void LowerHLIR(mlir::MLIRContext* ctx, Module* mod);
void LowerToLLVM(mlir::MLIRContext* ctx, Module* mod, bool debug_llvm_lowering, bool no_verify);
void CodeGenModule(mlir::MLIRContext* ctx, Module* mod, bool no_verify);
}

#endif // SOURCE_CG_CODEGEN_HH
