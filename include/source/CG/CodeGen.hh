#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <source/Support/Utils.hh>

namespace mlir {
class MLIRContext;
}

namespace llvm {
class ThreadPool;
class TargetMachine;
}

namespace src {
class Module;
class Expr;

enum class ObjectFormat {
    Assembly,
    Executable,
    ObjectFile,
};

void EmitModule(Module* mod, int opt_level, ObjectFormat fmt, const StringMap<bool>& target_features, const fs::path& path);
void CodeGenModule(mlir::MLIRContext* ctx, Module* mod, bool no_verify);
void GenerateLLVMIR(Module* mod, int opt_level, llvm::TargetMachine* machine);
void LowerHLIR(mlir::MLIRContext* ctx, Module* mod, bool no_verify);
void LowerToLLVM(mlir::MLIRContext* ctx, Module* mod, bool debug_llvm_lowering, bool no_verify);
}

#endif // SOURCE_CG_CODEGEN_HH
