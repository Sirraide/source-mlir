#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

namespace src {
class Module;
void LowerHLIR(Module* mod);
void LowerToLLVM(Module* mod, bool debug_llvm_lowering, bool no_verify);
void CodeGenModule(Module* mod, bool no_verify);
}

#endif // SOURCE_CG_CODEGEN_HH
