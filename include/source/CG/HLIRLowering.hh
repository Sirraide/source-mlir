#ifndef SOURCE_HLIR_HLIRLOWERING_HH
#define SOURCE_HLIR_HLIRLOWERING_HH

#include <source/Core.hh>

namespace src {
void LowerHLIR(Module* mod);
void LowerToLLVM(Module* mod, bool debug_llvm_lowering, bool no_verify);
} // namespace src

#endif // SOURCE_HLIR_HLIRLOWERING_HH
