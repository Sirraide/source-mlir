#ifndef SOURCE_HLIR_HLIRLOWERING_HH
#define SOURCE_HLIR_HLIRLOWERING_HH

#include <source/Core.hh>

namespace src {
void LowerToLLVM(Module* mod, bool debug_llvm_lowering);
} // namespace src

#endif // SOURCE_HLIR_HLIRLOWERING_HH
