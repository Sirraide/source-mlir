#ifndef SOURCE_HLIR_HLIRLOWERING_HH
#define SOURCE_HLIR_HLIRLOWERING_HH

#include <hlir/HLIRDialect.hh>

namespace mlir::hlir {
auto CreateLowerToLLVMPass() -> std::unique_ptr<mlir::Pass>;
}

#endif // SOURCE_HLIR_HLIRLOWERING_HH
