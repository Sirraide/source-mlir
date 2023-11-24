#ifndef SOURCE_HLIR_HLIRDIALECT_HH
#define SOURCE_HLIR_HLIRDIALECT_HH

/// Include order matters here!
// clang-format off

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/VectorInterfaces.h>

#include <source/HLIR/HLIROpsDialect.h.inc>

#include <source/HLIR/HLIREnumAttr.h.inc>

#define GET_TYPEDEF_CLASSES
#include <source/HLIR/HLIROpsTypes.h.inc>

#define GET_OP_CLASSES
#include <source/HLIR/HLIROps.h.inc>
// clang-format on

namespace hlir = ::mlir::hlir;

#endif // SOURCE_HLIR_HLIRDIALECT_HH
