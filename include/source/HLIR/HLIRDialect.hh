#ifndef SOURCE_HLIR_HLIRDIALECT_HH
#define SOURCE_HLIR_HLIRDIALECT_HH

/// Include order matters here!
// clang-format off
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <hlir/HLIROpsDialect.h.inc>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#define GET_TYPEDEF_CLASSES
#include <hlir/HLIROpsTypes.h.inc>

#define GET_OP_CLASSES
#include <hlir/HLIROps.h.inc>


// clang-format on

namespace hlir = ::mlir::hlir;

#endif // SOURCE_HLIR_HLIRDIALECT_HH
