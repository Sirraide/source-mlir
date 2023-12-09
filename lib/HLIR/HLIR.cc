// clang-format off
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

/// Include order matters here!
#include <source/HLIR/HLIRDialect.hh>
#include <source/HLIR/HLIROpsDialect.cpp.inc>
#include <source/HLIR/HLIREnumAttr.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <source/HLIR/HLIROpsTypes.cpp.inc>

#define GET_OP_CLASSES
#include <source/HLIR/HLIROps.cpp.inc>

// clang-format on

/// This file is only for things that take a long time to compile and
/// should not be changed if possible. Place HLIR-related constructs
/// in HLIRDialect.cc instead.

void mlir::hlir::HLIRDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <source/HLIR/HLIROpsTypes.cpp.inc>
        >();

    addOperations<
#define GET_OP_LIST
#include <source/HLIR/HLIROps.cpp.inc>
        >();
}

namespace mlir::hlir {
void InitContext(MLIRContext& mctx) {
    mlir::registerAllDialects(mctx);
    mlir::registerBuiltinDialectTranslation(mctx);
    mlir::registerLLVMDialectTranslation(mctx);
    mctx.loadDialect< // clang-format off
        mlir::func::FuncDialect,
        mlir::scf::SCFDialect,
        mlir::affine::AffineDialect,
        mlir::LLVM::LLVMDialect,
        mlir::index::IndexDialect,
        mlir::math::MathDialect,
        mlir::cf::ControlFlowDialect,
        mlir::DLTIDialect
    >(); // clang-format on
    mctx.printOpOnDiagnostic(true);
    mctx.printStackTraceOnDiagnostic(true);
}

void AddLegalDialects(ConversionTarget& target) {
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect< // clang-format off
        mlir::hlir::HLIRDialect,
        mlir::func::FuncDialect,
        mlir::scf::SCFDialect,
        mlir::arith::ArithDialect,
        mlir::affine::AffineDialect,
        mlir::LLVM::LLVMDialect,
        mlir::index::IndexDialect,
        mlir::math::MathDialect,
        mlir::cf::ControlFlowDialect,
        mlir::DLTIDialect
    >(); // clang-format on
}

} // namespace mlir::hlir
