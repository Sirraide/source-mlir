// clang-format off
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <fmt/format.h>

#include <source/HLIR/HLIRDialect.hh>
#include <source/HLIR/HLIROpsDialect.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <source/HLIR/HLIROpsTypes.cpp.inc>

#define GET_OP_CLASSES
#include <source/HLIR/HLIROps.cpp.inc>

// clang-format on

void hlir::HLIRDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <source/HLIR/HLIROpsTypes.cpp.inc>
        >();

    addOperations<
#define GET_OP_LIST
#include <source/HLIR/HLIROps.cpp.inc>
        >();
}

auto hlir::FuncOp::parse(OpAsmParser& parser, OperationState& result) -> ParseResult {
    /// Dispatch to function op interface.
    static const auto build_func_type =
        [](
            Builder& builder,
            ArrayRef<Type> argTypes,
            ArrayRef<Type> results,
            function_interface_impl::VariadicFlag,
            std::string&
        ) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser,
        result,
        false,
        getFunctionTypeAttrName(result.name),
        build_func_type,
        getArgAttrsAttrName(result.name),
        getResAttrsAttrName(result.name)
    );
}

void hlir::FuncOp::print(::mlir::OpAsmPrinter& p) {
    /// Dispatch to function op interface.
    function_interface_impl::printFunctionOp(
        p,
        *this,
        false,
        getFunctionTypeAttrName(),
        getArgAttrsAttrName(),
        getResAttrsAttrName()
    );
}

/// Copied from mlir::CallOp::verifySymbolUses().
auto hlir::CallOp::verifySymbolUses(SymbolTableCollection& symbolTable) -> LogicalResult {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
        return emitOpError("requires a 'callee' symbol reference attribute");
    FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
    if (!fn)
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    if (fnType.getNumInputs() != getNumOperands())
        return emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
        if (getOperand(i).getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                << fnType.getInput(i) << ", but provided "
                << getOperand(i).getType() << " for operand number " << i;

    if (fnType.getNumResults() != getNumResults())
        return emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
        if (getResult(i).getType() != fnType.getResult(i)) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getResultTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }

    return success();
}
