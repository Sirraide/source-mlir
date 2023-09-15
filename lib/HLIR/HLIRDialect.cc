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

void hlir::FuncOp::build(
    OpBuilder& odsBuilder,
    OperationState& odsState,
    StringRef name,
    FunctionType type,
    ArrayRef<NamedAttribute> attrs
) {
    buildWithEntryBlock(odsBuilder, odsState, name, type, attrs, type.getInputs());
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

auto hlir::PrintOp::verify() -> mlir::LogicalResult {
    auto op = getOperand();
    if (not op.getType().isa<StringType>())
        return emitOpError("operand of print must be a string");
    return mlir::success();
}

auto hlir::StringOp::verify() -> mlir::LogicalResult {
    return mlir::success();
}