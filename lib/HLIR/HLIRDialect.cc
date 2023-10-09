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

/// Get the possible successor regions that can be branched
/// to from either a region in this op or the parent operation.
void hlir::IfOp::getSuccessorRegions(
    /// Nullopt if we’re branching from the parent.
    std::optional<unsigned> index,
    llvm::ArrayRef<mlir::Attribute> operands,
    llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions
) {
    /// The regions in this operation branch to the parent region.
    if (index) {
        regions.push_back(mlir::RegionSuccessor(getResults()));
        return;
    }

    /// Ignore the else region if it is empty.
    auto* else_ = &this->getElseRegion();
    if (else_->empty()) else_ = nullptr;

    /// If the condition is known, add whichever one will be executed.
    if (auto cond = operands.front().dyn_cast_or_null<mlir::IntegerAttr>(); cond) {
        if (cond.getValue().isOne()) regions.push_back(mlir::RegionSuccessor(&getThenRegion()));
        else if (else_) regions.push_back(mlir::RegionSuccessor(else_));
    }

    /// Otherwise, add both.
    else {
        regions.push_back(mlir::RegionSuccessor(&getThenRegion()));
        if (else_) regions.push_back(mlir::RegionSuccessor(else_));
    }
}

void hlir::IfOp::build(mlir::OpBuilder& builder, mlir::OperationState& result, mlir::Value cond, bool withElseRegion) {
    build(builder, result, std::nullopt, cond, withElseRegion);
}

void hlir::IfOp::build(mlir::OpBuilder& builder, mlir::OperationState& result, mlir::TypeRange resultTypes, mlir::Value cond, bool withElseRegion) {
    result.addOperands(cond);
    result.addTypes(resultTypes);

    mlir::Region* thenRegion = result.addRegion();
    thenRegion->push_back(new mlir::Block());
    if (resultTypes.empty())
        IfOp::ensureTerminator(*thenRegion, builder, result.location);

    mlir::Region* elseRegion = result.addRegion();
    if (withElseRegion) {
        elseRegion->push_back(new mlir::Block());
        if (resultTypes.empty())
            IfOp::ensureTerminator(*elseRegion, builder, result.location);
    }
}
