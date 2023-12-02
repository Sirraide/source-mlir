#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Operation.h>
#include <source/HLIR/HLIRDialect.hh>
#include <source/Support/Utils.hh>

using u64 = std::uint64_t;
using i64 = std::int64_t;

static void PrintType(mlir::Type t, mlir::AsmPrinter& p) {
    using namespace hlir;
    llvm::TypeSwitch<mlir::Type>(t)
        .Case<ArrayType>([&](auto t) { t.print(p); })
        .Case<ClosureType>([&](auto t) { t.print(p); })
        .Case<ReferenceType>([&](auto t) { t.print(p); })
        .Case<OptRefType>([&](auto t) { t.print(p); })
        .Case<SliceType>([&](auto t) { t.print(p); })
        .Case<TokenType>([&](auto) { p << "<token>"; })
        .Default([&](auto t) { p.printType(t); });
}

void hlir::HLIRDialect::printType(Type t, DialectAsmPrinter& p) const {
    PrintType(t, p);
}

void hlir::ArrayType::print(AsmPrinter& p) const {
    PrintType(getElem(), p);
    p << "[" << getSize() << "]";
}

::mlir::Type hlir::ArrayType::parse(AsmParser&) { Todo(); }

void hlir::ClosureType::print(AsmPrinter& p) const {
    p << "closure ";
    PrintType(getElem(), p);
}

::mlir::Type hlir::ClosureType::parse(AsmParser&) { Todo(); }

void hlir::OptRefType::print(AsmPrinter& p) const {
    PrintType(getElem(), p);
    p << "&?";
}

::mlir::Type hlir::OptRefType::parse(AsmParser&) { Todo(); }

void hlir::ReferenceType::print(AsmPrinter& p) const {
    PrintType(getElem(), p);
    p << "&";
}

::mlir::Type hlir::ReferenceType::parse(AsmParser&) { Todo(); }

void hlir::SliceType::print(AsmPrinter& p) const {
    PrintType(getElem(), p);
    p << "[]";
}

::mlir::Type hlir::SliceType::parse(AsmParser&) { Todo(); }

void hlir::ArrayDecayOp::print(OpAsmPrinter& p) {
    p << " " << getOperand() << " to ";
    PrintType(getType(), p);
}

auto hlir::ArrayDecayOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::CallOp::print(OpAsmPrinter& p) {
    if (getInlineCall()) p << " inline";
    if (auto cc = getCc().getCallingConv(); cc != LLVM::CConv::C)
        p << " " << LLVM::cconv::stringifyCConv(cc);
    p << " @" << getCallee();

    auto args = getOperands();
    if (not args.empty()) {
        p << "(";
        bool first = true;
        for (auto arg : args) {
            if (first) first = false;
            else p << ", ";
            p << arg;
        }
        p << ")";
    }

    auto res = getYield();
    if (res) {
        p << " -> ";
        PrintType(res.getType(), p);
    }

    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        {getCcAttrName(), getCalleeAttrName(), getInlineCallAttrName()}
    );
}

auto hlir::CallOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::ChainExtractLocalOp::print(OpAsmPrinter& p) {
    p << " chain " << getStructRef() << ", " << getIdx().getValue().getZExtValue();
}

auto hlir::ChainExtractLocalOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::ConstructOp::print(OpAsmPrinter& p) {
    p << " " << getObject() << " " << getInitKind();
    if (not getCtor().empty()) {
        p << " " << getCtorAttr();
        if (auto a = getArgs(); not a.empty()) {
            p << "(";
            bool first = true;
            for (auto arg : a) {
                if (first) first = false;
                else p << ", ";
                PrintType(arg.getType(), p);
                p << " " << arg;
            }
            p << ")";
        }
    } else {
        if (auto a = getArgs(); not a.empty()) {
            Assert(a.size() == 1);
            p << " ";
            PrintType(a.front().getType(), p);
            p << " " << a.front();
        }
    }
}

auto hlir::ConstructOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

auto hlir::DeferOp::getScopeOp() -> ScopeOp {
    Assert(&getBody());
    Assert(not getBody().getBlocks().empty());
    return cast<ScopeOp>(getBody().front().getOperations().front());
}

void hlir::DeferOp::print(OpAsmPrinter& p) {
    p << " ";
    p.printRegion(getScopeOp().getBody(), false, true, false);
}

auto hlir::DeferOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::DeleteOp::print(OpAsmPrinter& p) {
    p << " " << getObject();
}

auto hlir::DeleteOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::DestroyOp::print(OpAsmPrinter& p) {
    p << " " << getObject() << " dtor " << getDtorAttr();
}

auto hlir::DestroyOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::DirectBrOp::print(OpAsmPrinter& p) {
    p << " to " << getDest();
    if (not getProt().empty()) {
        p << " unwind ";
        bool first = true;
        for (auto prot : getProt()) {
            if (first) first = false;
            else p << ", ";
            p << prot;
        }
    }
}

auto hlir::DirectBrOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

auto hlir::FuncOp::parse(
    [[maybe_unused]] OpAsmParser& parser,
    [[maybe_unused]] OperationState& result
) -> ParseResult {
    Todo();
    /*/// Dispatch to function op interface.
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
    );*/
}

void hlir::FuncOp::print(OpAsmPrinter& p) {
    p << " " << getLinkage().getLinkage() << " ";
    if (getCc() != LLVM::CConv::C) p << getCc() << " ";

    p.printSymbolName(getName());

    auto ftype = getFunctionType();
    if (auto params = ftype.getNumInputs()) {
        p << "(";
        for (unsigned i = 0; i < params; i++) {
            if (i != 0) p << ", ";
            PrintType(ftype.getInput(i), p);
        }
        p << ")";
    }

    if (getSpecialMember()) p << " smf";
    if (getVariadic()) p << " variadic";

    if (ftype.getNumResults()) {
        Assert(ftype.getNumResults() == 1);
        p << " -> ";
        PrintType(ftype.getResult(0), p);
    }

    function_interface_impl::printFunctionAttributes(
        p,
        *this,
        {
            getFunctionTypeAttrName(),
            getArgAttrsAttrName(),
            getResAttrsAttrName(),
            getLinkageAttrName(),
            getCcAttrName(),
            getSpecialMemberAttrName(),
            getVariadicAttrName(),
            "sym_visibility",
        }
    );

    if (auto& body = getBody(); not body.empty()) {
        p << " ";
        p.printRegion(body, false, true);
    }
}

void hlir::InvokeClosureOp::print(OpAsmPrinter& p) {
    p << " " << getClosure() << "(";
    bool first = true;
    for (auto a : getArgs()) {
        if (first) first = false;
        else p << ", ";
        p << a;
    }
    p << ") -> " << getType();
}

auto hlir::InvokeClosureOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::LiteralOp::print(OpAsmPrinter& p) {
    if (auto s = dyn_cast<SliceType>(getType())) {
        p << " slice ref " << s.getElem() << " " << getOperand(0);
        p << ", " << getOperand(1);
    } else {
        Unreachable();
    }
}

auto hlir::LiteralOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::LoadOp::print(OpAsmPrinter& p) {
    p << " ";
    PrintType(getType().getType(), p);
    p << " from " << getOperand();
}

auto hlir::LoadOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::LocalOp::print(OpAsmPrinter& p) {
    if (getDtorFlag()) p << " flag";
    p << " ";
    PrintType(getType().getElem(), p);
    p << ", align " << getAlignment().getValue().getZExtValue();
}

auto hlir::LocalOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::MakeClosureOp::print(OpAsmPrinter& p) {
    p << " { @" << getProcedure();
    if (auto env = getEnv()) p << ", " << env << " }";
    else p << ", null }";
}

auto hlir::MakeClosureOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::NewOp::print(OpAsmPrinter& p) {
    p << " ";
    PrintType(getResult().getType().getElem(), p);
}

auto hlir::NewOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::NilOp::print(OpAsmPrinter& p) {
    p << " ";
    PrintType(getResult().getType(), p);
}

auto hlir::NilOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::NotOp::print(OpAsmPrinter& p) {
    p << " " << getOperand();
}

auto hlir::NotOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::PointerEqOp::print(OpAsmPrinter& p) {
    p << " " << getLhs() << ", " << getRhs();
}

auto hlir::PointerEqOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }


void hlir::PointerNeOp::print(OpAsmPrinter& p) {
    p << " " << getLhs() << ", " << getRhs();
}

auto hlir::PointerNeOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::ReturnOp::print(OpAsmPrinter& p) {
    if (getYield()) {
        p << " ";
        PrintType(getYield().getType(), p);
        p << " " << getYield();
    }

    if (not getProt().empty()) {
        p << " unwind ";
        bool first = true;
        for (auto prot : getProt()) {
            if (first) first = false;
            else p << ", ";
            p << prot;
        }
    }

    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        {getOperandSegmentSizesAttrName()}
    );
}

auto hlir::ReturnOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::ScopeOp::print(OpAsmPrinter& p) {
    p << " ";
    if (getRes()) {
        PrintType(getRes().getType(), p);
        p << " ";
    }

    p.printRegion(getBody(), false, true, false);
}

auto hlir::ScopeOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::SliceDataOp::print(OpAsmPrinter& p) {
    p << " ";
    PrintType(getOperand().getType(), p);
    p << " " << getOperand();
}

auto hlir::SliceDataOp::parse(OpAsmParser&, OperationState&) -> ParseResult {
    Todo();
}

void hlir::SliceSizeOp::print(OpAsmPrinter& p) {
    p << " ";
    PrintType(getOperand().getType(), p);
    p << " " << getOperand();
}

auto hlir::SliceSizeOp::parse(OpAsmParser&, OperationState&) -> ParseResult {
    Todo();
}

void hlir::StoreOp::print(OpAsmPrinter& p) {
    p << " into " << getAddr() << ", ";
    PrintType(getValue().getType(), p);
    p << " " << getValue();
    p << ", align " << getAlignment().getValue().getZExtValue();
}

auto hlir::StoreOp::parse(OpAsmParser&, OperationState&) -> ParseResult {
    Todo();
}

void hlir::StructGEPOp::print(OpAsmPrinter& p) {
    p << " " << getStructRef() << ", " << getIdx().getValue().getZExtValue();
    p << " -> ";
    PrintType(getType(), p);
}

auto hlir::StructGEPOp::parse(OpAsmParser&, OperationState&) -> ParseResult {
    Todo();
}

void hlir::UnreachableOp::print(OpAsmPrinter&) {}

auto hlir::UnreachableOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::BitCastOp::print(OpAsmPrinter& p) {
    p << " ";
    PrintType(getOperand().getType(), p);
    p << " " << getOperand() << " to ";
    PrintType(getType(), p);
}

auto hlir::BitCastOp::parse(OpAsmParser&, OperationState&) -> ParseResult {
    Todo();
}

void hlir::YieldOp::print(OpAsmPrinter& p) {
    if (getYield()) {
        p << " ";
        PrintType(getYield().getType(), p);
        p << " " << getYield();
    }

    if (not getProt().empty()) {
        p << " unwind ";
        bool first = true;
        for (auto prot : getProt()) {
            if (first) first = false;
            else p << ", ";
            p << prot;
        }
    }

    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        {getOperandSegmentSizesAttrName()}
    );
}

auto hlir::YieldOp::parse(OpAsmParser&, OperationState&) -> ParseResult { Todo(); }

void hlir::ZeroinitialiserOp::print(OpAsmPrinter& p) {
    p << " " << getOperand();
}

auto hlir::ZeroinitialiserOp::parse(OpAsmParser&, OperationState&) -> ParseResult {
    Todo();
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
    if (
        getNumOperands() < fnType.getNumInputs() or
        (getNumOperands() > fnType.getNumInputs() and not fn.getVariadic())
    ) return emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
        if (getOperand(i).getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                << fnType.getInput(i) << ", but provided "
                << getOperand(i).getType() << " for operand number " << i;

    if (getYield() and getYield().getType() != fnType.getResult(0)) {
        auto diag = emitOpError("result type mismatch");
        diag.attachNote() << "      op result types: " << getYield().getType();
        diag.attachNote() << "function result types: " << fnType.getResults();
        return diag;
    }

    return success();
}

/// ===========================================================================
///  Data Layout Interface.
/// ===========================================================================
#define DEFINE_DEFAULT_DATA_LAYOUT(type, size, align)                                               \
    auto hlir::type::getTypeSizeInBits(                                                             \
        const ::mlir::DataLayout&,                                                                  \
        ::mlir::DataLayoutEntryListRef                                                              \
    ) const -> llvm::TypeSize {                                                                     \
        return llvm::TypeSize::getFixed(size); /** FIXME: Use data layout. **/                      \
    }                                                                                               \
                                                                                                    \
    auto hlir::type::getTypeSize(                                                                   \
        const ::mlir::DataLayout& dl,                                                               \
        ::mlir::DataLayoutEntryListRef e                                                            \
    ) const -> llvm::TypeSize {                                                                     \
        return llvm::TypeSize::getFixed(src::utils::AlignTo<u64>(getTypeSizeInBits(dl, e), 8) / 8); \
    }                                                                                               \
                                                                                                    \
    auto hlir::type::getABIAlignment(                                                               \
        const ::mlir::DataLayout&,                                                                  \
        ::mlir::DataLayoutEntryListRef                                                              \
    ) const -> u64 {                                                                                \
        return align; /** FIXME: Use data layout. **/                                               \
    }                                                                                               \
                                                                                                    \
    auto hlir::type::getPreferredAlignment(                                                         \
        const ::mlir::DataLayout& dl,                                                               \
        ::mlir::DataLayoutEntryListRef e                                                            \
    ) const -> u64 {                                                                                \
        return getABIAlignment(dl, e);                                                              \
    }

DEFINE_DEFAULT_DATA_LAYOUT(SliceType, 128, 8)
DEFINE_DEFAULT_DATA_LAYOUT(ClosureType, 64, 8)
DEFINE_DEFAULT_DATA_LAYOUT(ReferenceType, 64, 8)
DEFINE_DEFAULT_DATA_LAYOUT(OptRefType, 64, 8)

auto hlir::ArrayType::getTypeSizeInBits(
    const ::mlir::DataLayout& dl,
    ::mlir::DataLayoutEntryListRef
) const -> llvm::TypeSize {
    return u64(getSize()) * dl.getTypeSizeInBits(getElem());
}

auto hlir::ArrayType::getTypeSize(
    const ::mlir::DataLayout& dl,
    ::mlir::DataLayoutEntryListRef e
) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(src::utils::AlignTo<u64>(getTypeSizeInBits(dl, e), 8) / 8);
}

auto hlir::ArrayType::getABIAlignment(
    const ::mlir::DataLayout& dl,
    ::mlir::DataLayoutEntryListRef
) const -> u64 {
    return dl.getTypeABIAlignment(getElem());
}

auto hlir::ArrayType::getPreferredAlignment(
    const ::mlir::DataLayout& dl,
    ::mlir::DataLayoutEntryListRef
) const -> u64 {
    return dl.getTypePreferredAlignment(getElem());
}
