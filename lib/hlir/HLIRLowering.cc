#include <hlir/HLIRDialect.hh>
#include <hlir/HLIRLowering.hh>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>

using namespace mlir;
namespace {
/*struct StringOpLowering final : public ConversionPattern {
    explicit StringOpLowering(MLIRContext* ctx)
        : ConversionPattern(hlir::PrintOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        Value s = getOrCreateGlobalString(
            op->getLoc(),
            rewriter,
            "str",
            StringAttr::get(
                op->getAttrOfType<StringAttr>("value").getValue(),
                op->getContext()
            )
        );
    }
};*/

struct HLIRToLLVMLoweringPass : public PassWrapper<HLIRToLLVMLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }

    void runOnOperation() override {
        LLVMConversionTarget target{getContext()};
        LLVMTypeConverter tc{&getContext()};
        RewritePatternSet patterns{&getContext()};

        target.addLegalOp<ModuleOp>();
        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        arith::populateArithToLLVMConversionPatterns(tc, patterns);
        populateMemRefToLLVMConversionPatterns(tc, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(tc, patterns);
        populateFuncToLLVMConversionPatterns(tc, patterns);
        //patterns.add<StringOpLowering, PrintOpLowering>();

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};
} // namespace

auto hlir::CreateLowerToLLVMPass() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<HLIRToLLVMLoweringPass>();
}