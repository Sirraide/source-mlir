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
#include <utils.hh>

using namespace mlir;
namespace {
/// Create a string.
Value CreateStringLiteral(
    Location loc,
    OpBuilder& builder,
    StringRef value,
    ModuleOp module
) {
    LLVM::GlobalOp op;

    {
        OpBuilder::InsertionGuard i{builder};
        builder.setInsertionPointToStart(module.getBody());

        auto string_type = LLVM::LLVMArrayType::get(
            IntegerType::get(builder.getContext(), 8),
            u32(value.size())
        );

        op = builder.create<LLVM::GlobalOp>(
            loc,
            string_type,
            true,
            LLVM::Linkage::Private,
            "asdf",
            builder.getStringAttr(value),
            0
        );
    }

    Value ptr = builder.create<LLVM::AddressOfOp>(loc, op);
    Value zero = builder.create<LLVM::ConstantOp>(
        loc,
        builder.getI64Type(),
        builder.getIndexAttr(0)
    );

    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        ptr,
        ArrayRef<Value>{zero, zero}
    );
}

struct StringOpLowering : public ConversionPattern {
    explicit StringOpLowering(MLIRContext* ctx)
        : ConversionPattern(hlir::StringOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        Value s = CreateStringLiteral(
            op->getLoc(),
            rewriter,
            cast<hlir::StringOp>(op).getValue(),
            op->getParentOfType<ModuleOp>()
        );

        rewriter.replaceOp(op, s);
        return success();
    }
};

struct PrintOpLowering : public ConversionPattern {
    explicit PrintOpLowering(MLIRContext* ctx)
        : ConversionPattern(hlir::PrintOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto puts_type = LLVM::LLVMFunctionType::get(
            rewriter.getI32Type(),
            {LLVM::LLVMPointerType::get(rewriter.getI8Type())},
            false
        );

        LLVM::LLVMFuncOp puts;
        {
            OpBuilder::InsertionGuard i{rewriter};
            rewriter.setInsertionPointToStart(op->getParentOfType<ModuleOp>().getBody());
            puts = rewriter.create<LLVM::LLVMFuncOp>(
                op->getLoc(),
                "puts",
                puts_type
            );
        }

        rewriter.create<LLVM::CallOp>(
            op->getLoc(),
            puts,
            ArrayRef<Value>{operands[0]}
        );

        rewriter.eraseOp(op);
        return success();
    }
};

struct HLIRToLLVMLoweringPass
    : public PassWrapper<HLIRToLLVMLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }

    void runOnOperation() final {
        LLVMConversionTarget target{getContext()};
        target.addLegalOp<ModuleOp>();

        LLVMTypeConverter tc{&getContext()};
        RewritePatternSet patterns{&getContext()};
        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        arith::populateArithToLLVMConversionPatterns(tc, patterns);
        populateMemRefToLLVMConversionPatterns(tc, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(tc, patterns);
        populateFuncToLLVMConversionPatterns(tc, patterns);
        patterns.add<StringOpLowering, PrintOpLowering>(&getContext());

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};
} // namespace

auto hlir::CreateLowerToLLVMPass() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<HLIRToLLVMLoweringPass>();
}