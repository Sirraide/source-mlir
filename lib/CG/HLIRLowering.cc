#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <source/CG/HLIRLowering.hh>
#include <source/HLIR/HLIRDialect.hh>
#include <source/Support/Utils.hh>

using namespace mlir;
namespace {
/// Lowering for string literals.
struct StringOpLowering : public ConversionPattern {
    explicit StringOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::StringOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto module = op->getParentOfType<ModuleOp>();
        auto str_op = cast<hlir::StringOp>(op);
        auto loc = op->getLoc();

        /// Create global string.
        OpBuilder::InsertionGuard i{rewriter};
        rewriter.setInsertionPointToStart(module.getBody());

        auto string_type = LLVM::LLVMArrayType::get(
            IntegerType::get(getContext(), 8),
            src::u32(str_op.getValue().size())
        );

        auto global = rewriter.create<LLVM::GlobalOp>(
            loc,
            string_type,
            true,
            LLVM::Linkage::Private,
            fmt::format(".str.data.{}", str_op.getIndex().getZExtValue()),
            str_op.getValueAttr(),
            0
        );

        /// Create a slice to wrap the string.
        auto ty = getTypeConverter()->convertType(str_op.getType().getType());
        auto slice_global = rewriter.create<LLVM::GlobalOp>(
            loc,
            ty,
            true,
            LLVM::Linkage::Private,
            fmt::format(".str.{}", str_op.getIndex().getZExtValue()),
            Attribute{},
            0
        );

        slice_global.getInitializerRegion().push_back(new Block);
        auto& block = slice_global.getInitializerRegion().front();
        rewriter.setInsertionPointToEnd(&block);

        /// Get string data pointer.
        Value ptr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        Value zero = rewriter.create<LLVM::ConstantOp>(
            loc,
            getTypeConverter<LLVMTypeConverter>()->getIndexType(),
            rewriter.getIndexAttr(0)
        );

        auto data_ptr = rewriter.create<LLVM::GEPOp>(
            loc,
            LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)),
            ptr,
            ArrayRef<Value>{zero, zero}
        );

        /// Get string size.
        auto size = rewriter.create<LLVM::ConstantOp>(
            loc,
            getTypeConverter<LLVMTypeConverter>()->getIndexType(),
            rewriter.getIndexAttr(src::i64(str_op.getValue().size()))
        );

        /// Create a poison slice and insert the data pointer and size.
        auto slice = rewriter.create<LLVM::UndefOp>(loc, ty);
        rewriter.create<LLVM::InsertValueOp>(loc, slice, data_ptr, ArrayRef<src::i64>{0});
        rewriter.create<LLVM::InsertValueOp>(loc, slice, size, ArrayRef<src::i64>{1});

        /// Store the slice in the global.
        rewriter.create<LLVM::ReturnOp>(loc, slice);

        /// Delete the string instruction.
        op->erase();
        return success();
    }
};

/// Lowering for accessing the data pointer of a slice.
struct SliceDataOpLowering : public ConversionPattern {
    explicit SliceDataOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::SliceDataOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto slice = cast<hlir::SliceDataOp>(op);
        auto loc = op->getLoc();

        /// Get the data pointer.
        auto data_ptr = rewriter.create<LLVM::ExtractValueOp>(
            loc,
            getTypeConverter()->convertType(slice.getType().getType()),
            operands[0],
            ArrayRef<src::i64>{0}
        );

        /// Replace the slice data op with the data pointer.
        rewriter.replaceOp(op, data_ptr);
        return success();
    }
};

/// Lowering for accessing the size of a slice.
struct SliceSizeOpLowering : public ConversionPattern {
    explicit SliceSizeOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::SliceSizeOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto slice = cast<hlir::SliceSizeOp>(op);
        auto loc = op->getLoc();

        /// Get the size.
        auto size = rewriter.create<LLVM::ExtractValueOp>(
            loc,
            getTypeConverter()->convertType(slice.getType().getType()),
            operands[0],
            ArrayRef<src::i64>{1}
        );

        /// Replace the slice size op with the size.
        rewriter.replaceOp(op, size);
        return success();
    }
};

/// Lowering for global refs.
struct GlobalRefOpLowering : public ConversionPattern {
    explicit GlobalRefOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::GlobalRefOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto global_ref = cast<hlir::GlobalRefOp>(op);
        auto loc = op->getLoc();

        /// Get the global pointer.
        auto global_ptr = rewriter.create<LLVM::AddressOfOp>(
            loc,
            getTypeConverter()->convertType(operands[0].getType()),
            global_ref.getName().getLeafReference().getValue()
        );

        /// Replace the global ref op with the global pointer.
        rewriter.replaceOp(op, global_ptr);
        return success();
    }
};

/// Lowering for loads.
struct LoadOpLowering : public ConversionPattern {
    explicit LoadOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::LoadOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto load = cast<hlir::LoadOp>(op);
        auto loc = op->getLoc();

        /// Load the value.
        auto value = rewriter.create<LLVM::LoadOp>(
            loc,
            getTypeConverter()->convertType(load.getType().getType()),
            operands[0]
        );

        /// Replace the load op with the loaded value.
        rewriter.replaceOp(op, value);
        return success();
    }
};

} // namespace

/*
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
};*/

namespace src {
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

        /// Convert slice types to structs of ptr + index.
        tc.addConversion([&](hlir::SliceType t) {
            auto elem = tc.convertType(t.getElem());
            return LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {LLVM::LLVMPointerType::get(elem), tc.getIndexType()}
            );
        });

        /// Convert reference types to ptr.
        tc.addConversion([&](hlir::ReferenceType ref) {
            return LLVM::LLVMPointerType::get(tc.convertType(ref.getElem()));
        });

        /// Convert none to void.
        tc.addConversion([&](NoneType) {
            return LLVM::LLVMVoidType::get(&getContext());
        });

        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        arith::populateArithToLLVMConversionPatterns(tc, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(tc, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(tc, patterns);
        populateFuncToLLVMConversionPatterns(tc, patterns);
        populateFuncToLLVMFuncOpConversionPattern(tc, patterns);
        index::populateIndexToLLVMConversionPatterns(tc, patterns);

        // clang-format off
        patterns.add<
            StringOpLowering,
            SliceDataOpLowering,
            SliceSizeOpLowering,
            GlobalRefOpLowering,
            LoadOpLowering
        >(&getContext(), tc);
        // clang-format on

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            fmt::print("Error converting module:");
            module.print(llvm::errs());
            signalPassFailure();
        }
    }
};
} // namespace src

void src::LowerToLLVM(Module* mod) {
    /// Lower the module.
    mlir::PassManager pm{&mod->context->mlir};
    pm.addPass(std::make_unique<HLIRToLLVMLoweringPass>());
    if (mlir::failed(pm.run(mod->mlir)))
        Diag::ICE(mod->context, mod->module_decl_location, "Module lowering failed");

    /// Convert to LLVM IR.
    mod->llvm = mlir::translateModuleToLLVMIR(mod->mlir, mod->context->llvm);
}

void src::Module::print_llvm() const {
    llvm->print(llvm::outs(), nullptr);
}
