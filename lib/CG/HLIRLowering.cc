#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

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

namespace src {
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
            u32(str_op.getValue().size())
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

        rewriter.replaceOp(op, global);
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
            getTypeConverter()->convertType(slice.getRes().getType()),
            operands[0],
            ArrayRef<i64>{0}
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
            getTypeConverter<LLVMTypeConverter>()->getIndexType(),
            operands[0],
            ArrayRef<i64>{1}
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
            getTypeConverter()->convertType(global_ref.getRes().getType()),
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

/// Lowering for literals.
struct LiteralOpLowering : public ConversionPattern {
    explicit LiteralOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::LiteralOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto literal = cast<hlir::LiteralOp>(op);
        auto loc = op->getLoc();

        /// How the literal is lowered depends on the type. Slice
        /// literals have a data pointer and a size.
        if (isa<hlir::SliceType>(literal.getValue().getType())) {
            /// Create a poison slice and insert the data pointer and size.
            auto ty = getTypeConverter()->convertType(literal.getValue().getType());
            auto s0 = rewriter.create<LLVM::UndefOp>(loc, ty);
            auto s1 = rewriter.create<LLVM::InsertValueOp>(loc, s0, operands[0], ArrayRef<i64>{0});
            auto s2 = rewriter.create<LLVM::InsertValueOp>(loc, s1, operands[1], ArrayRef<i64>{1});
            rewriter.replaceOp(op, s2);
            return success();
        }

        /// Invalid literal type.
        else {
            return failure();
        }
    }
};

/// Lowering for array decay ops.
struct ArrayDecayOpLowering : public ConversionPattern {
    explicit ArrayDecayOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::ArrayDecayOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto bitcast = cast<hlir::ArrayDecayOp>(op);
        auto loc = op->getLoc();

        /// Ref-to-array to ref-to-elem casts are GEPs.
        auto gep = rewriter.create<LLVM::GEPOp>(
            loc,
            getTypeConverter()->convertType(bitcast.getType()),
            operands[0],
            ArrayRef{mlir::LLVM::GEPArg(0), mlir::LLVM::GEPArg(0)},
            true
        );

        rewriter.replaceOp(op, gep);
        return success();
    }
};

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
            Assert(elem, "Slice type has invalid element type", (t.dump(), 0));
            return LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {LLVM::LLVMPointerType::get(elem), tc.getIndexType()}
            );
        });

        /// Convert reference types to ptr.
        tc.addConversion([&](hlir::ReferenceType ref) {
            auto elem = tc.convertType(ref.getElem());
            Assert(elem, "Reference type has invalid element type", (ref.dump(), 0));
            return LLVM::LLVMPointerType::get(elem);
        });

        /// Convert array types to arrays.
        tc.addConversion([&](hlir::ArrayType arr) {
            auto elem = tc.convertType(arr.getElem());
            Assert(elem, "Array type has invalid element type", (arr.dump(), 0));
            return LLVM::LLVMArrayType::get(elem, unsigned(arr.getSize()));
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
            LoadOpLowering,
            LiteralOpLowering,
            ArrayDecayOpLowering
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

void src::LowerToLLVM(Module* mod, bool debug_llvm_lowering) {
    /// Lower the module.
    if (debug_llvm_lowering) mod->context->mlir.disableMultithreading();
    mlir::PassManager pm{&mod->context->mlir};
    pm.addPass(std::make_unique<HLIRToLLVMLoweringPass>());
    if (debug_llvm_lowering) pm.enableIRPrinting();
    if (mlir::failed(pm.run(mod->mlir)))
        Diag::ICE(mod->context, mod->module_decl_location, "Module lowering failed");
}

void src::Module::print_llvm() {
    if (not llvm) llvm = mlir::translateModuleToLLVMIR(mlir, context->llvm);
    llvm->print(llvm::outs(), nullptr);
}

int src::Module::run(int opt_level) {
    Assert(not is_logical_module, "Module is not executable");
    Assert(mlir, "Must codegen module before executing");

    /// Create optimiser.
    auto engine = mlir::ExecutionEngine::create(
        mlir,
        {
            .jitCodeGenOptLevel = llvm::CodeGenOpt::Level(std::clamp(opt_level, 0, 3)),
        }
    );
    Assert(engine, "Failed to create execution engine");

    /// Invoke __src_main.
    auto result = engine.get()->invokePacked("__src_main");
    if (result) {
        Diag::Error(context, {}, "Execution failed: {}", llvm::toString(std::move(result)));
        return 1;
    }
    return 0;
}
