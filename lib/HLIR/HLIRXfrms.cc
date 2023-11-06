#include <mlir/IR/IRMapping.h>
#include <source/HLIR/HLIRDialect.hh>
#include <source/Support/Utils.hh>

namespace mlir::hlir {
namespace {
/// Perform constructor call insertion.
struct CtorInsertXfrm : public OpRewritePattern<LocalOp> {
    CtorInsertXfrm(MLIRContext* ctx)
        : OpRewritePattern<LocalOp>(ctx, /*benefit=*/1) {}

    auto matchAndRewrite(
        LocalOp op,
        PatternRewriter& rewriter
    ) const -> LogicalResult override {
        if (not op.getUninit()) return success();
        op.setUninit(false);
        rewriter.setInsertionPointAfter(op);
        rewriter.create<hlir::ZeroinitialiserOp>(
            op->getLoc(),
            op
        );
        return success();
    }
};

/// Perform inlining of calls annotated with `inline`.
struct MandatoryInliningXfrm : public OpRewritePattern<CallOp> {
    MandatoryInliningXfrm(MLIRContext* ctx)
        : OpRewritePattern<CallOp>(ctx, /*benefit=*/1) {}

    auto matchAndRewrite(
        CallOp call,
        PatternRewriter& rewriter
    ) const -> LogicalResult override {
        if (not call.getInlineCall()) return success();
        auto mod = call->getParentOfType<ModuleOp>();
        auto func = mod.lookupSymbol(call.getCallee());
        if (not func) return emitError(
            call.getLoc(),
            fmt::format("could not find callee '{}'", call.getCallee())
        );

        /// Split block containing the call.
        auto first_half = call->getBlock();
        auto second_half = first_half->splitBlock(call);

        /// Map region arguments to call arguments.
        auto* region = &cast<FuncOp>(func).getBody();
        IRMapping m;
        for (src::u32 i = 0; i < region->getNumArguments(); i++)
            m.map(region->getArgument(i), call.getArgs()[i]);

        /// Inline the region inbetween.
        rewriter.cloneRegionBefore(
            *region,
            *call->getParentRegion(),
            second_half->getIterator(),
            m
        );

        /// If the region contains only one block, we can splice
        /// all the blocks together and replace the yield of the
        /// call with the argument of the return expression.
        if (llvm::hasSingleElement(*region)) {
            rewriter.mergeBlocks(&*std::next(first_half->getIterator()), first_half);
            rewriter.mergeBlocks(second_half, first_half);
            auto FindRet = [](Operation& op) { return isa<hlir::ReturnOp>(op); };
            auto ret = src::rgs::find_if(first_half->getOperations(), FindRet);
            if (auto y = call.getYield()) rewriter.replaceAllUsesWith(y, ret->getOperand(0));
            rewriter.eraseOp(&*ret);
            rewriter.eraseOp(call);
            return success();
        }

        /// Branch to it.
        rewriter.setInsertionPointToEnd(first_half);
        rewriter.create<cf::BranchOp>(
            call->getLoc(),
            ValueRange{},
            &*std::next(first_half->getIterator())
        );

        /// Add a block argument for the return value to the second half
        /// and replace uses of the call with that argument.
        if (auto y = call.getYield()) {
            second_half->addArgument(y.getType(), call->getLoc());
            rewriter.replaceAllUsesWith(y, second_half->getArgument(0));
        }

        /// Replace return ops with a branch to the second half.
        for (
            auto it = std::next(first_half->getIterator()), end = second_half->getIterator();
            it != end;
            it++
        ) {
            it->walk([&](Operation* op) {
                if (auto ret = dyn_cast<hlir::ReturnOp>(op)) {
                    rewriter.setInsertionPoint(op);
                    rewriter.replaceOpWithNewOp<cf::BranchOp>(
                        op,
                        second_half,
                        op->getOperands()
                    );
                }
            });
        }

        /// Finally, erase the call.
        rewriter.eraseOp(call);
        return success();
    }
};
} // namespace
} // namespace src

void hlir::CallOp::getCanonicalizationPatterns(
    RewritePatternSet& results,
    MLIRContext* context
) {
    results.add<MandatoryInliningXfrm, CtorInsertXfrm>(context);
}