#include <mlir/IR/IRMapping.h>
#include <source/CG/HLIRLowering.hh>
#include <source/Frontend/AST.hh>
#include <source/HLIR/HLIRDialect.hh>
#include <source/Support/Utils.hh>

namespace mlir::hlir {
namespace {
/*
/// Perform constructor call insertion.
struct CtorInsertXfrm : public OpRewritePattern<LocalOp> {
    CtorInsertXfrm(MLIRContext* ctx)
        : OpRewritePattern<LocalOp>(ctx, */
/*benefit=*//*1) {}

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
};*/

/// Inline a region before an operation.
///
/// If possible, no extra blocks will be inserted for the
/// region, and it will instead be spliced into the block
/// containing the operation.
///
/// This will *not* remove the operation from its parent,
/// even if it happens to be of the same type as the yield
/// instruction(s) of the region.
///
/// \tparam YieldOperation The yield instruction of the
///      inlined region, e.g. hlir::ReturnOp for functions.
/// \param rewriter The rewriter to use.
/// \param before The operation before which to inline.
/// \param region The region to inline.
/// \param replace_with_yield Whether to replace uses of
///     the operation with the region’s yield.
/// \param m Mapping between ops and region arguments.
template <typename YieldOperation>
void InlineRegion(
    RewriterBase& rewriter,
    Operation* before,
    Region* region,
    [[maybe_unused]] bool replace_with_yield = false,
    IRMapping* m = nullptr
) {
    /// Split block containing the op.
    auto first_half = before->getBlock();
    auto second_half = first_half->splitBlock(before);

    /// Inline the region inbetween.
    IRMapping default_mapping;
    rewriter.cloneRegionBefore(
        *region,
        *before->getParentRegion(),
        second_half->getIterator(),
        m ? *m : default_mapping
    );

    /// Merge blocks if the region contains only one block.
    if (llvm::hasSingleElement(*region)) {
        auto FindYield = [&](Operation& op) { return &op != before and isa<YieldOperation>(op); };

        /// Split blocks together.
        rewriter.mergeBlocks(&*std::next(first_half->getIterator()), first_half);
        rewriter.mergeBlocks(second_half, first_half);

        /// Get yield instruction.
        auto yield = src::rgs::find_if(first_half->getOperations(), FindYield);

        /// Replace the operation’s yield with the region’s yield.
        if (replace_with_yield) {
            Assert(yield != first_half->getOperations().end(), "Missing yield in region that yields value");
            rewriter.replaceAllUsesWith(before->getResult(0), yield->getOperand(0));
        }

        /// Erase the yield. It may be missing in some cases.
        if (yield != first_half->getOperations().end()) rewriter.eraseOp(&*yield);
        return;
    }

    /// Branch to the first block.
    rewriter.setInsertionPointToEnd(first_half);
    rewriter.create<cf::BranchOp>(
        before->getLoc(),
        ValueRange{},
        &*std::next(first_half->getIterator())
    );

    /// If the region yields a value, add a block argument for the return
    /// value to the second half and replace uses of the operation with
    /// that argument.
    if (replace_with_yield) {
        auto y = before->getResult(0);
        second_half->addArgument(y.getType(), before->getLoc());
        rewriter.replaceAllUsesWith(y, second_half->getArgument(0));
    }

    /// Replace yield ops with a branch to the second half.
    for (
        auto it = std::next(first_half->getIterator()), end = second_half->getIterator();
        it != end;
        it++
    ) {
        it->walk([&](Operation* op) {
            if (op != before and isa<YieldOperation>(op)) {
                rewriter.setInsertionPoint(op);
                rewriter.replaceOpWithNewOp<cf::BranchOp>(
                    op,
                    second_half,
                    op->getOperands()
                );
            }
        });
    }
}

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

        /// Map region arguments to call arguments.
        auto* region = &cast<FuncOp>(func).getBody();
        IRMapping m;
        for (src::u32 i = 0; i < region->getNumArguments(); i++)
            m.map(region->getArgument(i), call.getArgs()[i]);

        InlineRegion<hlir::ReturnOp>(
            rewriter,
            call,
            region,
            true,
            &m
        );

        rewriter.eraseOp(call);
        return success();
    }
};

struct CFGSimplifier {
    struct Scope {
        hlir::ScopeOp scope;
        SmallVector<DeferOp> deferred;
    };

    hlir::FuncOp f;
    std::vector<Scope> entered_scopes{};
    SmallVector<DeferOp> to_delete{};
    mlir::OpBuilder b{f->getContext()};
    mlir::IRRewriter rewriter{b};

    void run() {
        auto op = cast<ScopeOp>(&f.getBody().front().front());
        ProcessScope(op);
        for (auto d : to_delete) d->erase();
    }

private:
    auto CurrScope() -> Scope* {
        return &entered_scopes.back();
    }

    void ProcessScope(ScopeOp scope) {
        entered_scopes.emplace_back(scope);
        defer { entered_scopes.pop_back(); };

        /// Process the body of the scope.
        auto& region = scope.getBody();
        for (auto& block : llvm::make_early_inc_range(region.getBlocks())) {
            for (auto& op : llvm::make_early_inc_range(block.getOperations())) {
                if (auto y = dyn_cast<YieldOp>(op)) LowerYield(y);
                else if (auto r = dyn_cast<ReturnOp>(op)) LowerReturn(r);
                else if (auto s = dyn_cast<ScopeOp>(op)) ProcessScope(s);
                else if (auto d = dyn_cast<DeferOp>(op)) {
                    ProcessScope(d.getScopeOp());
                    CurrScope()->deferred.emplace_back(d);
                    op.remove();
                    to_delete.emplace_back(d);
                }
            }
        }
    }

    void EmitScope(Operation* before, Scope* sc) {
        for (auto d : src::vws::reverse(sc->deferred)) {
            InlineRegion<YieldOp>(
                rewriter,
                before,
                &d.getScopeOp().getBody(),
                false,
                nullptr
            );
        }
    }

    void LowerYield(YieldOp y) {
        if (y.getLowered()) return;
        y.setLowered(true);

        /// Emit deferred material.
        EmitScope(y, CurrScope());
    }

    void LowerReturn(ReturnOp r) {
        if (r.getLowered()) return;
        r.setLowered(true);

        /// Emit deferred material.
        for (auto& sc : src::vws::reverse(entered_scopes)) EmitScope(r, &sc);
    }
};

/// Inline control-flow related operations such as
/// scope, defer, yield.
void SimplfyCFG(hlir::FuncOp f) {
    CFGSimplifier s{f};
    s.run();
}

} // namespace
} // namespace mlir::hlir

void src::LowerHLIR(Module* mod) {
    for (auto f : mod->functions)
        if (f->body)
            mlir::hlir::SimplfyCFG(cast<hlir::FuncOp>(f->mlir_func));
}

void hlir::CallOp::getCanonicalizationPatterns(
    RewritePatternSet& results,
    MLIRContext* context
) {
    results.add<MandatoryInliningXfrm>(context);
}
