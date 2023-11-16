#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
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

struct DeferInliningXfrm {
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
        /// Find scope op.
        ScopeOp op;
        for (auto& block : f.getBody().getBlocks()) {
            for (auto& i : block.getOperations()) {
                if (auto o = dyn_cast<ScopeOp>(i)) {
                    op = o;
                    break;
                }
            }
        }

        if (not op) return;
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
        scope.getBody().walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (auto y = dyn_cast<YieldOp>(op)) LowerYield(y);
            else if (auto r = dyn_cast<ReturnOp>(op)) LowerReturn(r);
            else if (auto s = dyn_cast<ScopeOp>(op)) ProcessScope(s);
            else if (auto d = dyn_cast<DeferOp>(op)) {
                ProcessScope(d.getScopeOp());
                CurrScope()->deferred.emplace_back(d);
                op->remove();
                to_delete.emplace_back(d);
                return WalkResult::skip();
            }

            return WalkResult::advance();
        });
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

struct ScopeInliningXfrm {
    FuncOp f;
    OpBuilder b{f.getContext()};
    IRRewriter rewriter{b};

    SmallVector<ScopeOp> scopes{};

    void run() {
        /// Collect all scopes.
        f.getBody().walk([&](ScopeOp scope) { scopes.push_back(scope); });

        /// Inline them.
        for (auto sc : scopes) ProcessScope(sc);

        /// Erase blocks with no predecessors as this pass tends
        /// to create a bunch of them if some of the scopes contained
        /// early returns.
        for (auto& block : llvm::make_early_inc_range(f.getBlocks())) {
            if (&block == &f.getBody().front()) continue;
            if (block.hasNoPredecessors()) block.erase();
        }
    }

private:
    void ProcessScope(ScopeOp scope) {
        /// Split block if the scope is not the first operation.
        auto first = scope->getBlock();
        auto second = first->splitBlock(scope);

        /// Move the contents of the region before the second block.
        rewriter.inlineRegionBefore(scope.getBody(), second);

        /// Merge the first block of the region into the old first block.
        rewriter.mergeBlocks(&*std::next(first->getIterator()), first);

        /// If the scope has more than one yield, we need to convert each
        /// one to a branch and add a block argument to the second block.
        Block* last_block_in_region = &*std::prev(second->getIterator());
        if (scope.getEarlyYield()) {
            if (scope.getRes()) second->addArgument(scope.getRes().getType(), scope->getLoc());
            for (
                auto it = first->getIterator(), end = last_block_in_region->getIterator();
                it != end;
                ++it
            ) {
                for (auto& op : llvm::make_early_inc_range(it->getOperations())) {
                    auto y = dyn_cast<YieldOp>(op);
                    if (not y) continue;
                    rewriter.replaceOpWithNewOp<cf::BranchOp>(
                        y,
                        second,
                        y.getYield() ? mlir::ValueRange{y.getYield()} : mlir::ValueRange{}
                    );
                }
            }

            /// Replace uses of the scope’s value with the block argument.
            if (scope.getRes()) rewriter.replaceAllUsesWith(
                scope.getRes(),
                second->getArgument(0)
            );
        }

        /// Otherwise, only the last op in the region can be a yield.
        else if (auto y = dyn_cast<YieldOp>(last_block_in_region->back())) {
            /// Replace uses of the yield the yielded value, if there is one.
            if (scope.getRes()) rewriter.replaceAllUsesWith(
                scope.getRes(),
                y.getYield()
            );

            /// Yeet.
            y.erase();

            /// Now, if the block before second is not closed, merge second into it.
            if (
                last_block_in_region->empty() or
                not last_block_in_region->back().hasTrait<OpTrait::IsTerminator>()
            ) rewriter.mergeBlocks(second, last_block_in_region);
        }

        /// Lastly, yeet the empty scope.
        scope.erase();
    }
};

/// Inline defer ops in the appropriate places.
void InlineDefers(src::Module* mod) {
    for (auto f : mod->functions) {
        if (f->body) {
            DeferInliningXfrm s{cast<FuncOp>(f->mlir_func)};
            s.run();
        }
    }
}

/// Inline scope bodies into their parent regions.
void InlineScopes(src::Module* mod) {
    for (auto f : mod->functions) {
        if (f->body) {
            ScopeInliningXfrm s{cast<FuncOp>(f->mlir_func)};
            s.run();
        }
    }
}

} // namespace
} // namespace mlir::hlir

void src::LowerHLIR(Module* mod) {
    mlir::hlir::InlineDefers(mod);
    mlir::hlir::InlineScopes(mod);
}

void hlir::CallOp::getCanonicalizationPatterns(
    RewritePatternSet& results,
    MLIRContext* context
) {
    results.add<MandatoryInliningXfrm>(context);
}
