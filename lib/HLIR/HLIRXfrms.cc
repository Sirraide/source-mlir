#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <source/CG/HLIRLowering.hh>
#include <source/Frontend/AST.hh>
#include <source/HLIR/HLIRDialect.hh>
#include <source/Support/Utils.hh>

namespace rgs = src::rgs;
namespace vws = src::vws;

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

/// Find the NCA of two MLIR blocks.
auto NCA(Region* a, Region* b) -> Region* {
    llvm::SmallPtrSet<Region*, 8> scopes{};

    for (; a; a = a->getParentRegion()) {
        scopes.insert(a);
        if (isa<FuncOp>(a->getParentOp())) break;
    }

    for (; b; b = b->getParentRegion()) {
        if (scopes.contains(b)) return b;
        if (isa<FuncOp>(b->getParentOp())) break;
    }

    return nullptr;
}

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
        auto yield = rgs::find_if(first_half->getOperations(), FindYield);

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

/// Reduce control flow to a series of scopes and basic blocks.
///
/// This is a complicated pass that takes a function and resolves
/// high-level control-flow related concepts such as return, break
/// continue, goto, defer, and destructors and converts all of them
/// to just scopes containing basic blocks.
struct DeferInliningXfrm {
    src::Context* src_ctx;
    hlir::FuncOp f;
    SmallVector<DeferOp> to_delete{};
    mlir::OpBuilder _b{f->getContext()};
    mlir::IRRewriter rewriter{_b};

    void run() {
        /// Find scope op.
        ScopeOp op;
        for (auto& block : f.getBody().getBlocks()) {
            for (auto& i : block.getOperations()) {
                if (auto o = dyn_cast<ScopeOp>(i)) {
                    op = o;
                    goto dewit;
                }
            }
        }

        /// No scope op.
        return;

    dewit:
        /// Inline defers before jumps as appropriate.
        ProcessScope(op);
        for (auto d : to_delete) d->erase();
    }

private:
    template <typename... Args>
    auto Error(auto op, fmt::format_string<Args...> fmt, Args&&... args) -> WalkResult {
        auto loc = src::Location::Decode(op.getSrcLoc());
        src::Diag::Error(src_ctx, loc, fmt, std::forward<Args>(args)...);
        return WalkResult::interrupt();
    }

    void ProcessScope(ScopeOp scope) {
        /// Process the body of the scope.
        scope.getBody().walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (auto y = dyn_cast<YieldOp>(op)) LowerYield(y);
            else if (auto r = dyn_cast<ReturnOp>(op)) LowerReturn(r);
            else if (auto s = dyn_cast<ScopeOp>(op)) ProcessScope(s);
            else if (auto b = dyn_cast<DirectBrOp>(op)) {
                LowerDirectBranch(b);
                return WalkResult::skip();
            } else if (auto d = dyn_cast<DeferOp>(op)) {
                ProcessScope(d.getScopeOp());
                op->remove();
                to_delete.emplace_back(d);
                return WalkResult::skip();
            }

            return WalkResult::advance();
        });
    }

    void Emit(Operation* before, Operation* o) {
        if (auto d = dyn_cast<DeferOp>(o)) {
            InlineRegion<YieldOp>(
                rewriter,
                before,
                &d.getScopeOp().getBody(),
                false,
                nullptr
            );
            return;
        }

        auto l = cast<LocalOp>(o);
        rewriter.setInsertionPoint(before);
        rewriter.create<DestroyOp>(l->getLoc(), l);
    }

    void LowerYield(YieldOp y) {
        /// Emit deferred material.
        for (auto prot : y.getProt()) Emit(y, prot.getDefiningOp());
        y.getProtMutable().clear();
    }

    void LowerReturn(ReturnOp r) {
        /// Emit deferred material.
        for (auto prot : r.getProt()) Emit(r, prot.getDefiningOp());
        r.getProtMutable().clear();
    }

    /// Break, continue, goto.
    void LowerDirectBranch(DirectBrOp b) {
        for (auto prot : b.getProt()) Emit(b, prot.getDefiningOp());
        rewriter.setInsertionPoint(b);
        rewriter.create<cf::BranchOp>(b->getLoc(), b.getDest());
        rewriter.eraseOp(b);

        /*/// The expressions to unwind are only computed during sema if this
        /// is a goto. For break/continue, it’s simpler if we just do that
        /// here. That works well because the semantics for unwinding break
        /// and continue are much simpler than that of goto.
        ///
        /// Thus, if we have no protected expressions, then this is may be a
        /// break or continue, so unwind now if need be.
        if (auto p = b.getProt(); p.empty()) {
            /// First, find the NCA of the source and destination blocks.
            auto src = b->getBlock();
            auto dst = b.getDest();
            auto nca = NCA(src->getParent(), dst->getParent());
            Assert(nca);

            /// Emit scopes up until we’re at the NCA.
            auto it = rgs::find_if(entered_scopes, [&](Scope& s) { return &s.scope.getBody() == nca; });
            Assert(it != entered_scopes.end());
            for (auto& s : rgs::subrange(std::next(it), entered_scopes.end()) | vws::reverse)
                EmitScope(b, &s);
        }

        /// Emit any protected expressions. Note that codegen has
        /// already reversed the order of these.
        else {
        }*/
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
            DeferInliningXfrm s{mod->context, cast<FuncOp>(f->mlir_func)};
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
