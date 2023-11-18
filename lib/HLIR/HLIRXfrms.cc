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

struct DeferInliningXfrm {
    struct Scope {
        using Entry = std::pair<DeferOp, Block*>;
        hlir::ScopeOp scope;
        SmallVector<Entry> deferred;
    };

    src::Context* src_ctx;
    hlir::FuncOp f;
    std::vector<Scope> entered_scopes{};
    SmallVector<DeferOp> to_delete{};
    mlir::OpBuilder _b{f->getContext()};
    mlir::IRRewriter rewriter{_b};

    /// Blocks that may not be crossed by a jump together
    /// with the instruction that prevents such crossing.
    DenseMap<Block*, Operation*> no_crossing{};

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

        /// No scope op, for some reason.
        if (not op) return;

        /// Collect all local ops and defer ops to determine what
        /// blocks may be crossed by a goto.
        InitScope(op);

        /// Inline defers before jumps as appropriate.
        ProcessScope(op);

        /// Finally, yeet defer ops.
        for (auto d : to_delete) d->erase();
    }

private:
    auto CurrScope() -> Scope* {
        return &entered_scopes.back();
    }

    template <typename... Args>
    auto Error(auto op, fmt::format_string<Args...> fmt, Args&&... args) -> WalkResult {
        auto loc = src::Location::Decode(op.getSrcLoc());
        src::Diag::Error(src_ctx, loc, fmt, std::forward<Args>(args)...);
        return WalkResult::interrupt();
    }

    template <typename... Args>
    auto Note(auto op, fmt::format_string<Args...> fmt, Args&&... args) -> WalkResult {
        auto loc = src::Location::Decode(op.getSrcLoc());
        src::Diag::Note(src_ctx, loc, fmt, std::forward<Args>(args)...);
        return WalkResult::interrupt();
    }

    void InitScope(ScopeOp scope) {
        /// Process the body of the scope.
        scope.getBody().walk([&](Operation* op) {
            if (auto l = dyn_cast<LocalOp>(op)) no_crossing.try_emplace(l->getBlock(), l);
            else if (auto d = dyn_cast<DeferOp>(op)) {
                InitScope(d.getScopeOp());
                no_crossing.try_emplace(d->getBlock(), d);
            }
        });
    }

    void ProcessScope(ScopeOp scope) {
        entered_scopes.emplace_back(scope);
        defer { entered_scopes.pop_back(); };

        /// Process the body of the scope.
        scope.getBody().walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (auto y = dyn_cast<YieldOp>(op)) LowerYield(y);
            else if (auto r = dyn_cast<ReturnOp>(op)) LowerReturn(r);
            else if (auto b = dyn_cast<DirectBrOp>(op)) return LowerDirectBranch(b);
            else if (auto s = dyn_cast<ScopeOp>(op)) ProcessScope(s);
            else if (auto d = dyn_cast<DeferOp>(op)) {
                ProcessScope(d.getScopeOp());
                CurrScope()->deferred.emplace_back(d, d->getBlock());
                // op->remove();
                to_delete.emplace_back(d);
                return WalkResult::skip();
            }

            return WalkResult::advance();
        });
    }

    void EmitScope(Operation* before, Scope* sc) {
        for (auto [d, _] : vws::reverse(sc->deferred)) EmitDefer(before, d);
    }

    void EmitDefer(Operation* before, DeferOp d) {
        InlineRegion<YieldOp>(
            rewriter,
            before,
            &d.getScopeOp().getBody(),
            false,
            nullptr
        );
    }

    auto IsBlock(Block* b) {
        return [b](auto&& p) { return p.second == b; };
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
        for (auto& sc : vws::reverse(entered_scopes)) EmitScope(r, &sc);
    }

    /// Break, continue, goto.
    auto LowerDirectBranch(DirectBrOp b) -> WalkResult {
        /// There are six possible cases here:
        ///
        ///   1. Near forwards jump to a block in the same scope.
        ///   2. Near backwards jump to a block in the same scope.
        ///   3. Far backwards jump to a block in a parent scope.
        ///   4. Far forwards jump to a block in a parent scope.
        ///   5. Upward cross jump into a child of a parent scope.
        ///   6. Downward cross jump into a child of the current scope.
        ///
        /// For each case, we have to determine what deferred material
        /// to emit and whether the jump is even legal in the first
        /// place. In general, forwards jumps are legal, iff they do not
        /// cross deferred material or variable declarations. Backwards
        /// jumps are legal are always legal.
        auto src = b->getBlock();
        auto dst = b.getDest();

        /// A jump from a block to itself is always legal.
        if (src == dst) {
            Assert(b.getOperation() == &src->back(), "Stray branch in block");

            /// Emit any defer ops in that block. We need to emit all defer
            /// ops because a branch from a block to itself must obviously
            /// be the last instruction in the block.
            for (auto [d, block] : CurrScope()->deferred | vws::reverse | vws::filter(IsBlock(src)))
                if (block == src)
                    EmitDefer(&src->back(), d);

            /// And branch to it.
            rewriter.setInsertionPoint(b);
            rewriter.create<cf::BranchOp>(b->getLoc(), b.getDest());
            rewriter.eraseOp(b);
            return WalkResult::skip();
        }

        /// Cases 1/2: Same scope.
        if (src->getParent() == dst->getParent()) {
            DominanceInfo dom{src->getParent()->getParentOp()};
            auto& tree = dom.getDomTree(src->getParent());
            auto IDom = [&](Block* block) {
                auto idom = tree.getNode(block)->getIDom();
                return idom ? idom->getBlock() : nullptr;
            };

            /// If src dominates dst, the branch is valid, if src dominates
            /// no deferred operations or variable declarations that dst does
            /// not dominate.
            if (dom.properlyDominates(src, dst)) {
                auto BranchMayCross = [&](Block* block) {
                    auto ReportIllegalJump = [&](Operation* op) {
                        Error(b, "Illegal jump target");
                        if (auto d = dyn_cast<DeferOp>(op)) Note(d, "Jump bypasses deferred expression here");
                        else if (auto l = dyn_cast<LocalOp>(op)) Note(l, "Jump bypasses variable declaration here");
                        return false;
                    };

                    /// If this is the block we’re branching from, only check
                    /// operations after the branch.
                    if (block == src) {
                        auto range = rgs::subrange(std::next(b->getIterator()), block->end());
                        auto it = rgs::find_if(range, [] (auto& op) { return isa<DeferOp, LocalOp>(op); });
                        if (it == range.end()) return true;
                        else return ReportIllegalJump(&*it);
                    }

                    /// Otherwise, check if the block is illegal to cross.
                    auto it = no_crossing.find(block);
                    if (it == no_crossing.end()) return true;
                    return ReportIllegalJump(it->second);
                };

                /// Walk up the dominator tree till we get to dst.
                for (Block *i = IDom(dst), *end = IDom(src); i and i != end; i = IDom(i))
                    if (not BranchMayCross(i))
                        return WalkResult::interrupt();
            }

            /// Otherwise, this is a backwards jump. These are always fine, but we
            /// have to unwind any deferred material between the src and dest.
            else {
                for (Block *i = IDom(dst), *end = IDom(src); i and i != end; i = IDom(i))
                    for (auto [d, _] : CurrScope()->deferred | vws::filter(IsBlock(i)) | vws::reverse)
                        EmitDefer(b, d);
            }

            /// Finally, lower the branch.
            rewriter.setInsertionPoint(b);
            rewriter.create<cf::BranchOp>(b->getLoc(), b.getDest());
            rewriter.eraseOp(b);
            return WalkResult::skip();
        }

        /// Case 3/4. Upwards jump.
        Region* target = src->getParent();
        for (Region* end = dst->getParent(); target and target != end; target = target->getParentRegion()) {}
        if (target) {
            auto scope = cast<ScopeOp>(target->getParentOp());

            /// Leave all scopes inbetween.
            for (auto sc : entered_scopes | vws::reverse) {
                if (sc.scope == scope) break;
                for (auto [d, _] : sc.deferred | vws::reverse)
                    EmitDefer(b, d);
            }

            Todo();
        }

        Todo();
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
