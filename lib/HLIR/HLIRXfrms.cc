#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
#include <source/CG/CodeGen.hh>
#include <source/Frontend/AST.hh>
#include <source/HLIR/HLIRDialect.hh>
#include <source/Support/Utils.hh>

namespace rgs = src::rgs;
namespace vws = src::vws;

namespace mlir::hlir {
namespace {
/// Find the NCA of two MLIR blocks.
[[maybe_unused]] auto NCA(Region* a, Region* b) -> Region* {
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

/// Create a loop.
///
/// Split the block before an operation and insert a loop at
/// its location; the operation will be left in the join block
/// of the loop.
///
/// This currently creates a loop from 0 to n (exclusive), but
/// can be amended if need be.
///
/// \param rewriter The rewriter to use.
/// \param op The operation before which the loop is created.
/// \param until Index until the loop should run (exclusive).
/// \param before_loop Callback run to insert code before the loop.
/// \param in_loop Callback run to emit the loop body. The current
///        value of the iteration variable is passed as an argument.
void CreateLoop(
    RewriterBase& rewriter,
    Operation* op,
    src::usz until,
    auto before_loop,
    auto in_loop
) {
    /// Split the block into two blocks so we create a loop here.
    auto loc = op->getLoc();
    auto this_block = op->getBlock();
    auto join_block = rewriter.splitBlock(this_block, op->getIterator());
    auto cond_block = rewriter.createBlock(join_block, IndexType::get(op->getContext()), loc);
    auto body_block = rewriter.createBlock(join_block);

    /// Add an index iterator with an initial value of zero.
    rewriter.setInsertionPointToEnd(this_block);
    auto zero = rewriter.create<index::ConstantOp>(loc, 0);
    auto end = rewriter.create<index::ConstantOp>(loc, until);
    std::invoke(before_loop);
    rewriter.create<cf::BranchOp>(loc, cond_block, mlir::ValueRange{zero});

    /// Check the index.
    rewriter.setInsertionPointToStart(cond_block);
    auto cond = rewriter.create<index::CmpOp>(
        loc,
        index::IndexCmpPredicate::EQ,
        cond_block->getArgument(0),
        end
    );

    /// Emit the loop body.
    rewriter.create<cf::CondBranchOp>(loc, cond, join_block, body_block);
    rewriter.setInsertionPointToStart(body_block);
    std::invoke(in_loop, cond_block->getArgument(0));

    /// Increment the iteration variable.
    auto one = rewriter.create<index::ConstantOp>(loc, 1);
    auto next = rewriter.create<index::AddOp>(loc, cond_block->getArgument(0), one);
    rewriter.create<cf::BranchOp>(loc, cond_block, mlir::ValueRange{next});
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
            } else if (auto x = dyn_cast<DestroyOp>(op)) {
                op->remove();
                to_delete.emplace_back(x);
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

        /// Destructors will be lowered to calls later on, so just copy
        /// them and insert them before the operation.
        if (auto d = dyn_cast<DestroyOp>(o)) {
            rewriter.setInsertionPoint(before);
            rewriter.clone(*o);
            return;
        }

        Unreachable();
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

struct DestroyOpLowering : public ConversionPattern {
    explicit DestroyOpLowering(MLIRContext* ctx)
        : ConversionPattern(hlir::DestroyOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        /// This is always just a function call.
        auto d = cast<hlir::DestroyOp>(op);
        rewriter.replaceOpWithNewOp<CallOp>(
            op,
            mlir::TypeRange{},
            d.getDtor(),
            false,
            LLVM::CConv::C,
            d.getObject()
        );
        return success();
    }
};

struct ConstructOpLowering : public ConversionPattern {
    explicit ConstructOpLowering(MLIRContext* ctx)
        : ConversionPattern(hlir::ConstructOp::getOperationName(), 1, ctx) {
    }

    void LowerTrivialCopyInit(
        ConstructOp c,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const {
        if (c.getArraySize() == 1) {
            rewriter.replaceOpWithNewOp<StoreOp>(
                c,
                args[0],
                args[1],
                DataLayout::closest(c).getTypeABIAlignment(c.getArgs()[0].getType())
            );

            return;
        }

        mlir::Value base = args[0];
        auto loc = c->getLoc();
        auto align = DataLayout::closest(c).getTypeABIAlignment(c.getArgs()[0].getType());
        auto EmitBase = [&] { /*base = rewriter.create<hlir::ArrayDecayOp>(loc, args[0]);*/ };
        auto EmitBody = [&](mlir::Value index) {
            auto addr = rewriter.create<OffsetOp>(
                loc,
                base,
                index
            );

            rewriter.create<StoreOp>(
                c->getLoc(),
                addr,
                args[1],
                align
            );
        };

        CreateLoop(rewriter, c, c.getArraySize(), EmitBase, EmitBody);
        rewriter.eraseOp(c);
    }

    void LowerInitCall(
        ConstructOp c,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const {
        if (c.getArraySize() == 1) {
            rewriter.replaceOpWithNewOp<CallOp>(
                c,
                mlir::TypeRange{},
                c.getCtor(),
                false,
                LLVM::CConv::C,
                args
            );
            return;
        }

        /// We need to replace only the first constructor argument.
        SmallVector<mlir::Value> ctor_args;
        ctor_args.insert(ctor_args.end(), args.begin(), args.end());


        mlir::Value base = args[0];
        auto loc = c->getLoc();
        auto EmitBase = [&] { /*args[0].dump(); base = rewriter.create<hlir::ArrayDecayOp>(loc, args[0]); */};
        auto EmitBody = [&](mlir::Value index) {
            auto addr = rewriter.create<OffsetOp>(
                loc,
                base,
                index
            );

            ctor_args[0] = addr;
            rewriter.create<CallOp>(
                loc,
                mlir::TypeRange{},
                c.getCtor(),
                false,
                LLVM::CConv::C,
                ctor_args
            );
        };

        CreateLoop(rewriter, c, c.getArraySize(), EmitBase, EmitBody);
        rewriter.eraseOp(c);
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        /// This is always just a function call.
        auto c = cast<hlir::ConstructOp>(op);
        switch (c.getInitKind()) {
            /// Separate op.
            case LocalInit::Zeroinit:
                rewriter.replaceOpWithNewOp<ZeroinitialiserOp>(op, args[0], c.getArraySize());
                return success();

            /// Store.
            case LocalInit::TrivialCopyInit:
                LowerTrivialCopyInit(c, args, rewriter);
                return success();

            /// Constructor call.
            case LocalInit::Init:
                LowerInitCall(c, args, rewriter);
                return success();
        }
        Unreachable();
    }
};

/// Inline defer ops in the appropriate places.
void InlineDefers(src::Module* mod) {
    for (auto& f : mod->mlir.getBodyRegion().getBlocks().front()) {
        if (auto func = dyn_cast<FuncOp>(f)) {
            DeferInliningXfrm s{mod->context, cast<FuncOp>(func)};
            s.run();
        }
    }
}

/// Inline scope bodies into their parent regions.
void InlineScopes(src::Module* mod) {
    for (auto& f : mod->mlir.getBodyRegion().getBlocks().front()) {
        if (auto func = dyn_cast<FuncOp>(f)) {
            ScopeInliningXfrm s{cast<FuncOp>(func)};
            s.run();
        }
    }
}
} // namespace

/// This only runs after initial control-flow lowering.
void AddLegalDialects(ConversionTarget& target);
struct HLIRXfrmPass
    : public PassWrapper<HLIRXfrmPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<LLVM::LLVMDialect, cf::ControlFlowDialect, index::IndexDialect>();
    }

    void runOnOperation() final {
        auto ctx = &getContext();
        ConversionTarget target{*ctx};
        RewritePatternSet patterns{ctx};
        AddLegalDialects(target);
        target.addIllegalOp<ConstructOp, DestroyOp>();
        patterns.add<ConstructOpLowering, DestroyOpLowering>(ctx);

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns)))) signalPassFailure();
    }
};

} // namespace mlir::hlir

void src::LowerHLIR(Module* mod) {
    /// These do not operate as a pass because individual operations interact
    /// in complex ways that a pass can’t model to well, at least not in my
    /// experience.
    mlir::hlir::InlineDefers(mod);
    mlir::hlir::InlineScopes(mod);

    /// Lowering that processes operations individually.
    mlir::PassManager pm{&mod->context->mlir};
    pm.addPass(std::make_unique<mlir::hlir::HLIRXfrmPass>());
    if (mlir::failed(pm.run(mod->mlir)))
        Diag::ICE(mod->context, mod->module_decl_location, "Module lowering failed");
}

void hlir::CallOp::getCanonicalizationPatterns(
    RewritePatternSet& results,
    MLIRContext* context
) {
    results.add<MandatoryInliningXfrm>(context);
}
