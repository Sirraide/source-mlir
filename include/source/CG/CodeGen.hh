#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <mlir/IR/Builders.h>
#include <source/Core.hh>
#include <source/HLIR/HLIRDialect.hh>

namespace src {
class WhileExpr;
class BinaryExpr;
class LocalDecl;
class LabelExpr;

class CodeGen {
    Module* const mod;
    Context* const ctx;
    mlir::MLIRContext* const mctx;
    mlir::OpBuilder builder;
    bool no_verify;
    usz anon_structs = 0;

    /// \brief Subsystem for managing deferred material and destructors.
    ///
    /// This ‘defer stack’ is actual not a single stack, but rather
    /// a *stack of stacks of stacks*: at the top-level, a stack holds
    /// entries for each scope in a function. This outermost stack
    /// is ‘the defer stack’.
    ///
    /// Each scope is further divided into regions, delineated by
    /// labels that are branched to by `goto`s, each of which is
    /// associated with a ‘stacklet’ that holds deferred material
    /// for that region. The former stacks are called ‘scope stacks’.
    ///
    /// Below, ‘the procedure proper’ refers to the parts of a proc
    /// that are not part of a deferred expression.
    class DeferInfo {
        /// Stacklet for a region that holds deferred material.
        struct Stacklet {
            LabelExpr* label{};
            SmallVector<Expr*> deferred_material{};
            mlir::func::FuncOp compacted{};
            usz vars_count{};
        };

        /// Stack of stacklets for an entire scope.
        struct Stack {
            SmallVector<Stacklet> stacklets{1};
            WhileExpr* scope_tag{};
        };

        /// Codegen context.
        CodeGen& CG;

        /// Stack of defer stacks for an entire function. This
        /// is a deque because we may end up creating new defer
        /// stacks while we’re emitting content stored in others.
        std::deque<Stack> stacks;

        /// Counter for emitting procedures.
        usz defer_procs{};

        /// Variables that have been emitted so far.
        SmallVector<mlir::Value> vars;

    public:
        /// RAII guard for entering a new block.
        ///
        /// This takes care of allocating a new defer stack, resetting
        /// tracked local variables, and emitting the defer stack that
        /// is added for this block.
        class BlockGuard {
            DeferInfo& DI;
            const std::size_t vars_count;
            mlir::Location location;

        public:
            BlockGuard(DeferInfo& DI, mlir::Location loc);
            ~BlockGuard();
        };

        /// RAII guard for entering a loop.
        class LoopGuard {
            DeferInfo& DI;

        public:
            LoopGuard(DeferInfo& DI, WhileExpr* loop);
            ~LoopGuard();
        };

        /// Initialise defer info.
        DeferInfo(CodeGen& CG) : CG(CG) {}

        /// Add an expression to be executed at the end of the scope.
        void AddDeferredExpression(Expr* e);

        /// Add a label to the stack.
        void AddLabel(LabelExpr* e);

        /// Add a local variable.
        void AddLocal(mlir::Value val);

        /// Emit all defer stacks up to a labelled expression.
        void EmitDeferStacksUpTo(Expr* stop_at);

        /// Emit all defer stacks for a return expression.
        void Return(bool last_instruction_in_function);

    private:
        /// Call a function that executes deferred expressions.
        void CallCleanupFunc(mlir::func::FuncOp func);

        /// Compact a stacklet to allow executing it multiple times.
        void Compact(Stacklet& s);

        /// Get the current defer stack.
        auto CurrentStack() -> Stack&;

        /// Emit the contents of a stack. Returns whether the
        /// expression we should stop at was found.
        bool Emit(Stack& s, bool compact, Expr* stop_at);

        /// Emit the contents of a stacklet.
        void Emit(Stacklet& s);

        /// Iterate over all defer stacks, in reverse.
        auto Iterate(Expr* stop_at);
    };

    /// Defer stack.
    ///
    /// We only need one because nested defer blocks will simply
    /// continue using this one because a defer block is always a
    /// nested scope.
    DeferInfo DI{*this};

    /// Procedure we’re currently emitting.
    ProcDecl* curr_proc{};

    CodeGen(Module* mod, bool no_verify)
        : mod(mod),
          ctx(mod->context),
          mctx(&ctx->mlir),
          builder(mctx),
          no_verify(no_verify) {}

    CodeGen(const CodeGen&) = delete;
    CodeGen(CodeGen&&) = delete;
    CodeGen& operator=(const CodeGen&) = delete;
    CodeGen& operator=(CodeGen&&) = delete;

public:
    static void Generate(Module* mod, bool no_verify) {
        Assert(not mod->context->has_error(), "Refusing to codegen broken module");
        CodeGen c{mod, no_verify};
        c.GenerateModule();
    }

private:
    /// Create an alloca for a local variable.
    [[nodiscard]] auto AllocateLocalVar(LocalDecl* decl) -> mlir::Value;

    /// Attach a block to the end of a region.
    auto Attach(mlir::Region* region, mlir::Block* block) -> mlir::Block*;

    /// Check if a block is closed.
    bool Closed();
    bool Closed(mlir::Block* block);

    template <typename T, typename... Args>
    auto Create(mlir::Location loc, Args&&... args) -> decltype(builder.create<T>(loc, std::forward<Args>(args)...));

    auto EmitReference(mlir::Location loc, Expr* decl) -> mlir::Value;

    auto EndLifetime(LocalDecl* decl);

    template <typename Op>
    void GenerateBinOp(BinaryExpr* b);

    template <typename Op>
    void GenerateCmpOp(BinaryExpr*, mlir::arith::CmpIPredicate pred);

    void Generate(Expr* expr);
    void GenerateModule();
    void GenerateProcedure(ProcDecl* proc);

    /// Retrieve the static chain pointer for a procedure.
    auto GetStaticChainPointer(ProcDecl* proc) -> mlir::Value;

    /// Perform preprocessing on locals to support nested procedures.
    void InitStaticChain(ProcDecl* proc, mlir::func::FuncOp f);

    auto Ty(Expr* type, bool for_closure = false) -> mlir::Type;
};
} // namespace src

#endif // SOURCE_CG_CODEGEN_HH
