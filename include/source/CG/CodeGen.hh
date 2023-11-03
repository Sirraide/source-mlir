#ifndef SOURCE_CG_CODEGEN_HH
#define SOURCE_CG_CODEGEN_HH

#include <mlir/IR/Builders.h>
#include <source/Core.hh>
#include <source/HLIR/HLIRDialect.hh>

namespace src {
class WhileExpr;
class BinaryExpr;
class LocalDecl;
class CodeGen {
    Module* const mod;
    Context* const ctx;
    mlir::MLIRContext* const mctx;
    mlir::OpBuilder builder;
    bool no_verify;
    usz anon_structs = 0;

    /// Everything related to defer goes here.
    class DeferInfo {
        /// Loop marker.
        struct Loop {
            WhileExpr* loop;
        };

        /// Deferred expression + number of variables
        /// that were in scope when the expression was
        /// added.
        struct DeferredMaterial {
            Expr* expr;
            usz vars_count;
        };

        /// Defer stack associated with a scope.
        using Entry = std::variant<DeferredMaterial, mlir::func::FuncOp>;
        struct Stack {
            SmallVector<Entry, 10> entries;

            void add(Entry e) { entries.push_back(e); }
            void compact(DeferInfo& DI);
            void emit(DeferInfo& DI);
        };

        /// Codegen context.
        CodeGen& CG;

        /// Stack of defer stacks.
        SmallVector<std::variant<Stack, Loop>, 10> stacks;

        /// Local variables that are currently in scope.
        SmallVector<mlir::Value> vars;

        /// Counter for emitting procedures.
        usz defer_procs{};

        /// Sanity check to check for iterator invalidation.
        bool may_compact = false;

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

        /// Add a local variable.
        void AddLocal(mlir::Value val);

        /// Emit all defer stacks up to a labelled expression.
        void EmitDeferStacksUpTo(Expr* stop_at);

        /// Emit all defer stacks for a return expression.
        void Return(bool last_instruction_in_function);

    private:
        /// Call a function that executes deferred expressions.
        void CallCleanupFunc(mlir::func::FuncOp func);

        /// Get the current defer stack.
        auto CurrentStack() -> Stack&;

        /// Iterate over the defer stacks in reverse.
        ///
        /// \param stop_at If this is not nullptr, iteration will
        ///     stop once the given expression is reached in
        ///     the stack.
        /// \param may_compact Whether to allow Stack::compact() calls
        ///     in the loop. This requires an extra stack that is created
        ///     by Iterate() if this flag is true.
        /// \return An iterator range that can be used to iterate over
        ///     the defer stacks.
        auto Iterate(Expr* stop_at, bool may_compact);
    };

    DeferInfo DI{*this};
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
