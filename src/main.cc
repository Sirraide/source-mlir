#include <hlir/HLIRDialect.hh>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <utils.hh>

using namespace llvm;

int main() {
    mlir::MLIRContext ctx;
    mlir::PassManager pm(&ctx);
    ctx.getOrLoadDialect<mlir::BuiltinDialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<hlir::HLIRDialect>();
    mlir::OpBuilder builder(&ctx);

    /// Create a function that returns void and takes no arguments.
    auto mod = mlir::ModuleOp::create(builder.getUnknownLoc(), "bla");
    builder.setInsertionPointToEnd(mod.getBody());
    auto funcType = builder.getFunctionType(builder.getI32Type(), {});
    auto func = builder.create<mlir::hlir::FuncOp>(builder.getUnknownLoc(), "main", funcType);

    /// Add a block to the function.
    auto& entryBlock = func.front();
    builder.setInsertionPointToStart(&entryBlock);

    /// Create a string constant.
    auto s = builder.create<mlir::hlir::StringOp>(builder.getUnknownLoc(), "Hello, World!");
    auto i = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 0, 32);
    auto c = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{i});
    mod->print(llvm::outs(), mlir::OpPrintingFlags{}.assumeVerified());
}
