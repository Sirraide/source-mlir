#include <clopts.hh>
#include <llvm/Support/PrettyStackTrace.h>
#include <source/CG/CodeGen.hh>
#include <source/CG/HLIRLowering.hh>
#include <source/Core.hh>
#include <source/Frontend/Parser.hh>
#include <source/Frontend/Sema.hh>

/*auto get_puts(
    mlir::PatternRewriter& rewriter,
    mlir::ModuleOp module,
    mlir::LLVM::LLVMDialect* llvmDialect
) {
    auto ctx = module->getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("puts"))
        return mlir::SymbolRefAttr::get(ctx, "puts");

    auto puts_type = mlir::LLVM::LLVMFunctionType::get(
        mlir::IntegerType::get(ctx, 32),
        {mlir::LLVM::LLVMPointerType::get(nullptr)},
        false
    );

    mlir::PatternRewriter::InsertionGuard guard{rewriter};
    rewriter.setInsertionPointToStart(module.getBody());
    auto puts = rewriter.create<mlir::LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(),
        "puts",
        puts_type
    );
    return mlir::SymbolRefAttr::get(ctx, "puts");
}*/

namespace detail {
using namespace command_line_options;
using options = clopts< // clang-format off
    positional<"file", "The file to compile">,
    flag<"--syntax-only", "Skip the semantic analysis step">,
    flag<"--ast", "Print the AST of the module after parsing">,
    flag<"--hlir", "Print the HLIR of the module">,
    flag<"--use-generic-assembly-format", "Print HLIR using the generic assembly format">,
    flag<"--debug-llvm", "Debug LLVM lowering process">,
    flag<"--llvm", "Print the LLVM IR of the module">,
    help<>
>; // clang-format on
}
using detail::options;

int main(int argc, char** argv) {
    llvm::EnablePrettyStackTrace();
    options::parse(argc, argv);

    /// Create context.
    src::Context ctx;
    auto& f = ctx.get_or_load_file(*options::get<"file">());

    /// Parse the file. Exit on error since, in that case, the
    /// parser returns nullptr.
    auto mod = src::Parser::Parse(ctx, f);
    if (ctx.has_error()) std::exit(1);

    /// Print the AST of the module, if requested.
    if (options::get<"--syntax-only">()) {
        if (options::get<"--ast">()) mod->print_ast();
        std::exit(0);
    }

    /// Perform semantic analysis.
    src::Sema::Analyse(mod.get());
    if (ctx.has_error()) std::exit(1);
    if (options::get<"--ast">()) {
        mod->print_ast();
        std::exit(0);
    }

    /// Generate HLIR. If this fails, that’s an ICE, so no
    /// need for error checking here.
    src::CodeGen::Generate(mod.get());
    if (ctx.has_error()) std::exit(1);
    if (options::get<"--hlir">()) {
        mod->print_hlir(options::get<"--use-generic-assembly-format">());
        std::exit(0);
    }

    /// Lower HLIR to LLVM IR.
    src::LowerToLLVM(mod.get(), options::get<"--debug-llvm">());
    if (ctx.has_error()) std::exit(1);
    if (options::get<"--llvm">()) {
        mod->print_llvm();
        std::exit(0);
    }

    /*    /// Notes:
        /// - ‘freeze’ keyword that makes a value const rather than forcing
        ///   it to be const in the declaration?
        ///
        /// - When parsing a declaration, the declaration itself is attached
        ///   to the nearest enclosing ‘declaration context’; the occurrence
        ///   of the declaration is replaced with an already resolved NameRef
        ///   to that declaration.

        /// Create a function that returns void and takes no arguments.
        auto mod = mlir::ModuleOp::create(builder.getUnknownLoc(), "bla");
        builder.setInsertionPointToEnd(mod.getBody());
        auto funcType = builder.getFunctionType({}, builder.getI32Type());
        auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);

        /// Add a block to the function.
        auto& entryBlock = *func.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        /// Create a string constant and print it.
        auto s = builder.create<mlir::hlir::StringOp>(builder.getUnknownLoc(), "Hello, World!");
        builder.create<mlir::hlir::PrintOp>(builder.getUnknownLoc(), s);

        /// Return 0.
        auto i = builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 42, 32);
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{i});

        fmt::print("=== Module before lowering ===\n");
        if (not mlir::succeeded(mod.verify())) return 1;
        mod->print(llvm::outs());

        /// Lower the module.
        pm.addPass(mlir::hlir::CreateLowerToLLVMPass());
        if (mlir::failed(pm.run(mod))) return 1;

        fmt::print("\n=== Module after lowering ===\n");
        mod->print(llvm::outs());

        /// Convert to LLVM IR.
        mlir::registerLLVMDialectTranslation(*mod->getContext());
        llvm::LLVMContext llvm_ctx;
        auto llvm_mod = mlir::translateModuleToLLVMIR(mod, llvm_ctx);

        fmt::print("\n=== Module after conversion to LLVM IR ===\n");
        llvm_mod->dump();*/
}
