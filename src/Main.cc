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
    option<"--colour", "Enable coloured output (default: auto)", values<"always", "auto", "never">>,
    option<"--dir", "Set module output directory">,
    multiple<option<"-I", "Add module import directory">>,
    experimental::short_option<"-O", "Optimisation level", values<0, 1, 2, 3, 4>>,
    flag<"-r", "JIT-compile and run the program after compiling">,
    flag<"--ast", "Print the AST of the module after parsing">,
    flag<"--debug-llvm", "Debug LLVM lowering process">,
    flag<"--describe-module", "Load file as a module and print its exports">,
    flag<"--exports", "Show exported declarations">,
    flag<"--hlir", "Print the HLIR of the module">,
    flag<"--llvm", "Print the LLVM IR of the module">,
    flag<"--no-verify", "Disable MLIR verification; CAUTION: this may lead to miscompilations">,
    flag<"--sema", "Run sema only">,
    flag<"--syntax-only", "Skip the semantic analysis step">,
    flag<"--use-generic-assembly-format", "Print HLIR using the generic assembly format">,
    help<>
>; // clang-format on
}
using detail::options;

int main(int argc, char** argv) {
    llvm::EnablePrettyStackTrace();
    auto opts = options::parse(argc, argv, [](auto&& s) -> bool { src::Diag::Fatal("{}", s); });

    /// Check if we want to use colours.
    bool use_colour = isatty(fileno(stdout));
    if (auto c = opts.get<"--colour">()) {
        if (*c == "always") use_colour = true;
        else if (*c == "never") use_colour = false;
    }

    /// Create context.
    src::Context ctx;
    auto& f = ctx.get_or_load_file(*opts.get<"file">());

    /// Add import paths.
    for (auto& path : *opts.get<"-I">()) {
        std::filesystem::path p{path};
        if (p.is_relative()) p = std::filesystem::current_path() / p;
        ctx.import_paths.push_back(std::move(p));
    }

    /// Describe module, if requested.
    if (opts.get<"--describe-module">()) {
        auto mod = src::Module::Deserialise(
            &ctx,
            auto{f.path()}.filename().replace_extension(""),
            {},
            llvm::ArrayRef(
                reinterpret_cast<const src::u8*>(f.data()),
                f.size()
            )
        );

        /// Check for errors.
        for (auto& exps : mod->exports)
            for (auto e : exps.second)
                e->print(false);
        std::exit(1);
    }

    /// Parse the file. Exit on error since, in that case, the
    /// parser returns nullptr.
    auto mod = src::Parser::Parse(ctx, f);
    if (ctx.has_error()) std::exit(1);

    /// Print the AST of the module, if requested.
    if (opts.get<"--syntax-only">()) {
        if (opts.get<"--ast">()) mod->print_ast(use_colour);
        std::exit(0);
    }

    /// Perform semantic analysis.
    src::Sema::Analyse(mod);
    if (ctx.has_error()) std::exit(1);
    if (opts.get<"--ast">() or opts.get<"--sema">()) {
        if (opts.get<"--ast">()) mod->print_ast(use_colour);
        std::exit(0);
    }

    /// Print exports if requested.
    if (opts.get<"--exports">()) {
        if (not mod->is_logical_module) src::Diag::Fatal(
            "Cannot print exports: not a logical module"
        );

        for (auto& exps : mod->exports)
            for (auto e : exps.second)
                e->print(false);

        std::exit(0);
    }

    /// Generate HLIR. If this fails, that’s an ICE, so no
    /// need for error checking here.
    src::CodeGen::Generate(mod, opts.get<"--no-verify">());
    if (ctx.has_error()) std::exit(1);
    if (opts.get<"--hlir">()) {
        mod->print_hlir(opts.get<"--use-generic-assembly-format">());
        std::exit(0);
    }

    /// Lower HLIR to LLVM IR.
    src::LowerToLLVM(mod, opts.get<"--debug-llvm">(), opts.get<"--no-verify">());
    if (ctx.has_error()) std::exit(1);
    if (opts.get<"--llvm">()) {
        mod->print_llvm(int(opts.get_or<"-O">(0)));
        std::exit(0);
    }

    /// Run the code if requested.
    if (opts.get<"-r">()) {
        if (mod->is_logical_module) src::Diag::Fatal("'-r' flag is invalid: cannot execute module");
        return mod->run(int(opts.get_or<"-O">(0)));
    }

    /// Emit the module to disk.
    auto dir = opts.get_or<"--dir">(std::filesystem::current_path());
    mod->emit_object_file(int(opts.get_or<"-O">(0)), fmt::format("{}/{}.o", dir, mod->name));

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
