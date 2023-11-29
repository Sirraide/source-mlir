#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <source/Core.hh>

void src::Module::emit_executable(int opt_level, const fs::path& location) {
    Assert(not is_logical_module, "Cannot emit logical module as executable");
    Todo();
}

void src::Module::emit_object_file(int opt_level, const fs::path& location) {
    GenerateLLVMIR(opt_level);

    /// Get target.
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    std::string error;
    auto triple = llvm::sys::getDefaultTargetTriple();
    auto target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (not error.empty() or not target) Diag::ICE(
        "Failed to lookup target triple '{}': {}",
        triple,
        error
    );

    /// Set target triple for the module.
    llvm->setTargetTriple(triple);

    /// Target options.
    llvm::TargetOptions opts;

    /// Enable PIC.
    auto reloc = llvm::Reloc::PIC_;
    llvm->setPICLevel(llvm::PICLevel::Level::BigPIC);
    llvm->setPIELevel(llvm::PIELevel::Level::Large);

    /// Get feature flags.
    std::string features;
    if (opt_level == 4) {
        StringMap<bool> feature_map;
        llvm::sys::getHostCPUFeatures(feature_map);
        for (auto& [feature, enabled] : feature_map)
            if (enabled)
                features += fmt::format("+{},", feature.str());
        if (not features.empty()) features.pop_back();
    }

    /// Get CPU.
    std::string cpu = "generic";
    if (opt_level) {
        cpu = llvm::sys::getHostCPUName();
        if (cpu.empty()) cpu = "generic";
    }

    /// Get opt level.
    llvm::CodeGenOptLevel opt;
    switch (opt_level) {
        case 0: opt = llvm::CodeGenOptLevel::None; break;
        case 1: opt = llvm::CodeGenOptLevel::Less; break;
        case 2: opt = llvm::CodeGenOptLevel::Default; break;
        default: opt = llvm::CodeGenOptLevel::Aggressive; break;
    }

    /// Emit to object file.
    auto machine = target->createTargetMachine(
        llvm->getTargetTriple(),
        cpu,          /// Target CPU
        features,     /// Features.
        opts,         /// Options.
        reloc,        /// Relocation model.
        std::nullopt, /// Code model.
        opt,          /// Opt level.
        false         /// JIT?
    );

    /// Set data layout for module.
    Assert(machine);
    llvm->setDataLayout(machine->createDataLayout());

    /*/// Run optimisation pipeline if optimisations are enabled.
    if (opt_level) optimise(src_mod, opt_level);
    */

    std::error_code ec;
    llvm::raw_fd_ostream dest{location.native(), ec, llvm::sys::fs::OF_None};
    if (ec) Diag::Fatal("Could not open file '{}': {}", location.native(), ec.message());

    /// No idea how or if the new pass manager can be used for this, so...
    llvm::legacy::PassManager pass;
    if (machine->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::ObjectFile))
        Diag::ICE("LLVM backend rejected object code emission passes");
    pass.run(*llvm);
    dest.flush();
}
