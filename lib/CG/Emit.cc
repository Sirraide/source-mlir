#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <source/CG/CodeGen.hh>
#include <source/Core.hh>

void src::EmitModule(
    Module* mod,
    int opt_level,
    ObjectFormat fmt,
    const StringMap<bool>& target_features,
    const fs::path& location
) {
    /// FIXME: Don’t think I need to explain what needs fixing here...
    static std::once_flag init;
    std::call_once(init, [] {
        const char* args[]{
            "srcc",
            "-x86-asm-syntax=intel",
            nullptr,
        };
        llvm::cl::ParseCommandLineOptions(2, args, "", &llvm::errs(), nullptr);
    });

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

    /// Get feature flags.
    std::string features;
    if (opt_level == 4) {
        StringMap<bool> feature_map;
        llvm::sys::getHostCPUFeatures(feature_map);
        for (auto& [feature, enabled] : feature_map)
            if (enabled)
                features += fmt::format("+{},", feature.str());
    }

    /// User-specified features are applied last.
    for (auto& [feature, enabled] : target_features)
        features += fmt::format("{}{},", enabled ? '+' : '-', feature.str());
    if (not features.empty()) features.pop_back();

    /// Get CPU.
    std::string cpu;
    if (opt_level == 4) cpu = llvm::sys::getHostCPUName();
    if (cpu.empty()) cpu = "generic";

    /// Target options.
    llvm::TargetOptions opts;

    /// Get opt level.
    llvm::CodeGenOptLevel opt;
    switch (opt_level) {
        case 0: opt = llvm::CodeGenOptLevel::None; break;
        case 1: opt = llvm::CodeGenOptLevel::Less; break;
        case 2: opt = llvm::CodeGenOptLevel::Default; break;
        default: opt = llvm::CodeGenOptLevel::Aggressive; break;
    }

    /// Create machine.
    auto reloc = llvm::Reloc::PIC_;
    auto machine = target->createTargetMachine(
        triple,
        cpu,          /// Target CPU
        features,     /// Features.
        opts,         /// Options.
        reloc,        /// Relocation model.
        std::nullopt, /// Code model.
        opt,          /// Opt level.
        false         /// JIT?
    );

    Assert(machine, "Failed to create target machine");
    GenerateLLVMIR(mod, opt_level, machine);

    /// Set target triple for the module.
    mod->llvm->setTargetTriple(triple);

    /// Set PIC level and DL.
    mod->llvm->setPICLevel(llvm::PICLevel::Level::BigPIC);
    mod->llvm->setPIELevel(llvm::PIELevel::Level::Large);
    mod->llvm->setDataLayout(machine->createDataLayout());

    /// Helper to emit an object/assembly file.
    auto EmitFile = [&](llvm::raw_pwrite_stream& stream) {
        /// No idea how or if the new pass manager can be used for this, so...
        llvm::legacy::PassManager pass;
        if (
            machine->addPassesToEmitFile(
                pass,
                stream,
                nullptr,
                fmt == ObjectFormat::Assembly
                    ? llvm::CodeGenFileType::AssemblyFile
                    : llvm::CodeGenFileType::ObjectFile
            )
        ) Diag::ICE("LLVM backend rejected object code emission passes");
        pass.run(*mod->llvm);
        stream.flush();
    };

    /// Write to stdout.
    if (location == "-") {
        /// '-' as the output file designates stdout. Only valid if we’re
        /// not emitting an executable.
        if (fmt != ObjectFormat::Assembly)
            Diag::Fatal("Can only emit assembly to stdout");
        EmitFile(llvm::outs());
        return;
    }

    /// Write to a file.
    auto temp = File::TempPath(__SRCC_OBJ_FILE_EXT);
    std::error_code ec;
    llvm::raw_fd_ostream stream{
        fmt == ObjectFormat::Executable ? temp.string() : location.string(),
        ec,
        llvm::sys::fs::OF_None,
    };

    if (ec) Diag::Fatal(
        "Could not open file '{}': {}",
        location.string(),
        ec.message()
    );

    EmitFile(stream);

    /// Stop here if we are not supposed to emit an executable.
    if (fmt != ObjectFormat::Executable) return;
    Assert(not mod->is_logical_module, "Cannot emit logical module as executable");

    /// Yeet object file when we’re done.
    defer { fs::remove(temp); };

#ifdef LLVM_ON_UNIX
    /// Find linker.
    auto link = llvm::sys::findProgramByName(__SRCC_CLANG_EXE);
    if (link.getError()) link = llvm::sys::findProgramByName("clang");
    if (auto e = link.getError()) Diag::Fatal("Could not find linker: {}", e.message());

    /// Add the object file as well as all imported modules.
    SmallVector<std::string> args;
    args.push_back(link.get());
    args.push_back(fs::absolute(temp).string());
    args.push_back("-o");
    args.push_back(fs::absolute(location).string());
    for (auto& m : mod->imports) args.push_back(fs::absolute(m.resolved_path).string());
    SmallVector<llvm::StringRef> args_ref;
    for (auto& a : args) args_ref.push_back(a);

    /// Run the linker.
    auto ret = llvm::sys::ExecuteAndWait(link.get(), args_ref);
    if (ret != 0) Diag::Fatal("Linker returned non-zero exit code: {}", ret);

#else
#    error Sorry, unsupported platform
#endif
}
