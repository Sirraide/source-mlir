#include <clang/AST/RecordLayout.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Config/config.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Compilation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/Utils.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/TargetParser/Host.h>
#include <memory>
#include <source/Frontend/Sema.hh>

#define bind SRC_BIND ImportContext::

template <>
struct fmt::formatter<clang::QualType> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(clang::QualType t, FormatContext& ctx) {
        std::string s;
        llvm::raw_string_ostream o{s};
        o << t;
        return fmt::format_to(ctx.out(), "{}", s);
    }
};

namespace src {
namespace {
struct ImportContext {
    Context* ctx;
    Module* out;
    Sema S{out};
    bool debug;
    clang::ASTContext* AST;
    clang::DiagnosticsEngine* D;
    llvm::DenseSet<const clang::Decl*> translated_decls;
    llvm::DenseMap<const clang::Type*, std::optional<Type>> translated_types;

    Type Char = new (out) IntType(Size::Bits(ctx->clang.getTarget().getCharWidth()), {});
    Type Short = new (out) IntType(Size::Bits(ctx->clang.getTarget().getShortWidth()), {});
    Type Int = new (out) IntType(Size::Bits(ctx->clang.getTarget().getIntWidth()), {});
    Type Long = new (out) IntType(Size::Bits(ctx->clang.getTarget().getLongWidth()), {});
    Type LongLong = new (out) IntType(Size::Bits(ctx->clang.getTarget().getLongLongWidth()), {});

    ImportContext(Module* out, bool debug)
        : ctx(out->context), out(out), debug(debug) {}

    void HandleTranslationUnit(clang::ASTContext& C) {
        AST = &C;
        D = &AST->getDiagnostics();
        for (auto d : C.getTranslationUnitDecl()->decls()) TranslateDecl(d);
    }

    void TranslateDecl(clang::Decl* d) {
        d = d->getCanonicalDecl();
        if (translated_decls.contains(d)) return;
        translated_decls.insert(d);
        llvm::TypeSwitch<clang::Decl*>(d)
            .Case<clang::LinkageSpecDecl>(bind TranslateLinkageDecl)
            .Case<clang::FunctionDecl>(bind TranslateFunctionDecl);
    }

    void TranslateLinkageDecl(clang::LinkageSpecDecl* l) {
        for (auto d : l->decls()) TranslateDecl(d);
    }

    void TranslateFunctionDecl(clang::FunctionDecl* f) {
        if (not f->isExternC()) return;
        if (f->isDeleted()) return;
        if (f->getLanguageLinkage() != clang::LanguageLinkage::CLanguageLinkage) return;

        auto ret = f->isNoReturn() ? Type::NoReturn : TranslateType(f->getReturnType());
        if (not ret.has_value()) return;
        SmallVector<Type> param_types;
        for (auto p : f->parameters()) {
            auto t = TranslateType(p->getType());
            if (not t.has_value()) return;
            param_types.push_back(*t);
        }

        auto type = new (out) ProcType(
            std::move(param_types),
            *ret,
            f->isVariadic(),
            {}
        );

        auto proc = new (out) ProcDecl(
            out,
            nullptr,
            f->getDeclName().getAsString(),
            type,
            {},
            Linkage::Imported,
            Mangling::None,
            {}
        );

        out->exports[proc->name].push_back(proc);
    }

    auto TranslateType(clang::QualType type) -> std::optional<Type> {
        if (
            auto it = translated_types.find(type.getTypePtr());
            it != translated_types.end()
        ) return it->second;
        auto type_ = TranslateTypeImpl(type);
        translated_types[type.getTypePtr()] = type_;
        if (type_.has_value()) Assert(S.AnalyseAsType(*type_));
        return type_;
    }

    auto TranslateTypeImpl(clang::QualType type) -> std::optional<Type> {
        if (type.isVolatileQualified()) return Unsupported(type);
        if (type.isNull()) return Unsupported(type);
        auto ptr = type.getTypePtr()->getCanonicalTypeUnqualified();

        /// Primitive types.
        if (ptr->isCharType()) return Char;
        if (ptr == AST->ShortTy or ptr == AST->UnsignedShortTy) return Short;
        if (ptr == AST->IntTy or ptr == AST->UnsignedIntTy) return Int;
        if (ptr == AST->LongTy or ptr == AST->UnsignedLongTy) return Long;
        if (ptr == AST->LongLongTy or ptr == AST->UnsignedLongLongTy) return LongLong;
        if (ptr == AST->VoidTy) return Type::Void;
        if (ptr == AST->BoolTy) return Type::Bool;

        /// Pointers.
        if (auto ty = dyn_cast<clang::PointerType>(ptr)) {
            auto elem = TranslateType(ty->getPointeeType());
            if (not elem.has_value()) return std::nullopt;
            return new (out) ReferenceType(*elem, {});
        }

        /// Opaque types.
        if (ptr->isIncompleteType()) {
            if (auto ty = dyn_cast<clang::RecordType>(ptr)) {
                auto decl = ty->getDecl();
                if (decl->isStruct()) {
                    auto name = decl->getNameAsString();
                    if (name.empty()) return Unsupported(type);
                    return new (out) OpaqueType(out, name, Mangling::None, {});
                }
            }
        }

        /// Arrays.
        if (auto arr = dyn_cast<clang::ConstantArrayType>(ptr)) {
            auto elem = TranslateType(arr->getElementType());
            if (not elem.has_value()) return std::nullopt;
            auto size = new (out) ConstExpr(nullptr, EvalResult{arr->getSize(), Type::Int}, {});
            return new (out) ArrayType(*elem, size, {});
        }

        /// Structs.
        if (auto decl = ptr->getAsCXXRecordDecl()) {
            if (not decl->isCLike()) return Unsupported(type);
            if (decl->isUnion()) return Unsupported(type);
            auto& layout = AST->getASTRecordLayout(decl);

            /// These may be recursive, so create the type upfront.
            auto s = new (out) StructType(
                out,
                std::string{decl->getName()},
                {},
                {},
                new (out) BlockExpr(out, out->global_scope),
                Mangling::None,
                {}
            );

            translated_types[type.getTypePtr()] = s;
            s->stored_alignment = layout.getAlignment().getAsAlign();
            s->stored_size = Size::Bytes(usz(layout.getSize().getQuantity()));
            s->sema.set_done();

            SmallVector<StructType::Field> fields;
            Size size;
            usz padding_count = 0;
            for (auto [i, f] : vws::enumerate(decl->fields())) {
                auto ty = TranslateType(f->getType());
                if (not ty.has_value()) return std::nullopt;
                if (f->isBitField()) return std::nullopt;
                auto offs = Size::Bits(layout.getFieldOffset(u32(i)));

                if (offs != size) {
                    Assert(offs > size);
                    fields.push_back(StructType::Field{
                        .name = fmt::format("#padding{}", padding_count++),
                        .type = ArrayType::GetByteArray(out, isz((offs - size).bytes())),
                        .offset = offs - size,
                        .index = u32(fields.size()),
                        .padding = true
                    });
                }

                fields.push_back(StructType::Field{
                    .name = std::string{f->getName()},
                    .type = *ty,
                    .offset = offs,
                    .index = u32(fields.size()),
                    .padding = false
                });

                size = offs + fields.back().type.size(ctx);
            }

            out->exports[s->name].push_back(s);
            s->all_fields = std::move(fields);
            return s;
        }

        return Unsupported(type);
    }

    auto Unsupported(clang::QualType type) -> std::optional<Type> {
        if (debug) Diag::Warning("Unsupported type '{}'", type);
        return std::nullopt;
    }
};

} // namespace
} // namespace src

auto src::Module::ImportCXXHeaders(
    Context* ctx,
    ArrayRef<StringRef> header_names,
    bool debug_cxx,
    Location loc
) -> Module* {
    clang::driver::Driver dr{
        __SRCC_CLANG_EXE,
        llvm::sys::getDefaultTargetTriple(),
        ctx->clang.getDiagnostics(),
        "Source Compiler",
        ctx->vfs
    };

    std::string code;
    for (auto& name : header_names) code += fmt::format("#include <{}>\n", name);
    auto buffer = llvm::MemoryBuffer::getMemBuffer(code);
    ctx->vfs->addFile("__srcc_cxx_headers.cc", 0, std::move(buffer));

/*

    /// Resolve system include dirs.
    std::unique_ptr<clang::driver::Compilation> C{dr.BuildCompilation(args)};
    auto clang_args =  C->getJobs().begin()->getArguments();
    llvm::opt::ArgStringList include_dirs;
    C->getDefaultToolChain().AddClangSystemIncludeArgs(C->getArgs(), include_dirs);
    C->getDefaultToolChain().AddClangCXXStdlibIncludeArgs(C->getArgs(), include_dirs);
    clang_args.insert(clang_args.end(), include_dirs.begin(), include_dirs.end());
    for (auto& arg : clang_args) {
        fmt::print("{} ", arg);
    }
*/
    /// Create compiler invocation.
    static constexpr const char* args[] = {
        __SRCC_CLANG_EXE,
        //"-###",
        "__srcc_cxx_headers.cc",
        "-std=c++2b",
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused",
        "-fwrapv",
        "-fcolor-diagnostics",
        "-fsyntax-only"
    };

    //"-cc1",
    //"-stdlib=libstdc++",

    clang::CreateInvocationOptions opts;
    opts.VFS = ctx->vfs;
    opts.Diags = &ctx->clang.getDiagnostics();
    std::shared_ptr<clang::CompilerInvocation> I = clang::createInvocation(args, opts);
    auto AST = clang::ASTUnit::LoadFromCompilerInvocation(
        I,
        std::make_shared<clang::PCHContainerOperations>(),
        ctx->clang.getDiagnosticsPtr(),
        &ctx->clang.getFileManager()
    );

/*
    auto ok = clang::CompilerInvocation::CreateFromArgs(
        *I,
        {clang_args.begin() + 1, clang_args.end()}, /// Skip '-cc1'.
        ctx->clang.getDiagnostics(),
        "srcc"
    );
*/
/*
    /// Set lang options and add include dirs.
    if (not ok) Diag::ICE("Failed to create CompilerInvocation");
    std::vector<std::string> includes;
    clang::LangOptions::setLangDefaults(
        I->getLangOpts(),
        clang::Language::CXX,
        ctx->clang.getTarget().getTriple(),
        includes,
        clang::LangStandard::lang_cxx26
    );

    for (auto& path : includes) {
        I->getHeaderSearchOpts().AddPath(
            path,
            clang::frontend::CXXSystem,
            false,
            true
        );
    }

    I->getHeaderSearchOpts().ResourceDir = dr.ResourceDir;*/


    if (not AST) {
        ctx->set_error();
        return nullptr;
    }

    auto mod = new Module(ctx, "<cxx-headers>", true, loc);
    ImportContext IC{mod, debug_cxx};
    IC.HandleTranslationUnit(AST->getASTContext());

    /// Analyse all exports.
    Sema::Analyse(mod);
    return ctx->has_error() ? nullptr : mod;
}