#include <clang/AST/RecordLayout.h>
#include <clang/Basic/Builtins.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Config/config.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
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
    clang::CompilerInstance& clang;
    Module* out;
    Sema S{out};
    bool debug;
    clang::ASTContext* AST;
    clang::DiagnosticsEngine* D;
    llvm::DenseSet<const clang::Decl*> translated_decls;
    llvm::DenseMap<const clang::Type*, std::optional<Type>> translated_types;

    Type Char = new (out) IntType(Size::Bits(clang.getTarget().getCharWidth()), {});
    Type Short = new (out) IntType(Size::Bits(clang.getTarget().getShortWidth()), {});
    Type Int = new (out) IntType(Size::Bits(clang.getTarget().getIntWidth()), {});
    Type Long = new (out) IntType(Size::Bits(clang.getTarget().getLongWidth()), {});
    Type LongLong = new (out) IntType(Size::Bits(clang.getTarget().getLongLongWidth()), {});

    ImportContext(Module* out, clang::CompilerInstance& clang, bool debug)
        : ctx(out->context), clang(clang), out(out), debug(debug) {}
    
    void HandleTranslationUnit(clang::ASTContext& C) {
        AST = &C;
        D = &AST->getDiagnostics();
        for (auto d : C.getTranslationUnitDecl()->decls()) TranslateDecl(d);
    }
    
    bool IsNonLibBuiltin(clang::FunctionDecl* f) {
#define LIBBUILTIN(Name, ...)
#define BUILTIN(Name, ...) case clang::Builtin::BI##Name:
        switch (f->getBuiltinID()) {
            default: return false;
#include <clang/Basic/Builtins.def>
            return true;
#undef LIBBUILTIN
#undef BUILTIN
        }
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
        if (IsNonLibBuiltin(f)) return;
        if (f->getLanguageLinkage() != clang::LanguageLinkage::CLanguageLinkage) return;

        auto ret = f->isNoReturn() ? Type::NoReturn : TranslateType(f->getReturnType());
        if (not ret.has_value()) return;
        std::deque<ParamInfo> param_types;
        for (auto p : f->parameters()) {
            auto t = TranslateType(p->getType());
            if (not t.has_value()) return;
            param_types.emplace_back(*t, Intent::CXXByValue);
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
            out->save(f->getDeclName().getAsString()),
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
                    return new (out) OpaqueType(out, out->save(name), Mangling::None, {});
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
                out->save(decl->getName()),
                {},
                {},
                {},
                {},
                nullptr,
                Mangling::None,
                {}
            );

            translated_types[type.getTypePtr()] = s;
            s->stored_alignment = layout.getAlignment().getAsAlign();
            s->stored_size = Size::Bytes(usz(layout.getSize().getQuantity()));
            s->sema.set_done();

            SmallVector<FieldDecl*> fields;
            Size size;
            usz padding_count = 0;
            for (auto [i, f] : llvm::enumerate(decl->fields())) {
                auto ty = TranslateType(f->getType());
                if (not ty.has_value()) return std::nullopt;
                if (f->isBitField()) return std::nullopt;
                auto offs = Size::Bits(layout.getFieldOffset(u32(i)));

                if (offs != size) {
                    Assert(offs > size);
                    fields.push_back(new (out) FieldDecl {
                        out->save(fmt::format("#padding{}", padding_count++)),
                        ArrayType::GetByteArray(out, isz((offs - size).bytes())),
                        {},
                        offs - size,
                        u32(fields.size()),
                        true
                    });
                }

                fields.push_back(new (out) FieldDecl{
                    out->save(f->getName()),
                    *ty,
                    {},
                    offs,
                    u32(fields.size()),
                    false
                });

                size = offs + fields.back()->type.size(ctx);
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
    StringRef module_name,
    bool debug_cxx,
    Location loc
) -> Module* {
    Assert(not header_names.empty());
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> file_system;
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> in_memory_fs;
    llvm::IntrusiveRefCntPtr<clang::FileManager> mgr;
    clang::CompilerInstance clang;
    ctx->init_clang(clang);
    file_system = std::make_unique<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    in_memory_fs = std::make_unique<llvm::vfs::InMemoryFileSystem>();
    file_system->pushOverlay(in_memory_fs);
    auto mod = Module::Create(ctx, module_name, true, loc);
    mgr = new clang::FileManager {clang.getFileSystemOpts(), file_system};

    std::string code;
    auto filename = fmt::format("<{}>", header_names.front());
    for (auto& name : header_names) code += fmt::format("#include <{}>\n", name);
    auto buffer = llvm::MemoryBuffer::getMemBuffer(code);
    in_memory_fs->addFile(filename, 0, std::move(buffer));

    const char* args[] = {
        __SRCC_CLANG_EXE,
        "-x",
        "c++",
        filename.data(),
        "-std=c++2b",
        "-Wall",
        "-Wextra",
        "-Werror=return-type",
        "-Wno-unused",
        "-fwrapv",
        "-fcolor-diagnostics",
        "-fsyntax-only"
    };

    clang::CreateInvocationOptions opts;
    opts.VFS = file_system;
    opts.Diags = &clang.getDiagnostics();
    auto AST = clang::ASTUnit::LoadFromCompilerInvocation(
        clang::createInvocation(args, opts),
        std::make_shared<clang::PCHContainerOperations>(),
        clang.getDiagnosticsPtr(),
        mgr.get()
    );

    if (not AST) {
        ctx->set_error();
        return nullptr;
    }

    ImportContext IC{mod, clang, debug_cxx};
    IC.HandleTranslationUnit(AST->getASTContext());
    Sema::Analyse(mod);
    return ctx->has_error() ? nullptr : mod;
}
