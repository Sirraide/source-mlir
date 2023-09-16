#include <source/CG/CodeGen.hh>
#include <source/Frontend/AST.hh>
#include <source/HLIR/HLIRDialect.hh>

/// ===========================================================================
///  Helpers
/// ===========================================================================
auto src::Location::mlir(Context* ctx) const -> mlir::Location {
    if (not seekable(ctx)) return mlir::UnknownLoc::get(&ctx->mlir);
    auto lc = seek_line_column(ctx);
    return mlir::FileLineColLoc::get(
        &ctx->mlir,
        ctx->files()[file_id]->path().native(),
        unsigned(lc.line),
        unsigned(lc.col)
    );
}

void src::Module::print_hlir(bool use_generic_assembly_format) const {
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm(use_generic_assembly_format);
    if (not use_generic_assembly_format) flags.assumeVerified();
    mlir->print(llvm::outs(), flags);
}

auto src::CodeGen::Ty(Expr* type) -> mlir::Type {
    Assert(isa<Type>(type), "Type is not a Type");
    switch (type->kind) {
        case Expr::Kind::BuiltinType: {
            switch (cast<BuiltinType>(type)->builtin_kind) {
                using K = BuiltinTypeKind;
                case K::Unknown: Unreachable();
                case K::Void: return mlir::NoneType::get(mctx);
                case K::Int: return mlir::IndexType::get(mctx);
            }
            Unreachable();
        }

        case Expr::Kind::FFIType: {
            switch (cast<FFIType>(type)->ffi_kind) {
                using K = FFITypeKind;
                case K::CChar:
                case K::CInt:
                    Assert(false, "TODO: FFI Types");
            }
        }

        case Expr::Kind::IntType: {
            auto ty = cast<IntType>(type);
            return mlir::IntegerType::get(mctx, unsigned(ty->bits));
        }

        case Expr::Kind::SliceType: {
            auto ty = cast<SliceType>(type);
            return hlir::SliceType::get(Ty(ty->elem));
        }

        case Expr::Kind::ReferenceType: {
            auto ty = cast<ReferenceType>(type);
            return hlir::ReferenceType::get(Ty(ty->elem));
        }

        case Expr::Kind::ProcType: {
            auto ty = cast<ProcType>(type);
            SmallVector<mlir::Type> params;
            for (auto p : ty->param_types) params.push_back(Ty(p));
            return mlir::FunctionType::get(mctx, params, Ty(ty->ret_type));
        }

        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::OptionalType:
            Unreachable();

        case Expr::Kind::BlockExpr:
        case Expr::Kind::InvokeExpr:
        case Expr::Kind::MemberAccessExpr:
        case Expr::Kind::DeclRefExpr:
        case Expr::Kind::IntegerLiteralExpr:
        case Expr::Kind::StringLiteralExpr:
        case Expr::Kind::ParamDecl:
        case Expr::Kind::ProcDecl:
        case Expr::Kind::CastExpr:
            Unreachable();
    }

    Unreachable();
}

/// ===========================================================================
///  Code Generation
/// ===========================================================================
void src::CodeGen::Generate(src::Expr* expr) {
    if (expr->emitted) return;
    expr->emitted = true;
    switch (expr->kind) {
        case Expr::Kind::BuiltinType:
        case Expr::Kind::FFIType:
        case Expr::Kind::IntType:
        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::SliceType:
        case Expr::Kind::OptionalType:
        case Expr::Kind::ProcType:
            Unreachable();

        case Expr::Kind::InvokeExpr: {
            /// Emit callee.
            auto e = cast<InvokeExpr>(expr);
            Generate(e->callee);

            /// Emit args.
            SmallVector<mlir::Value> args;
            for (auto a : e->args) {
                Generate(a);
                args.push_back(a->mlir);
            }

            /// We emit every call as an indirect call.
            auto call_op = builder.create<mlir::func::CallIndirectOp>(
                e->location.mlir(ctx),
                e->callee->mlir.getType().cast<mlir::FunctionType>().getResults(),
                e->callee->mlir,
                args
            );

            /// The operation only has a result if the function’s
            /// return type is not void.
            if (not Type::Equal(e->type, Type::Void)) e->mlir = call_op.getResult(0);
        } break;

        case Expr::Kind::DeclRefExpr: {
            /// If the operand is a function, create a function constant.
            auto e = cast<DeclRefExpr>(expr);
            if (auto f = dyn_cast<ProcDecl>(e->decl)) {
                Generate(e->decl);
                e->mlir = e->decl->mlir;
            } else {
                Unreachable();
            }
        } break;

        case Expr::Kind::MemberAccessExpr: {
            /// Emit the base object.
            auto e = cast<MemberAccessExpr>(expr);
            Generate(e->object);

            /// Member of a slice.
            if (isa<SliceType>(e->object->type)) {
                if (e->member == "data") {
                    e->mlir = builder.create<hlir::SliceDataOp>(
                        e->location.mlir(ctx),
                        e->object->mlir
                    );
                } else if (e->member == "size") {
                    e->mlir = builder.create<hlir::SliceSizeOp>(
                        e->location.mlir(ctx),
                        e->object->mlir
                    );
                } else {
                    Unreachable();
                }
            } else {
                Unreachable();
            }
        } break;

        case Expr::Kind::CastExpr: {
            auto c = cast<CastExpr>(expr);
            Generate(c->operand);
            switch (c->cast_kind) {
                case CastKind::LValueToRValue: {
                    /// Load the value.
                    c->mlir = builder.create<hlir::LoadOp>(
                        c->location.mlir(ctx),
                        c->operand->mlir
                    );
                } break;
            }
        } break;

        case Expr::Kind::BlockExpr: {
            auto e = cast<BlockExpr>(expr);
            for (auto s : e->exprs) Generate(s);
            if (not Type::Equal(e->type, Type::Void))
                e->mlir = e->exprs.back()->mlir;
        } break;

        case Expr::Kind::StringLiteralExpr: {
            auto str = cast<StrLitExpr>(expr);
            str->mlir = builder.create<hlir::GlobalRefOp>(
                str->location.mlir(ctx),
                Ty(str->type),
                mlir::SymbolRefAttr::get(mctx, fmt::format(".str.{}", str->index))
            );
        } break;

        case Expr::Kind::ProcDecl: {
            auto e = cast<ProcDecl>(expr);
            e->mlir = builder.create<mlir::func::ConstantOp>(
                e->location.mlir(ctx),
                Ty(e->type),
                mlir::SymbolRefAttr::get(mctx, e->name)
            );
        } break;

        case Expr::Kind::IntegerLiteralExpr:
        case Expr::Kind::ParamDecl:
            Unreachable();
    }
}

void src::CodeGen::GenerateModule() {
    /// Initialise MLIR module.
    mod->mlir = mlir::ModuleOp::create(
        mod->module_decl_location.mlir(ctx),
        mod->name.empty() ? "__exe__" : mod->name
    );

    /// Codegen string literals.
    builder.setInsertionPointToEnd(mod->mlir.getBody());
    for (auto [i, s] : vws::enumerate(mod->strtab)) {
        auto op = builder.create<hlir::StringOp>(
            builder.getUnknownLoc(),
            s,
            APInt(64, u64(i))
        );
    }

    /// Codegen functions.
    for (auto f : mod->functions) {
        builder.setInsertionPointToEnd(mod->mlir.getBody());
        GenerateProcedure(f);
    }

    /// Verify the IR.
    if (not mlir::succeeded(mod->mlir.verify()))
        Diag::ICE(ctx, mod->module_decl_location, "Module verification failed");
}

void src::CodeGen::GenerateProcedure(ProcDecl* proc) {
    /// Create the function.
    auto ty = Ty(proc->type);

    mlir::NamedAttribute attrs[] = {
        mlir::NamedAttribute{
            builder.getStringAttr("sym_visibility"),
            builder.getStringAttr(proc->exported ? "global" : "internal"),
        },
    };

    auto func = builder.create<mlir::func::FuncOp>(
        proc->location.mlir(ctx),
        proc->name,
        ty.cast<mlir::FunctionType>()
        //,        ArrayRef<mlir::NamedAttribute>{attrs}
    );

    /// Generate the function body, if there is one.
    if (proc->body) {
        builder.setInsertionPointToEnd(func.addEntryBlock());
        Generate(proc->body);

        /// If the last block is empty, return void.
        if (Type::Equal(cast<ProcType>(proc->type)->ret_type, Type::Void)) {
            if (func.back().empty() or not func.back().back().hasTrait<mlir::OpTrait::IsTerminator>()) {
                builder.create<mlir::func::ReturnOp>(
                    proc->location.mlir(ctx),
                    ArrayRef<mlir::Value>{}
                );
            }
        }
    }
}
