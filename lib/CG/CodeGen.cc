#include <llvm/Support/OptimizedStructLayout.h>
#include <mlir/AsmParser/AsmParser.h>
#include <ranges>
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

auto src::CodeGen::AllocateLocalVar(src::LocalDecl* decl) -> mlir::Value {
    /// Captured variables are stored in the static chain area.
    if (decl->captured) return Create<hlir::StructGEPOp>(
        decl->location.mlir(ctx),
        decl->parent->captured_locals_ptr,
        decl->capture_index
    );

    return Create<hlir::LocalOp>(
        decl->location.mlir(ctx),
        Ty(decl->type),
        decl->type.align(ctx) / 8,
        not decl->init,
        decl->deleted_or_moved
    );
}

auto src::CodeGen::Attach(mlir::Region* region, mlir::Block* block) -> mlir::Block* {
    region->getBlocks().insert(region->end(), block);
    return block;
}

bool src::CodeGen::Closed() {
    return Closed(builder.getBlock());
}

/// Create a function and execute a callback to populate its body.
template <typename Callable>
auto src::CodeGen::CreateProcedure(mlir::FunctionType type, StringRef name, Callable callable) {
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToEnd(mod->mlir.getBody());
    auto func = Create<hlir::FuncOp>(
        builder.getUnknownLoc(),
        name,
        mlir::LLVM::Linkage::Private,
        mlir::LLVM::CConv::C,
        type
    );

    return std::invoke(std::forward<Callable>(callable), func);
}

/// Create an external function.
void src::CodeGen::CreateExternalProcedure(mlir::FunctionType type, StringRef name) {
    if (procs.contains(name)) return;
    CreateProcedure(type, name, [&](hlir::FuncOp proc) {
        proc.eraseBody();
        proc.setPrivate();
        procs.insert(proc.getName());
    });
}

bool src::CodeGen::Closed(mlir::Block* block) {
    return not block->empty() and block->back().hasTrait<mlir::OpTrait::IsTerminator>();
}

template <typename T, typename... Args>
auto src::CodeGen::Create(mlir::Location loc, Args&&... args) -> decltype(builder.create<T>(loc, std::forward<Args>(args)...)) {
    if (auto b = builder.getBlock(); not b->empty() and b->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        /// If the block is already closed, create a new one, or
        /// simply drop the instruction if it is a terminator.
        if constexpr (T::template hasTrait<mlir::OpTrait::IsTerminator>()) return {};
        builder.createBlock(b->getParent());
    }

    /// Create the instruction.
    return builder.create<T>(loc, std::forward<Args>(args)...);
}

auto src::CodeGen::EmitReference([[maybe_unused]] mlir::Location loc, src::Expr* decl) -> mlir::Value {
    /// If the operand is a function, create a function constant.
    if (auto p = dyn_cast<ProcDecl>(decl)) return Create<mlir::func::ConstantOp>(
        loc,
        Ty(p->type),
        mlir::SymbolRefAttr::get(mctx, p->name)
    );

    Unreachable();
}

auto src::CodeGen::EndLifetime([[maybe_unused]] LocalDecl* decl) {
    Todo();
}

void src::CodeGen::InitStaticChain(ProcDecl* proc, hlir::FuncOp func) {
    if (proc->captured_locals.empty() and not proc->takes_static_chain) return;

    /// Collect all captured variables; static chain is first.
    SmallVector<llvm::OptimizedStructLayoutField, 10> captured;
    if (proc->takes_static_chain) captured.push_back({
        nullptr,
        8,              /// FIXME: Get size of pointer type from context.
        llvm::Align(8), /// FIXME: Get alignment of pointer type from context.
        0,
    });

    /// Add a field for each captured variable.
    for (auto v : proc->captured_locals) {
        captured.push_back({
            v,
            u64(v->type.size_bytes(ctx)),
            llvm::Align(u64(v->type.align(ctx) / 8)),
            llvm::OptimizedStructLayoutField::FlexibleOffset,
        });
    }

    /// Optimise the layout.
    const auto [size, align] = llvm::performOptimizedStructLayout(captured);

    /// Combine the allocas to a struct type. Even though the offsets of
    /// all fields are known, we may still need to emit padding to align
    /// each field to its required offset since LLVM doesn’t know anything
    /// about the offsets.
    SmallVector<StructType::Field> fields;
    isz total_size = 0;
    for (const auto& [i, var] : vws::enumerate(captured)) {
        /// Zero is the static chain.
        if (proc->takes_static_chain and i == 0) {
            fields.emplace_back(
                "",
                new (mod) ReferenceType(proc->parent->captured_locals_type, {}),
                0,
                false
            );

            total_size += 8; /// FIXME: Get pointer alignment from context.
            continue;
        }

        /// Anything else is a captured local. Note: const_cast is
        /// safe here because we passed in non-const LocalDecl*’s
        /// above.
        auto v = static_cast<LocalDecl*>(const_cast<void*>(var.Id));
        v->capture_index = isz(i);

        /// Insert padding if required.
        if (total_size != isz(var.Offset)) {
            v->capture_index++;
            fields.emplace_back(
                "",
                ArrayType::GetByteArray(mod, isz(var.Offset) - total_size),
                total_size,
                true
            );
        }

        total_size = isz(var.Offset) + v->type.size_bytes(ctx);
        fields.emplace_back(
            "",
            v->type,
            isz(var.Offset),
            false
        );
    }

    /// Create a struct type and finalise it.
    std::string name = fmt::format("struct.anon.{}", anon_structs++);
    auto s = new (mod) StructType(mod, std::move(name), std::move(fields), nullptr, {});

    /// The alignment and size are just set to what LLVM’s algorithm told us
    /// they should be. In particular, we need *not* ensure that the size is
    /// a multiple of the alignment, as we will never have an array of these
    /// anyway.
    s->stored_alignment = isz(align.value());
    s->stored_size = isz(size);
    s->sema.set_done();

    /// Associate it with the procedure and create the vars area.
    proc->captured_locals_type = s;
    proc->captured_locals_ptr = Create<hlir::LocalOp>(
        builder.getUnknownLoc(),
        Ty(s),
        s->stored_alignment,
        false,
        false
    );

    /// Save the parent’s chain pointer if there is one.
    if (proc->takes_static_chain) {
        Assert(func.getNumArguments() == proc->params.size() + 1);

        /// Save the pointer.
        Create<hlir::StoreOp>(
            builder.getUnknownLoc(),
            proc->captured_locals_ptr,
            func.getArgument(u32(proc->params.size())), /// (!)
            8                                           /// FIXME: Get alignment of pointer type from context.
        );
    }
}

auto src::CodeGen::GetStaticChainPointer(ProcDecl* proc) -> mlir::Value {
    /// Current procedure.
    if (curr_proc == proc) {
        Assert(proc->captured_locals_ptr);
        return proc->captured_locals_ptr;
    }

    /// Load the parent’s chain pointer.
    mlir::Value chain = Create<hlir::ChainExtractLocalOp>(
        builder.getUnknownLoc(),
        curr_proc->captured_locals_ptr,
        0
    );

    /// Walk up the stack till we reach the desired procedure.
    for (auto p = curr_proc->parent; p != proc; p = p->parent) {
        chain = Create<hlir::ChainExtractLocalOp>(
            builder.getUnknownLoc(),
            chain,
            0
        );
    }

    return chain;
}

auto src::CodeGen::Ty(Expr* type, bool for_closure) -> mlir::Type {
    Assert(isa<Type>(type), "Type is not a Type");
    switch (type->kind) {
        case Expr::Kind::BuiltinType: {
            switch (cast<BuiltinType>(type)->builtin_kind) {
                using K = BuiltinTypeKind;
                case K::Unknown: Unreachable();
                case K::Void: return mlir::NoneType::get(mctx);
                case K::Int: return mlir::IntegerType::get(mctx, 64); /// FIXME: Get width from context.
                case K::Bool: return mlir::IntegerType::get(mctx, 1);
                case K::NoReturn: return mlir::NoneType::get(mctx);
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
            auto ty = cast<SingleElementTypeBase>(type);
            return hlir::ReferenceType::get(Ty(ty->elem));
        }

        case Expr::Kind::ScopedPointerType: {
            auto ty = cast<SingleElementTypeBase>(type);
            return hlir::ScopedPointerType::get(Ty(ty->elem));
        }

        case Expr::Kind::ArrayType: {
            auto ty = cast<ArrayType>(type);
            return hlir::ArrayType::get(Ty(ty->elem), usz(ty->dimension()));
        }

        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType: {
            auto ty = cast<SingleElementTypeBase>(type);
            return Ty(ty->elem);
        }

        case Expr::Kind::ClosureType:
            return hlir::ClosureType::get(Ty(cast<ClosureType>(type)->proc_type, true));

        case Expr::Kind::ProcType: {
            auto ty = cast<ProcType>(type);
            SmallVector<mlir::Type> params;
            for (auto p : ty->param_types) params.push_back(Ty(p));

            /// Add an extra parameter for the static chain pointer, unless
            /// this is for a closure type, in which case the environment
            /// will already be added anyway.
            if (not for_closure and ty->static_chain_parent) {
                Assert(ty->static_chain_parent->captured_locals_type);
                params.push_back(hlir::ReferenceType::get(Ty(ty->static_chain_parent->captured_locals_type)));
            }

            /// To ‘return void’, we have to set no return type at all.
            if (not ty->ret_type->as_type.yields_value) return mlir::FunctionType::get(mctx, params, {});
            else return mlir::FunctionType::get(mctx, params, Ty(ty->ret_type));
        }

        case Expr::Kind::OptionalType:
            Todo();

        case Expr::Kind::StructType: {
            auto s = cast<StructType>(type);
            if (s->mlir) return s->mlir;

            /// Collect element types.
            auto range = s->field_types() | vws::transform([&](auto& t) { return Ty(t); });
            SmallVector<mlir::Type> elements{range.begin(), range.end()};

            /// Named struct.
            if (not s->name.empty()) {
                s->mlir = mlir::LLVM::LLVMStructType::getNewIdentified(
                    mctx,
                    s->name,
                    elements,
                    false
                );
            }

            /// Literal struct.
            else {
                s->mlir = mlir::LLVM::LLVMStructType::getLiteral(
                    mctx,
                    elements,
                    false
                );
            }

            return s->mlir;
        }

        case Expr::Kind::ExportExpr:
        case Expr::Kind::AssertExpr:
        case Expr::Kind::ConstExpr:
        case Expr::Kind::ReturnExpr:
        case Expr::Kind::LoopControlExpr:
        case Expr::Kind::GotoExpr:
        case Expr::Kind::LabelExpr:
        case Expr::Kind::AnchorExpr:
        case Expr::Kind::EmptyExpr:
        case Expr::Kind::DeferExpr:
        case Expr::Kind::BlockExpr:
        case Expr::Kind::InvokeExpr:
        case Expr::Kind::InvokeBuiltinExpr:
        case Expr::Kind::MemberAccessExpr:
        case Expr::Kind::ScopeAccessExpr:
        case Expr::Kind::DeclRefExpr:
        case Expr::Kind::ModuleRefExpr:
        case Expr::Kind::LocalRefExpr:
        case Expr::Kind::BoolLiteralExpr:
        case Expr::Kind::IntegerLiteralExpr:
        case Expr::Kind::StringLiteralExpr:
        case Expr::Kind::ProcDecl:
        case Expr::Kind::CastExpr:
        case Expr::Kind::IfExpr:
        case Expr::Kind::WhileExpr:
        case Expr::Kind::UnaryPrefixExpr:
        case Expr::Kind::BinaryExpr:
        case Expr::Kind::LocalDecl:
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
        case Expr::Kind::ClosureType:
        case Expr::Kind::ArrayType:
        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            Unreachable();

        /// These are no-ops.
        case Expr::Kind::StructType:
        case Expr::Kind::ModuleRefExpr:
        case Expr::Kind::EmptyExpr:
            break;

        case Expr::Kind::InvokeExpr: {
            auto e = cast<InvokeExpr>(expr);
            Generate(e->callee);

            /// Emit args.
            SmallVector<mlir::Value> args;
            for (auto a : e->args) {
                Generate(a);
                args.push_back(a->mlir);
            }

            /// If the callee is a procedure decl, call it directly.
            if (auto proc = dyn_cast<mlir::func::ConstantOp>(e->callee->mlir.getDefiningOp())) {
                /// If the callee takes a static chain pointer, retrieve
                /// it and add it to the argument list.
                if (auto chain = cast<ProcType>(e->callee->type)->static_chain_parent)
                    args.push_back(GetStaticChainPointer(chain));

                /// Create the call.
                auto call_op = Create<hlir::CallOp>(
                    e->location.mlir(ctx),
                    cast<mlir::FunctionType>(proc.getType()).getResults(),
                    proc.getValue(),
                    false,
                    mlir::LLVM::CConv::C,
                    args
                );

                /// The operation only has a result if the function’s
                /// return type is not void.
                if (e->type.yields_value) e->mlir = call_op.getYield();
            }

            /// If the callee is a closure, then use a special operation for that.
            else if (auto c = dyn_cast<ClosureType>(e->callee->type)) {
                auto call_op = Create<hlir::InvokeClosureOp>(
                    e->location.mlir(ctx),
                    e->type.yields_value ? mlir::TypeRange{Ty(e->type)} : mlir::TypeRange{},
                    e->callee->mlir,
                    args
                );

                /// The operation only has a result if the function’s
                /// return type is not void.
                if (e->type.yields_value) e->mlir = call_op.getResult();
            }

            else {
                Unreachable("Indirect calls must be closure calls.");
            }
        } break;

        case Expr::Kind::InvokeBuiltinExpr: {
            auto i = cast<InvokeBuiltinExpr>(expr);
            switch (i->builtin) {
                case Builtin::New: {
                    i->mlir = Create<hlir::NewOp>(
                        i->location.mlir(ctx),
                        hlir::ReferenceType::get(Ty(i->args[0]))
                    );
                    return;
                }

                /// Delete calls the destructor of an object.
                case Builtin::Delete: {
                    Generate(i->args[0]);
                    auto var = cast<LocalRefExpr>(i->args[0])->decl;
                    EndLifetime(var);
                    return;
                }
            }

            Unreachable();
        }

        case Expr::Kind::DeclRefExpr: {
            auto e = cast<DeclRefExpr>(expr);
            e->mlir = EmitReference(e->location.mlir(ctx), e->decl);
        } break;

        case Expr::Kind::LocalRefExpr: {
            auto var = cast<LocalRefExpr>(expr);

            /// Easy case: the variable we’re accessing is in the same
            /// scope as the reference.
            if (var->parent == var->decl->parent) {
                Assert(var->decl->mlir);
                var->mlir = var->decl->mlir;
            }

            /// This is the complicated one. We need to retrieve the
            /// address of this variable (note that this is an lvalue!)
            /// via the static chain.
            else {
                /// Get the frame pointer of the procedure containing
                /// the variable declaration.
                auto& locals = var->decl->parent->captured_locals;
                auto var_index = std::distance(locals.begin(), rgs::find(locals, var->decl));
                var->mlir = Create<hlir::StructGEPOp>(
                    var->location.mlir(ctx),
                    GetStaticChainPointer(var->decl->parent),
                    var_index + var->decl->parent->nested
                );
            }
        } break;

        case Expr::Kind::MemberAccessExpr: {
            /// Emit the base object.
            auto e = cast<MemberAccessExpr>(expr);
            Generate(e->object);

            /// The object may be an lvalue; if so, yield the address
            /// rather than loading the entire object.
            if (e->object->is_lvalue) {
                Assert(e->is_lvalue, "Accessing a member of an lvalue should yield an lvalue");
                Assert(e->field, "Struct field not set for member access");
                e->mlir = Create<hlir::StructGEPOp>(
                    e->location.mlir(ctx),
                    e->object->mlir,
                    e->field->index
                );
            }

            /// Object is an rvalue.
            else {
                /// Member of a slice.
                if (isa<SliceType>(e->object->type)) {
                    if (e->member == "data") {
                        e->mlir = Create<hlir::SliceDataOp>(
                            e->location.mlir(ctx),
                            e->object->mlir
                        );
                    } else if (e->member == "size") {
                        e->mlir = Create<hlir::SliceSizeOp>(
                            e->location.mlir(ctx),
                            Ty(Type::Int),
                            e->object->mlir
                        );
                    } else {
                        Unreachable();
                    }
                } else {
                    Unreachable();
                }
            }
        } break;

        case Expr::Kind::ScopeAccessExpr: {
            auto sa = cast<ScopeAccessExpr>(expr);
            Generate(sa->object);
            Generate(sa->resolved);
            sa->mlir = EmitReference(sa->location.mlir(ctx), sa->resolved);
        } break;

        case Expr::Kind::CastExpr: {
            auto c = cast<CastExpr>(expr);

            /// Emit the operand.
            Generate(c->operand);

            /// Note that some of the casts perform the same operation,
            /// but they are logically distinct in that they yield different
            /// types and value categories at the AST level.
            switch (c->cast_kind) {
                /// Load a value or a reference.
                case CastKind::LValueRefToLValue:
                case CastKind::LValueToRValue: {
                    Assert(c->operand->is_lvalue);
                    c->mlir = Create<hlir::LoadOp>(
                        c->location.mlir(ctx),
                        c->operand->mlir
                    );
                } break;

                /// Reference <-> LValue; Nop
                ///
                /// These are logical operations only and no-ops
                /// at the IR level.
                case CastKind::ReferenceToLValue:
                case CastKind::LValueToReference:
                    c->mlir = c->operand->mlir;
                    break;

                /// Proper casts are all handled the same.
                case CastKind::Implicit:
                case CastKind::Soft:
                case CastKind::Hard: {
                    /// No-op.
                    if (Type::Equal(c->operand->type, c->type)) {
                        c->mlir = c->operand->mlir;
                    }

                    /// Procedure to closure casts.
                    else if (isa<ProcType>(c->operand->type) and isa<ClosureType>(c->type)) {
                        auto proc = c->operand->mlir.getDefiningOp<mlir::func::ConstantOp>();
                        auto proc_type = cast<ProcType>(c->operand->type);

                        /// If the procedure is a nested function that takes a static
                        /// chain, retrieve the appropriate chain pointer.
                        if (proc_type->static_chain_parent) {
                            auto chain = GetStaticChainPointer(proc_type->static_chain_parent);
                            c->mlir = Create<hlir::MakeClosureOp>(
                                c->location.mlir(ctx),
                                proc.getValue(),
                                Ty(c->type),
                                chain
                            );
                        }

                        /// Otherwise, leave the data pointer empty; the backend will
                        /// set it to null during lowering.
                        else {
                            c->mlir = Create<hlir::MakeClosureOp>(
                                c->location.mlir(ctx),
                                proc.getValue(),
                                Ty(c->type)
                            );
                        }
                    }

                    /// Integer-to-integer casts.
                    else if (c->operand->type.is_int(true) and c->type.is_int(true)) {
                        auto from_size = c->operand->type.size(ctx);
                        auto to_size = c->type.size(ctx);

                        /// Truncation.
                        if (from_size > to_size) {
                            c->mlir = Create<mlir::arith::TruncIOp>(
                                c->location.mlir(ctx),
                                mlir::IntegerType::get(mctx, unsigned(to_size)),
                                c->operand->mlir
                            );
                        }

                        /// Extension.
                        else if (from_size < to_size) {
                            /// Since all of our integers are signed, we always use sign
                            /// extension, except that, if we’re extending an i1, we use
                            /// zero-extension, as in case of an i1 with the value 1 (true)
                            /// sign-extension would yield -1 instead of 1.
                            if (from_size == 1) {
                                c->mlir = Create<mlir::arith::ExtUIOp>(
                                    c->location.mlir(ctx),
                                    mlir::IntegerType::get(mctx, unsigned(to_size)),
                                    c->operand->mlir
                                );
                            } else {
                                c->mlir = Create<mlir::arith::ExtSIOp>(
                                    c->location.mlir(ctx),
                                    mlir::IntegerType::get(mctx, unsigned(to_size)),
                                    c->operand->mlir
                                );
                            }
                        }

                        /// No-op.
                        else {
                            c->mlir = c->operand->mlir;
                        }
                    }

                    else {
                        Unreachable();
                    }
                }
            }
        } break;

        case Expr::Kind::ConstExpr:
            Todo();

        case Expr::Kind::DeferExpr: {
            auto d = Create<hlir::DeferOp>(expr->location.mlir(ctx));
            expr->mlir = d;
            mlir::OpBuilder::InsertionGuard guard{builder};
            builder.setInsertionPointToEnd(&d.getBody().front());
            Generate(cast<DeferExpr>(expr)->expr);
            if (not Closed(builder.getBlock())) Create<hlir::YieldOp>(
                builder.getUnknownLoc(),
                mlir::Value{}
            );
        } break;

        case Expr::Kind::LoopControlExpr: {
            auto l = cast<LoopControlExpr>(expr);
            const auto loc = expr->location.mlir(ctx);

            /// Emit the branch.
            Create<hlir::DirectBrOp>(
                loc,
                l->is_continue ? l->target->cond_block : l->target->join_block,
                expr->location.encode()
            );
        } break;

        case Expr::Kind::GotoExpr: {
            auto g = cast<GotoExpr>(expr);

            /// Anchors are only used for backward branches; thus, when we
            /// get here, all protected expressions for that anchor have
            /// already been emitted, so we can just add them here.
            SmallVector<mlir::Value> protected_exprs;
            if (g->anchor) {
                for (auto e : vws::reverse(g->anchor->protected_exprs)) {
                    Assert(e->mlir);
                    protected_exprs.push_back(e->mlir);
                }
            }

            Create<hlir::DirectBrOp>(
                g->location.mlir(ctx),
                g->target->block,
                expr->location.encode(),
                protected_exprs
            );
        } break;

        case Expr::Kind::LabelExpr: {
            auto l = cast<LabelExpr>(expr);

            /// Insert the block that we’ve already created for this label.
            if (l->used) {
                Create<mlir::cf::BranchOp>(
                    l->location.mlir(ctx),
                    l->block
                );

                builder.getBlock()->getParent()->push_back(l->block);
                builder.setInsertionPointToEnd(l->block);
            }

            /// Emit the labelled expression.
            Generate(l->expr);
        } break;

        /// Emit the wrapped expression.
        case Expr::Kind::AnchorExpr: {
            auto a = cast<AnchorExpr>(expr);
            Generate(a->expr);
            a->mlir = a->expr->mlir;
        } break;

        /// Nothing to do here other than emitting the underlying decl.
        case Expr::Kind::ExportExpr: {
            auto e = cast<ExportExpr>(expr);
            Generate(e->expr);
        } break;

        case Expr::Kind::ReturnExpr: {
            auto r = cast<ReturnExpr>(expr);
            if (r->value) Generate(r->value);

            /// Return the value.
            if (not Closed()) {
                Create<hlir::ReturnOp>(
                    r->location.mlir(ctx),
                    r->value ? r->value->mlir : mlir::Value{}
                );
            }
        } break;

        case Expr::Kind::AssertExpr: {
            auto a = cast<AssertExpr>(expr);
            Generate(a->cond);
            Create<mlir::cf::AssertOp>(
                expr->location.mlir(ctx),
                a->cond->mlir,
                std::string_view{a->message_string}
            );
        } break;

        case Expr::Kind::BlockExpr: {
            auto e = cast<BlockExpr>(expr);
            if (e->exprs.empty()) break;

            /// Create a scope for the block.
            const bool yields_value = e->type.yields_value;
            auto b = Create<hlir::ScopeOp>(
                e->location.mlir(ctx),
                not yields_value ? mlir::Type{}
                : e->is_lvalue   ? hlir::ReferenceType::get(Ty(e->type))
                                 : Ty(e->type)
            );

            /// Associate block with scope op.
            e->scope_op = b;

            /// Emit contained expressions.
            mlir::OpBuilder::InsertionGuard guard{builder};
            builder.setInsertionPointToEnd(&b.getBody().front());
            for (auto s : e->exprs) Generate(s);
            if (yields_value) e->mlir = b.getRes();
            if (not Closed(builder.getBlock())) {
                Create<hlir::YieldOp>(
                    e->location.mlir(ctx),
                    yields_value ? e->exprs.back()->mlir : mlir::Value{}
                );
            }
        } break;

        case Expr::Kind::StringLiteralExpr: {
            auto str = cast<StrLitExpr>(expr);

            /// Create a global ref to the string data. This returns
            /// a reference to an array of chars.
            auto str_value = mod->strtab[str->index];
            auto i8 = mlir::IntegerType::get(mctx, 8);
            auto str_arr = Create<hlir::GlobalRefOp>(
                str->location.mlir(ctx),
                hlir::ReferenceType::get(hlir::ArrayType::get(i8, str_value.size())),
                mlir::SymbolRefAttr::get(mctx, fmt::format(".str.data.{}", str->index))
            );

            /// Insert a conversion from i8[size]& to i8&.
            auto str_ptr = Create<hlir::ArrayDecayOp>(
                str->location.mlir(ctx),
                str_arr
            );

            /// Create an integer holding the string size.
            auto str_size = Create<mlir::index::ConstantOp>(
                str->location.mlir(ctx),
                mod->strtab[str->index].size() - 1 /// Exclude null terminator.
            );

            /// Create a slice.
            str->mlir = Create<hlir::LiteralOp>(
                str->location.mlir(ctx),
                hlir::SliceType::get(i8),
                str_ptr,
                str_size
            );
        } break;

        /// Create a bool constant.
        case Expr::Kind::BoolLiteralExpr: {
            auto e = cast<BoolLitExpr>(expr);
            e->mlir = Create<mlir::arith::ConstantIntOp>(
                e->location.mlir(ctx),
                e->value,
                mlir::IntegerType::get(mctx, 1)
            );
        } break;

        /// Create an integer constant.
        case Expr::Kind::IntegerLiteralExpr: {
            auto e = cast<IntLitExpr>(expr);

            /// Explicit-width integer types.
            if (auto int_ty = dyn_cast<IntType>(e->type)) {
                e->mlir = Create<mlir::arith::ConstantIntOp>(
                    e->location.mlir(ctx),
                    e->value,
                    mlir::IntegerType::get(mctx, unsigned(int_ty->bits))
                );
            }

            /// `int` type.
            else if (Type::Equal(e->type, Type::Int)) {
                e->mlir = Create<mlir::arith::ConstantOp>(
                    e->location.mlir(ctx),
                    Ty(Type::Int),
                    builder.getI64IntegerAttr(e->value)
                );
            }
        } break;

        case Expr::Kind::LocalDecl: {
            /// Currently, we only have local variables.
            auto e = cast<LocalDecl>(expr);

            /// If the variable hasn’t already been allocated, do so now.
            if (not e->mlir) e->mlir = AllocateLocalVar(e);

            /// If there is an initialiser, emit it.
            if (e->init) {
                Generate(e->init);
                Create<hlir::StoreOp>(
                    e->init->location.mlir(ctx),
                    e->mlir,
                    e->init->mlir,
                    e->type.align(ctx) / 8
                );
            }
        } break;

        /// If expressions.
        case Expr::Kind::IfExpr: {
            /// Emit the condition.
            auto e = cast<IfExpr>(expr);
            Generate(e->cond);

            /// Create the blocks that we need.
            bool has_yield = e->type.yields_value;
            auto cond_block = builder.getBlock();
            auto region = cond_block->getParent();
            auto then = new mlir::Block;
            auto join = e->type.is_noreturn ? nullptr : new mlir::Block;
            auto else_ = e->else_ ? new mlir::Block : join;
            auto if_loc = e->location.mlir(ctx);

            /// Add an argument to the join block if the expression
            /// yields a value.
            if (has_yield) {
                Assert(join);
                auto ty = Ty(e->type);
                if (e->is_lvalue) ty = hlir::ReferenceType::get(ty);
                join->addArgument(ty, if_loc);
            }

            /// Emit the conditional branch.
            Create<mlir::cf::CondBranchOp>(if_loc, e->cond->mlir, then, else_);

            /// Helper to emit a branch.
            auto EmitBranch = [&](Expr* expr, mlir::Block* block) {
                Attach(region, block);
                builder.setInsertionPointToEnd(block);
                Generate(expr);
                if (join) Create<mlir::cf::BranchOp>(
                    if_loc,
                    join,
                    has_yield ? expr->mlir : mlir::ValueRange{}
                );
            };

            /// Emit the then and else branches.
            EmitBranch(e->then, then);
            if (e->else_) EmitBranch(e->else_, else_);

            /// Finally, resume inserting in the join block.
            if (join) {
                Attach(region, join);
                builder.setInsertionPointToEnd(join);
                if (has_yield) e->mlir = join->getArgument(0);
            }
        } break;

        case Expr::Kind::WhileExpr: {
            auto w = cast<WhileExpr>(expr);

            /// Create a new block for the condition so we can branch
            /// to it and emit the condition there.
            auto region = builder.getBlock()->getParent();
            w->cond_block = Attach(region, new mlir::Block);
            Create<mlir::cf::BranchOp>(w->location.mlir(ctx), w->cond_block);
            builder.setInsertionPointToEnd(w->cond_block);
            Generate(w->cond);

            /// Emit the branch to the body.
            auto body = Attach(region, new mlir::Block);
            w->join_block = new mlir::Block;
            Create<mlir::cf::CondBranchOp>(w->location.mlir(ctx), w->cond->mlir, body, w->join_block);

            /// Emit the body.
            builder.setInsertionPointToEnd(body);
            Generate(w->body);
            Create<mlir::cf::BranchOp>(w->location.mlir(ctx), w->cond_block);

            /// Insert the join block and continue inserting there.
            Attach(region, w->join_block);
            builder.setInsertionPointToEnd(w->join_block);
        } break;

        case Expr::Kind::UnaryPrefixExpr: {
            auto u = cast<UnaryPrefixExpr>(expr);
            Generate(u->operand);
            switch (u->op) {
                default: Unreachable("Invalid unary operator: {}", Spelling(u->op));

                /// Dereference.
                case Tk::Star: {
                    u->mlir = Create<hlir::LoadOp>(
                        u->location.mlir(ctx),
                        u->operand->mlir
                    );
                } break;
            }
        } break;

        case Expr::Kind::BinaryExpr: {
            auto b = cast<BinaryExpr>(expr);
            Generate(b->lhs);
            Generate(b->rhs);
            switch (b->op) {
                using namespace mlir::arith;
                using namespace hlir;
                default: Unreachable("Invalid binary operator: {}", Spelling(b->op));
                case Tk::Plus: GenerateBinOp<AddIOp>(b); break;
                case Tk::Minus: GenerateBinOp<SubIOp>(b); break;
                case Tk::Star: GenerateBinOp<MulIOp>(b); break;
                case Tk::StarStar: GenerateBinOp<mlir::math::IPowIOp>(b); break;
                case Tk::Slash: GenerateBinOp<DivSIOp>(b); break;
                case Tk::Percent: GenerateBinOp<RemSIOp>(b); break;
                case Tk::Xor: GenerateBinOp<XOrIOp>(b); break;
                case Tk::ShiftLeft: GenerateBinOp<ShLIOp>(b); break;
                case Tk::ShiftRight: GenerateBinOp<ShRSIOp>(b); break;
                case Tk::ShiftRightLogical: GenerateBinOp<ShRUIOp>(b); break;
                case Tk::EqEq: GenerateCmpOp<CmpIOp>(b, CmpIPredicate::eq); break;
                case Tk::Neq: GenerateCmpOp<CmpIOp>(b, CmpIPredicate::ne); break;
                case Tk::Lt: GenerateCmpOp<CmpIOp>(b, CmpIPredicate::slt); break;
                case Tk::Gt: GenerateCmpOp<CmpIOp>(b, CmpIPredicate::sgt); break;
                case Tk::Le: GenerateCmpOp<CmpIOp>(b, CmpIPredicate::sle); break;
                case Tk::Ge: GenerateCmpOp<CmpIOp>(b, CmpIPredicate::sge); break;

                /// TODO: Short-circuiting if operating on bool.
                case Tk::And: GenerateBinOp<AndIOp>(b); break;
                case Tk::Or: GenerateBinOp<OrIOp>(b); break;

                /// Assignment and reference binding.
                case Tk::Assign:
                case Tk::RDblArrow: {
                    Create<hlir::StoreOp>(
                        b->location.mlir(ctx),
                        b->lhs->mlir,
                        b->rhs->mlir,
                        b->type.align(ctx) / 8
                    );

                    /// Yields lhs as lvalue.
                    b->mlir = b->lhs->mlir;
                } break;
            }
        } break;

        /// Handled by the code that emits a DeclRefExpr.
        case Expr::Kind::ProcDecl: break;
    }
}

template <typename Op>
void src::CodeGen::GenerateBinOp(src::BinaryExpr* b) {
    b->mlir = Create<Op>(
        b->location.mlir(ctx),
        b->lhs->mlir,
        b->rhs->mlir
    );
}

template <typename Op>
void src::CodeGen::GenerateCmpOp(BinaryExpr* b, mlir::arith::CmpIPredicate pred) {
    b->mlir = Create<Op>(
        b->location.mlir(ctx),
        pred,
        b->lhs->mlir,
        b->rhs->mlir
    );
}

void src::CodeGen::GenerateModule() {
    mctx->loadDialect<hlir::HLIRDialect>();

    /// Initialise MLIR module.
    mod->mlir = mlir::ModuleOp::create(
        mod->module_decl_location.mlir(ctx),
        mod->name.empty() ? "__exe__" : mod->name
    );

/*    /// Set size of pointer.
    auto ptr_size = mlir::DataLayoutEntryAttr::get(
        hlir::ReferenceType::get(mlir::IntegerType::get(mctx, 1)),   /// Elem is irrelevant.
        mlir::IntegerAttr::get(mlir::IntegerType::get(mctx, 64), 64) /// FIXME: use context.
    );

    /// Crate data layout.
    mod->mlir->setAttr(
        mlir::DLTIDialect::kDataLayoutAttrName,
        mlir::DataLayoutSpecAttr::get(mctx, {ptr_size})
    );*/

    /// Codegen string literals.
    builder.setInsertionPointToEnd(mod->mlir.getBody());
    for (auto [i, s] : vws::enumerate(mod->strtab)) {
        Create<hlir::StringOp>(
            builder.getUnknownLoc(),
            StringRef(s.data(), s.size()),
            APInt(64, u64(i))
        );
    }

    /// Codegen imports.
    builder.setInsertionPointToEnd(mod->mlir.getBody());
    for (auto& m : mod->imports) {
        for (auto& exp : m.mod->exports) {
            for (auto e : exp.second) {
                if (auto p = dyn_cast<ProcDecl>(e)) GenerateProcedure(p);
                else Generate(e);
            }
        }
    }

    /// Codegen functions.
    for (auto f : mod->functions) {
        builder.setInsertionPointToEnd(mod->mlir.getBody());
        GenerateProcedure(f);
    }

    /// Delete all function constants.
    mod->mlir.getBodyRegion().walk([](mlir::Operation* op) {
        if (isa<mlir::func::ConstantOp>(op)) op->erase();
    });

    /// Verify the IR.
    if (not no_verify and not mlir::succeeded(mod->mlir.verify()))
        Diag::ICE(ctx, mod->module_decl_location, "Module verification failed");
}

void src::CodeGen::GenerateProcedure(ProcDecl* proc) {
    /// Create the function.
    auto ty = Ty(proc->type);
    tempset curr_proc = proc;

    using L = mlir::LLVM::Linkage;
    auto func = Create<hlir::FuncOp>(
        proc->location.mlir(ctx),
        proc->name,
        proc->exported or proc->imported ? L::External : L::Private,
        mlir::LLVM::CConv::C,
        ty.cast<mlir::FunctionType>()
    );

    /// Associate the function with the procedure.
    proc->mlir_func = func;

    /// Generate the function body, if there is one.
    if (proc->body) {
        /// Entry block must be created first so we can access parameter values.
        builder.setInsertionPointToEnd(&func.front());

        /// Create, but do not insert, blocks for all labels that are actually branched to.
        for (auto& [_, l] : proc->labels)
            if (l->used)
                l->block = new mlir::Block;

        /// Perform the transformations required to make local variables
        /// declared in this procedure accessible to its nested procedures.
        InitStaticChain(proc, func);

        /// Create local variables for parameters.
        for (auto [i, p] : vws::enumerate(proc->params)) {
            p->emitted = true;
            p->mlir = AllocateLocalVar(p);

            /// Store the initial parameter value in the variable.
            Create<hlir::StoreOp>(
                p->location.mlir(ctx),
                p->mlir,
                func.getArgument(u32(i)),
                p->type.align(ctx) / 8
            );
        }

        /// Emit the body.
        Generate(proc->body);

        /// Insert a return expression at the end if there isn’t already one.
        ///
        /// Note: No need to worry about deferred expressions here or anything since
        /// that has already been taken care of by the code that emits the body.
        if (func.back().empty() or not func.back().back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            /// Function returns void.
            if (Type::Equal(proc->ret_type, Type::Void)) {
                Create<hlir::ReturnOp>(proc->location.mlir(ctx), mlir::Value{});
            }

            /// Function does not return, or all paths return a value, but there
            /// is no return expression at the very end.
            else if (Type::Equal(proc->ret_type, Type::NoReturn) or not proc->body->implicit) {
                Create<mlir::LLVM::UnreachableOp>(proc->location.mlir(ctx));
            }

            /// Function is a `= <expr>` function that returns its body.
            else {
                Assert(proc->body->mlir, "Inferred procedure body must yield a value");
                Create<hlir::ReturnOp>(
                    proc->location.mlir(ctx),
                    proc->body->mlir
                );
            }
        }
    }

    /// Otherwise, drop the entry block.
    else {
        func.eraseBody();
        func.setPrivate();
    }
}
