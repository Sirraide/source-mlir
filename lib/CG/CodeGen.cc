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
            auto ty = cast<ReferenceType>(type);
            return hlir::ReferenceType::get(Ty(ty->elem));
        }

        case Expr::Kind::ProcType: {
            auto ty = cast<ProcType>(type);
            SmallVector<mlir::Type> params;
            for (auto p : ty->param_types) params.push_back(Ty(p));

            /// To ‘return void’, we have to set no return type at all.
            if (not ty->ret_type->as_type.yields_value) return mlir::FunctionType::get(mctx, params, {});
            else return mlir::FunctionType::get(mctx, params, Ty(ty->ret_type));
        }

        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::OptionalType:
            Todo();

        case Expr::Kind::AssertExpr:
        case Expr::Kind::ReturnExpr:
        case Expr::Kind::BlockExpr:
        case Expr::Kind::InvokeExpr:
        case Expr::Kind::MemberAccessExpr:
        case Expr::Kind::DeclRefExpr:
        case Expr::Kind::BoolLiteralExpr:
        case Expr::Kind::IntegerLiteralExpr:
        case Expr::Kind::StringLiteralExpr:
        case Expr::Kind::ParamDecl:
        case Expr::Kind::ProcDecl:
        case Expr::Kind::CastExpr:
        case Expr::Kind::IfExpr:
        case Expr::Kind::UnaryPrefixExpr:
        case Expr::Kind::BinaryExpr:
        case Expr::Kind::VarDecl:
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
            if (e->type.yields_value) e->mlir = call_op.getResult(0);
        } break;

        case Expr::Kind::DeclRefExpr: {
            auto e = cast<DeclRefExpr>(expr);

            /// If the operand is a function, create a function constant.
            if (auto p = dyn_cast<ProcDecl>(e->decl)) {
                e->mlir = builder.create<mlir::func::ConstantOp>(
                    e->location.mlir(ctx),
                    Ty(p->type),
                    mlir::SymbolRefAttr::get(mctx, p->name)
                );
            }

            /// If it is a variable, then this is a variable reference.
            else if (isa<VarDecl, ParamDecl>(e->decl)) {
                /// Variable must have already been emitted. This is
                /// an lvalue, so no loading happens here.
                Assert(e->decl->mlir);
                e->mlir = e->decl->mlir;
            }

            else {
                Unreachable();
            }
        } break;

        case Expr::Kind::MemberAccessExpr: {
            /// Emit the base object.
            auto e = cast<MemberAccessExpr>(expr);
            Generate(e->object);

            /// The object may be an lvalue; if so, yield the address
            /// rather than loading the entire object.
            if (e->object->is_lvalue) {
                Todo("Access member of lvalue");
            }

            /// Object is an rvalue.
            else {
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

        case Expr::Kind::CastExpr: {
            auto c = cast<CastExpr>(expr);
            Generate(c->operand);

            /// Note that some of the casts perform the same operation,
            /// but they are logically distinct in that they yield different
            /// types and value categories at the AST level.
            switch (c->cast_kind) {
                /// Load a value or a reference.
                case CastKind::LValueRefToLValue:
                case CastKind::LValueToRValue: {
                    Assert(c->operand->is_lvalue);
                    c->mlir = builder.create<hlir::LoadOp>(
                        c->location.mlir(ctx),
                        c->operand->mlir
                    );
                } break;

                /// Reference <-> LValue.
                ///
                /// These are logical operations only and no-ops
                /// at the IR level.
                case CastKind::ReferenceToLValue:
                case CastKind::LValueToReference:
                    c->mlir = c->operand->mlir;
                    break;

                case CastKind::Implicit: {
                    /// Integer-to-integer casts.
                    if (c->operand->type.is_int(true) and c->type.is_int(true)) {
                        auto from_size = c->operand->type.size(ctx);
                        auto to_size = c->type.size(ctx);

                        /// Truncation.
                        if (from_size > to_size) {
                            c->mlir = builder.create<mlir::arith::TruncIOp>(
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
                                c->mlir = builder.create<mlir::arith::ExtUIOp>(
                                    c->location.mlir(ctx),
                                    mlir::IntegerType::get(mctx, unsigned(to_size)),
                                    c->operand->mlir
                                );
                            } else {
                                c->mlir = builder.create<mlir::arith::ExtSIOp>(
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

        case Expr::Kind::ReturnExpr: {
            auto r = cast<ReturnExpr>(expr);
            if (r->value) Generate(r->value);
            builder.create<mlir::func::ReturnOp>(
                r->location.mlir(ctx),
                r->value ? ArrayRef<mlir::Value>{r->value->mlir} : ArrayRef<mlir::Value>{}
            );
        } break;

        case Expr::Kind::AssertExpr: {
            auto a = cast<AssertExpr>(expr);
            Generate(a->cond);
            builder.create<mlir::cf::AssertOp>(
                expr->location.mlir(ctx),
                a->cond->mlir,
                std::string_view{a->message_string}
            );
        } break;

        case Expr::Kind::BlockExpr: {
            auto e = cast<BlockExpr>(expr);
            for (auto s : e->exprs) Generate(s);
            if (not e->exprs.empty()) e->mlir = e->exprs.back()->mlir;
        } break;

        case Expr::Kind::StringLiteralExpr: {
            auto str = cast<StrLitExpr>(expr);

            /// Create a global ref to the string data. This returns
            /// a reference to an array of chars.
            auto str_value = mod->strtab[str->index];
            auto i8 = mlir::IntegerType::get(mctx, 8);
            auto str_arr = builder.create<hlir::GlobalRefOp>(
                str->location.mlir(ctx),
                hlir::ReferenceType::get(hlir::ArrayType::get(i8, str_value.size())),
                mlir::SymbolRefAttr::get(mctx, fmt::format(".str.data.{}", str->index))
            );

            /// Insert a conversion from i8[size]& to i8&.
            auto str_ptr = builder.create<hlir::ArrayDecayOp>(
                str->location.mlir(ctx),
                str_arr
            );

            /// Create an integer holding the string size.
            auto str_size = builder.create<mlir::index::ConstantOp>(
                str->location.mlir(ctx),
                mod->strtab[str->index].size() - 1 /// Exclude null terminator.
            );

            /// Create a slice.
            str->mlir = builder.create<hlir::LiteralOp>(
                str->location.mlir(ctx),
                hlir::SliceType::get(i8),
                str_ptr,
                str_size
            );
        } break;

        /// Create a bool constant.
        case Expr::Kind::BoolLiteralExpr: {
            auto e = cast<BoolLitExpr>(expr);
            e->mlir = builder.create<mlir::arith::ConstantIntOp>(
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
                e->mlir = builder.create<mlir::arith::ConstantIntOp>(
                    e->location.mlir(ctx),
                    e->value,
                    mlir::IntegerType::get(mctx, unsigned(int_ty->bits))
                );
            }

            /// `int` type.
            else if (Type::Equal(e->type, Type::Int)) {
                e->mlir = builder.create<mlir::arith::ConstantOp>(
                    e->location.mlir(ctx),
                    Ty(Type::Int),
                    builder.getI64IntegerAttr(e->value)
                );
            }
        } break;

        case Expr::Kind::VarDecl: {
            /// Currently, we only have local variables.
            auto e = cast<VarDecl>(expr);
            e->mlir = builder.create<hlir::LocalVarOp>(
                e->location.mlir(ctx),
                Ty(e->type),
                e->type.align(ctx) / 8
            );

            /// If there is an initialiser, emit it.
            if (e->init) {
                Generate(e->init);
                builder.create<hlir::StoreOp>(
                    e->init->location.mlir(ctx),
                    e->mlir,
                    e->init->mlir,
                    e->type.align(ctx) / 8
                );
            }

            /// Otherwise, zero-initialise it.
            else {
                builder.create<hlir::ZeroinitialiserOp>(
                    e->location.mlir(ctx),
                    e->mlir,
                    e->type.size_bytes(ctx)
                );
            }
        } break;

        /// If expressions.
        case Expr::Kind::IfExpr: {
            /// TODO: Emit if expressions using blocks rather than scf.if.
            ///
            /// IMPLEMENTATION OF DEFER:
            ///
            /// For break/continue/return and `defer`: Each defer block becomes
            /// a separate function that is called every time a scope is exited.
            ///
            /// ALL allocas that have been emitted up to that point are passed
            /// to that function, which means that, while we are emitting the
            /// contents of the defer block into a separate function, we can
            /// simply ‘remap’ the VarDecl nodes to map to the function arguments
            /// rather than to their allocas.
            ///
            /// EXAMPLE. Given the following code.
            ///
            /// ```
            /// defer a;
            /// defer b;
            /// if c {
            ///     defer d;
            ///     if d return;
            ///     defer e;
            ///     f;
            ///     return;
            /// }
            /// defer c;
            /// ```
            ///
            /// TODO: Formalise the following algorithm:
            ///
            /// As `defer` expressions are reached, they are placed on a stack;
            /// each scope has its own defer stack. When the first if expression
            /// above is reached, the top-level stack T contains `T: expr(a), expr(b)`;
            /// we then add a new stack for the if expression I1 that links to the
            /// top level stack: `I1(T)`; when `defer d` is reached, we add it
            /// to I1: `I1(T): expr(d)`. When the return after that is reached,
            /// we have to *emit the current stack*.
            ///
            /// The procedure is as follows: we create a new function, F, and pass
            /// to it all currently emitted allocas. Into F, we emit all stack items
            /// in lifo order until we hit a `deferfunc` item or the stack is exhausted.
            ///
            /// STEP X: If a deferfunc is reached, we simply call it and stop. We switch back
            /// from emitting F to the actual function we’re emitting and insert a call
            /// to F in the original function. We then emit the `return` or `branch` to
            /// whatever we’re branching.
            ///
            /// If, on the other hand, the stack is exhausted, we inspect the parent
            /// stack: if there is none, we simply push `F` as a deferfunc onto the
            /// current stack and go to step X. If there is a parent stack, then, if
            /// that stack’s topmost item is a deferfunc (or if the stack is empty),
            /// call it (in F), and we’re done. Push F as a deferfunc and goto step X.
            /// If the parent stack has other items on it, create a separate function
            /// F' and start over with this algorithm and *emit* the parent stack. This will
            /// leave a single deferfunc on the stack, which we call and then push F
            /// as a deferfunc and goto step X.
            ///
            /// In any case, any
            /// association with the parent stack is *cleared*. Thus, before emitting
            /// the `return`, the stacks look like this:
            ///
            ///   - T: expr(a), expr(b)
            ///   - I1(T): expr(d)
            ///
            /// After emitting the return, the stacks look like this:
            ///
            ///   - T: deferfunc(<executes a, b>);
            ///   - I1: deferfunc(<executes d and deferfunc(a, b)>);
            ///
            /// At the end of a scope, we *emit* the defer stack of that scope.

            /// Emit the condition.
            auto e = cast<IfExpr>(expr);
            Generate(e->cond);

            /// Determine the result type of this operation.
            SmallVector<mlir::Type, 1> result_type;
            if (e->type.yields_value) {
                auto ty = Ty(e->type);
                if (e->is_lvalue) ty = hlir::ReferenceType::get(ty);
                result_type.push_back(ty);
            }

            /// Create the if operation.
            auto if_op = builder.create<mlir::scf::IfOp>(
                e->location.mlir(ctx),
                result_type,
                e->cond->mlir,
                /*withElseRegion=*/e->else_ != nullptr
            );

            /// Helper to emit a branch and yield its value if this
            /// expression does no yield void and if there is not
            /// already a terminator in the branch.
            auto EmitBranch = [&](Expr* body, mlir::Block* block) {
                builder.setInsertionPointToStart(block);
                Generate(body);
                if (e->type.yields_value) {
                    if (
                        block->begin() == block->end() or
                        not std::prev(block->end())->hasTrait<mlir::OpTrait::IsTerminator>()
                    ) {
                        builder.create<mlir::scf::YieldOp>(
                            body->mlir.getLoc(),
                            body->mlir
                        );
                    }
                }
            };

            /// Emit the branches and set the result, if applicable.
            EmitBranch(e->then, &*if_op.getThenRegion().begin());
            if (e->else_) EmitBranch(e->else_, &*if_op.getElseRegion().begin());
            if (e->type.yields_value) e->mlir = if_op.getResult(0);
            builder.setInsertionPointAfter(if_op);
        } break;

        case Expr::Kind::UnaryPrefixExpr: {
            auto u = cast<UnaryPrefixExpr>(expr);
            Generate(u->operand);
            switch (u->op) {
                default: Unreachable("Invalid unary operator: {}", Spelling(u->op));

                /// Dereference.
                case Tk::Star: {
                    u->mlir = builder.create<hlir::LoadOp>(
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
                    builder.create<hlir::StoreOp>(
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

        /// Handled by GenerateProcedure().
        case Expr::Kind::ParamDecl: Unreachable();
    }
}

template <typename Op>
void src::CodeGen::GenerateBinOp(src::BinaryExpr* b) {
    b->mlir = builder.create<Op>(
        b->location.mlir(ctx),
        b->lhs->mlir,
        b->rhs->mlir
    );
}

template <typename Op>
void src::CodeGen::GenerateCmpOp(BinaryExpr* b, mlir::arith::CmpIPredicate pred) {
    b->mlir = builder.create<Op>(
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

    /// Codegen string literals.
    builder.setInsertionPointToEnd(mod->mlir.getBody());
    for (auto [i, s] : vws::enumerate(mod->strtab)) {
        builder.create<hlir::StringOp>(
            builder.getUnknownLoc(),
            StringRef(s.data(), s.size()),
            APInt(64, u64(i))
        );
    }

    /// Codegen functions.
    for (auto f : mod->functions) {
        builder.setInsertionPointToEnd(mod->mlir.getBody());
        GenerateProcedure(f);
    }

    /// Verify the IR.
    if (not no_verify and not mlir::succeeded(mod->mlir.verify()))
        Diag::ICE(ctx, mod->module_decl_location, "Module verification failed");
}

void src::CodeGen::GenerateProcedure(ProcDecl* proc) {
    /// Create the function.
    auto ty = Ty(proc->type);

    mlir::NamedAttribute attrs[] = {
        mlir::NamedAttribute{
            builder.getStringAttr("sym_visibility"),
            builder.getStringAttr(proc->exported ? "public" : "private"),
        },
    };

    auto func = builder.create<mlir::func::FuncOp>(
        proc->location.mlir(ctx),
        proc->name,
        ty.cast<mlir::FunctionType>(),
        ArrayRef<mlir::NamedAttribute>{attrs}
    );

    /// Generate the function body, if there is one.
    if (proc->body) {
        /// Create local variables for parameters.
        builder.setInsertionPointToEnd(func.addEntryBlock());
        for (auto [i, p] : vws::enumerate(proc->params)) {
            p->emitted = true;
            p->mlir = builder.create<hlir::LocalVarOp>(
                p->location.mlir(ctx),
                Ty(p->type),
                p->type.align(ctx) / 8
            );

            /// Store the initial parameter value in the variable.
            builder.create<hlir::StoreOp>(
                p->location.mlir(ctx),
                p->mlir,
                func.getArgument(u32(i)),
                p->type.align(ctx) / 8
            );
        }

        /// Emit the body.
        Generate(proc->body);

        /// Insert a return expression at the end if there isn’t already one.
        if (func.back().empty() or not func.back().back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            /// Function returns void.
            if (Type::Equal(proc->ret_type, Type::Void)) {
                builder.create<mlir::func::ReturnOp>(
                    proc->location.mlir(ctx),
                    ArrayRef<mlir::Value>{}
                );
            }

            /// Function does not return, or all paths return a value, but there
            /// is no return expression at the very end.
            else if (Type::Equal(proc->ret_type, Type::NoReturn) or not proc->body->implicit) {
                builder.create<mlir::LLVM::UnreachableOp>(proc->location.mlir(ctx));
            }

            /// Function is a `= <expr>` function that returns its body.
            else {
                Assert(proc->body->mlir, "Inferred procedure body must yield a value");
                builder.create<mlir::func::ReturnOp>(
                    proc->location.mlir(ctx),
                    proc->body->mlir
                );
            }
        }
    }
}
