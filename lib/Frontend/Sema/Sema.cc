#include <source/Frontend/Sema.hh>

/// ===========================================================================
///  Helpers
/// ===========================================================================
bool src::Sema::Convert(Expr*& e, Expr* type) {
    /// Sanity checks.
    if (e->sema.errored or type->sema.errored) return false;
    Assert(type->sema.ok, "Cannot convert to unanalysed type");
    Assert(isa<Type>(type));
    auto from = e->type;
    auto to = type->as_type;

    /// If the types are equal, then they’re convertible to one another.
    if (Type::Equal(from, to)) return true;

    /// Smaller integer types can be converted to larger integer types.
    if (from.is_int(true) and to.is_int(true)) {
        auto from_size = from.size(mod->context);
        auto to_size = to.size(mod->context);
        if (from_size <= to_size) {
            InsertImplicitCast(e, type);
            return true;
        }
    }

    /// No other conversions are supported yet.
    return Error(e, "Cannot convert from '{}' to '{}'", from, to);
}

void src::Sema::InsertImplicitCast(src::Expr*& e, src::Expr* to) {
    Expr* cast = new (mod) CastExpr(CastKind::Implicit, e, to, e->location);
    Analyse(cast);
    e = cast;
}

void src::Sema::InsertLValueToRValueConversion(src::Expr*& e) {
    Expr* cast = new (mod) CastExpr(CastKind::LValueToRValue, e, Type::Unknown, e->location);
    Analyse(cast);
    e = cast;
}

bool src::Sema::MakeDeclType(src::Expr*& e) {
    if (not AnalyseAsType(e)) return false;
    if (Type::Equal(e, Type::Void)) return Error(e, "Cannot declare a variable of type 'void'");

    /// IDEA: `proc&` is a function pointer, `proc^` a closure.
    if (isa<ProcType>(e)) return Error(e, "Sorry, closures are not supported yet");
    return true;
}

/// ===========================================================================
///  Analysis
/// ===========================================================================
bool src::Sema::Analyse(src::Expr*& e) {
    /// Don’t analyse the same expression twice.
    if (e->sema.analysed or e->sema.in_progress) return e->sema.ok;
    if (e->sema.errored) return false;
    e->sema.set_in_progress();
    defer { e->sema.set_done(); };
    switch (e->kind) {
        /// No-ops.
        case Expr::Kind::BuiltinType:
        case Expr::Kind::FFIType:
            break;

        /// Integers are of type int.
        case Expr::Kind::IntegerLiteralExpr:
            cast<IntLitExpr>(e)->stored_type = BuiltinType::Int(mod, e->location);
            break;

        /// String literals are u8 slices.
        case Expr::Kind::StringLiteralExpr: {
            auto str = cast<StrLitExpr>(e);
            auto loc = str->location;
            Expr* u8 = IntType::Create(mod, 8);
            Analyse(u8);

            /// Unlike in C++, string literals are *not* lvalues; rather a
            /// new string slice is constructed every time a string literal
            /// is used.
            str->stored_type = new (mod) SliceType(u8, loc);
            Analyse(str->stored_type);
        } break;

        /// Procedures are handled elsewhere.
        case Expr::Kind::ProcDecl:
            AnalyseProcedure(cast<ProcDecl>(e));
            break;

        /// `i0` is illegal.
        case Expr::Kind::IntType:
            if (cast<IntType>(e)->bits == 0) Error(e, "Integer type bit width cannot be 0");
            break;

        /// No restrictions here, but check the element types.
        case Expr::Kind::ReferenceType:
        case Expr::Kind::ScopedPointerType:
        case Expr::Kind::OptionalType:
        case Expr::Kind::SliceType:
            /// TODO: Type inference and checking that we don’t have e.g. `void[]`.
            Analyse(cast<SingleElementTypeBase>(e)->elem);
            break;

        /// Parameters and return type must be complete.
        case Expr::Kind::ProcType: {
            auto type = cast<ProcType>(e);

            for (auto& param : type->param_types) {
                if (not AnalyseAsType(param) or not MakeDeclType(param)) e->sema.set_errored();
                else if (Type::Equal(param, Type::Unknown)) {
                    e->sema.set_errored();
                    Error(param, "Type inference is not permitted here");
                }
            }

            if (not AnalyseAsType(type->ret_type)) e->sema.set_errored();
            else if (Type::Equal(type->ret_type, Type::Unknown)) {
                e->sema.set_errored();
                Error(type->ret_type, "Type inference is not permitted here");
            }
        } break;

        /// The type of a block is the type of the last expression.
        case Expr::Kind::BlockExpr: {
            auto b = cast<BlockExpr>(e);
            tempset curr_scope = b->scope;
            long last = std::ssize(b->exprs) - 1;
            for (auto&& [i, expr] : vws::enumerate(b->exprs)) {
                if (not Analyse(expr) and i == last) e->sema.set_errored();
                else if (i == last) b->stored_type = expr->type;
            }
        } break;

        /// An invoke expression may be a procedure call, or a declaration.
        case Expr::Kind::InvokeExpr: {
            /// Analyse the callee first.
            auto invoke = cast<InvokeExpr>(e);
            if (not Analyse(invoke->callee)) return e->sema.set_errored();

            /// If the callee is of function type, and not a type itself,
            /// then this is a function call.
            if (auto ptype = dyn_cast<ProcType>(invoke->callee->type); ptype and not isa<Type>(invoke->callee)) {
                /// Analyse the arguments.
                for (auto& arg : invoke->args)
                    if (not Analyse(arg))
                        e->sema.set_errored();

                /// Make sure the types match.
                for (auto&& [param, arg] : llvm::zip(ptype->param_types, invoke->args)) {
                    if (not Convert(arg, param)) {
                        e->sema.set_errored();
                        Error(
                            arg,
                            "Argument type '{}' is not convertible to parameter type '{}'",
                            arg->type,
                            param
                        );
                    }
                }

                /// Make sure there are as many arguments as parameters.
                if (invoke->args.size() != ptype->param_types.size()) {
                    e->sema.set_errored();
                    Error(
                        invoke,
                        "Expected {} arguments, but got {}",
                        ptype->param_types.size(),
                        invoke->args.size()
                    );
                }

                /// The type of the expression is the return type of the callee.
                invoke->stored_type = ptype->ret_type;
            }

            /// Otherwise, if the callee is a type, then this is a declaration.
            else if (isa<Type>(invoke->callee)) {
                if (not MakeDeclType(invoke->callee)) e->sema.set_errored();

                /// The arguments must be DeclRefExprs.
                for (auto& arg : invoke->args) {
                    if (not isa<DeclRefExpr>(arg)) {
                        e->sema.set_errored();
                        Error(arg, "Expected identifier in declaration");
                    }
                }

                /// Helper to create a var decl.
                auto MakeVar = [&](Expr* name, Expr* init) -> VarDecl* {
                    auto dr = cast<DeclRefExpr>(name);
                    return new (mod) VarDecl(
                        mod,
                        dr->name,
                        invoke->callee,
                        init,
                        dr->scope == mod->global_scope
                            ? Linkage::Internal
                            : Linkage::Local,
                        Mangling::Source,
                        name->location
                    );
                };

                /// Rewrite the invocation to a declaration.
                if (invoke->args.size() == 1) {
                    e = MakeVar(invoke->args.front(), invoke->init);
                    return Analyse(e);
                }

                /// If the invoke expression contains multiple declarations
                /// rewrite to a VarListDecl expr.
                else {
                    Todo();
                }
            }

            /// Otherwise, no idea what this is supposed to be.
            else {
                e->sema.set_errored();
                Error(invoke->callee, "Expected procedure or type");
            }
        } break;

        /// Cast expression.
        case Expr::Kind::CastExpr: {
            auto m = cast<CastExpr>(e);
            Analyse(m->operand);
            switch (m->cast_kind) {
                case CastKind::LValueToRValue: {
                    /// Only inserted by sema.
                    auto ref = cast<ReferenceType>(m->operand->type);
                    m->stored_type = ref->elem;
                } break;

                /// Only generated by sema, so no-ops.
                case CastKind::Implicit: break;
            }
        } break;

        /// Member access into a type.
        case Expr::Kind::MemberAccessExpr: {
            auto m = cast<MemberAccessExpr>(e);
            if (not Analyse(m->object)) return e->sema.set_errored();

            /// If the object is a reference type, dereference it.
            while (isa<ReferenceType>(m->object->type))
                InsertLValueToRValueConversion(m->object);

            /// A slice type has a `data` and a `size` member.
            if (auto slice = dyn_cast<SliceType>(m->object->type)) {
                if (m->member == "data") {
                    m->stored_type = new (mod) ReferenceType(slice->elem, m->location);
                    return Analyse(m->stored_type);
                }

                if (m->member == "size") {
                    m->stored_type = BuiltinType::Int(mod, m->location);
                    return true;
                }

                return Error(m, "Type '{}' has no '{}' member", slice, m->member);
            }

            return Error(m, "Cannot perform member access on type '{}'", m->object->type);
        }

        /// Perform name lookup in scope.
        case Expr::Kind::DeclRefExpr: {
            auto* d = cast<DeclRefExpr>(e);
            d->scope->visit(d->name, false, [&](auto&& decls) {
                Assert(not decls.empty(), "Ill-formed symbol table entry");

                /// If there are multiple declarations, take the last.
                Analyse(decls.back());
                d->decl = decls.back();
                return utils::StopIteration;
            });

            /// If we didn’t find anything, then this is an error.
            if (not d->decl) return Error(e, "Unknown symbol '{}'", d->name);

            /// The type of this is the type of the referenced expression.
            /// TODO: This is an lvalue.
            d->stored_type = d->decl->type;
        } break;

        /// Make sure the type is valid.
        case Expr::Kind::ParamDecl: {
            auto param = cast<ParamDecl>(e);
            if (not AnalyseAsType(param->stored_type) or not MakeDeclType(param->stored_type))
                return e->sema.set_errored();
            /// TODO: Check for redeclaration?
        } break;

        /// Variable declaration.
        case Expr::Kind::VarDecl: {
            auto var = cast<VarDecl>(e);
            if (not AnalyseAsType(var->stored_type)) return e->sema.set_errored();

            /// If the type is unknown, then we must infer it from
            /// the initialiser.
            if (Type::Equal(var->stored_type, Type::Unknown)) {
                if (not var->init) return Error(var, "Type inference requires an initialiser");
                if (not Analyse(var->init)) return e->sema.set_errored();
                var->stored_type = var->init->type;
                if (not MakeDeclType(var->stored_type)) return e->sema.set_errored();
            }

            /// Otherwise, the type of the declaration must be valid, and if there
            /// is an initialiser, it must be convertible to the type of the variable.
            else {
                if (not MakeDeclType(var->stored_type)) return e->sema.set_errored();
                if (var->init) {
                    /// No need to set the variable itself to errored since we know its type.
                    if (not Analyse(var->init)) return false;
                    if (not Convert(var->init, var->stored_type)) {
                        Error(
                            var->init,
                            "Initialiser type '{}' is not convertible to variable type '{}'",
                            var->init->type,
                            var->stored_type
                        );
                    }
                }
            }

            /// Add the variable to the current scope.
            if (not var->sema.errored) curr_scope->declare(var->name, var);
        } break;

        /// Binary operators are complicated.
        case Expr::Kind::BinaryExpr: {
            auto b = cast<BinaryExpr>(e);
            if (not Analyse(b->lhs) or not Analyse(b->rhs))
                return e->sema.set_errored();
            switch (b->op) {
                default: Unreachable("Invalid binary operator");

                /// Arithmetic operators.
                ///
                /// Note: `and` and `or` are emitted lazily, but this
                /// is irrelevant to semantic analysis, so we don’t
                /// care about that here.
                case Tk::Plus:
                case Tk::Minus:
                case Tk::Star:
                case Tk::StarStar:
                case Tk::Slash:
                case Tk::Percent:
                case Tk::And:
                case Tk::Or:
                case Tk::Xor:
                case Tk::ShiftLeft:
                case Tk::ShiftRight:
                case Tk::ShiftRightLogical: {
                    /// Both types must be integers.
                    if (
                        not b->lhs->type.is_int(true) or
                        not b->rhs->type.is_int(true)
                    ) return Error( //
                        b,
                        "Operands of '{}' must be integers, but got '{}' and '{}'",
                        Spelling(b->op),
                        b->lhs->type,
                        b->rhs->type
                    );

                    /// The smaller integer is cast to the larger type if they
                    /// don’t have the same size.
                    if (not Type::Equal(b->lhs->type, b->rhs->type)) {
                        auto lsz = b->lhs->type.size(mod->context);
                        auto rsz = b->rhs->type.size(mod->context);
                        if (lsz >= rsz) {
                            if (not Convert(b->rhs, b->lhs->type))
                                return b->sema.set_errored();
                        } else {
                            if (not Convert(b->lhs, b->rhs->type))
                                return b->sema.set_errored();
                        }
                    }

                    /// The result type is that integer type.
                    b->stored_type = b->lhs->type;
                } break;

                /// Comparison operators.
                case Tk::EqEq:
                case Tk::Neq:
                case Tk::Lt:
                case Tk::Gt:
                case Tk::Le:
                case Tk::Ge: {
                    /// Both types must be equal.
                    if (
                        not Convert(b->lhs, b->rhs->type) and
                        not Convert(b->rhs, b->lhs->type)
                    ) return Error( //
                        b,
                        "Cannot compare '{}' with '{}'",
                        Spelling(b->op),
                        b->lhs->type,
                        b->rhs->type
                    );

                    /// The type of a comparison is bool.
                    b->stored_type = BuiltinType::Bool(mod, b->location);
                } break;
            }
        }
    }

    /// Can’t check for 'ok' as that may not be set yet.
    return not e->sema.errored;
}

bool src::Sema::AnalyseAsType(Expr*& e) {
    Analyse(e);
    if (e->sema.ok and not isa<Type>(e)) Error(e->location, "Not a type");
    return e->sema.ok;
}

void src::Sema::AnalyseProcedure(src::ProcDecl* proc) {
    tempset curr_proc = proc;

    /// Validate the function type.
    if (not AnalyseAsType(cast<TypedExpr>(proc)->stored_type))
        proc->sema.set_errored();

    /// If there is no body, then there is nothing to do.
    if (not proc->body) return;

    /// Analyse the body. If either it or the procedure
    /// contains an error, we can’t check if the procedure
    /// has a return statement.
    Expr* body = proc->body;
    Analyse(body);
    if (not body->sema.ok or not proc->sema.ok) return;

    /// TODO: Make sure all paths return a value.
}

void src::Sema::AnalyseModule() {
    curr_scope = mod->global_scope;

    /// Analyse all functions.
    for (auto& f : mod->functions) AnalyseProcedure(f);
}
