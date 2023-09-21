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

        /// No other valid integer conversions.
        return false;
    }

    /// Conversions involving references.
    if (isa<ReferenceType>(from) or isa<ReferenceType>(to)) {
        /// Base types are equal.
        if (Type::Equal(from.strip_refs, to.strip_refs)) {
            auto from_depth = from.ref_depth;
            auto to_depth = to.ref_depth;

            /// If the depth we’re converting to is one greater than
            /// the depth of the expression, and the expression is an
            /// lvalue, then this is reference binding.
            if (to_depth == from_depth + 1 and e->is_lvalue) {
                e = new (mod) CastExpr(CastKind::ReferenceBinding, e, type, e->location);
                Analyse(e);
                return true;
            }

            /// If the type we’re converting to is not a reference, then
            /// this is lvalue to rvalue conversion.
            if (to_depth == 0) {
                InsertLValueToRValueConversion(e);
                return true;
            }

            /// If the depth of the type we’re converting to is less than
            /// the depth of the type we’re converting from, then this is
            /// implicit dereferencing.
            if (to_depth < from_depth) {
                InsertImplicitDereference(e, from_depth - to_depth);
                return true;
            }
        }
    }

    /// No other conversions are supported.
    return false;
}

void src::Sema::InsertImplicitCast(Expr*& e, Expr* to) {
    Expr* cast = new (mod) CastExpr(CastKind::Implicit, e, to, e->location);
    Analyse(cast);
    e = cast;
}

void src::Sema::InsertImplicitDereference(Expr*& e, isz depth) {
    for (isz i = depth; i; i--) {
        e = new (mod) CastExpr(
            CastKind::ImplicitDereference,
            e,
            Type::Unknown,
            e->location
        );

        Analyse(e);
    }
}

void src::Sema::InsertLValueReduction(Expr*& e) {
    if (isa<ReferenceType>(e->type)) {
        Expr* cast = new (mod) CastExpr(
            CastKind::LValueReduction,
            e,
            Type::Unknown,
            e->location
        );

        Analyse(cast);
        e = cast;
    }
}

void src::Sema::InsertLValueToRValueConversion(Expr*& e) {
    if (not e->sema.errored and e->is_lvalue) {
        Expr* cast = new (mod) CastExpr(
            CastKind::LValueToRValue,
            e,
            Type::Unknown,
            e->location
        );

        Analyse(cast);
        e = cast;
    }
}

bool src::Sema::MakeDeclType(Expr*& e) {
    if (not AnalyseAsType(e)) return false;
    if (Type::Equal(e, Type::Void)) return Error(e, "Cannot declare a variable of type 'void'");

    /// IDEA: `proc&` is a function pointer, `proc^` a closure.
    if (isa<ProcType>(e)) return Error(e, "Sorry, closures are not supported yet");
    return true;
}

/// ===========================================================================
///  Analysis
/// ===========================================================================
bool src::Sema::Analyse(Expr*& e) {
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
                else if (i == last) {
                    b->stored_type = expr->type;
                    b->is_lvalue = expr->is_lvalue;
                }
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
                invoke->is_lvalue = isa<ReferenceType>(ptype->ret_type);
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
                        Linkage::Local,
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
                /// Only generated by sema. This performs reference
                /// collapsing, yielding an rvalue.
                case CastKind::LValueToRValue:
                    m->stored_type = m->operand->type.strip_refs;
                    break;

                /// Only generated by sema. This performs reference
                /// collapsing, yielding an lvalue.
                case CastKind::LValueReduction:
                    Assert(m->operand->is_lvalue);
                    m->stored_type = m->operand->type.strip_refs;
                    m->is_lvalue = true;
                    break;

                /// Only generated by sema. Dereference a reference
                /// once, yielding an lvalue.
                case CastKind::ImplicitDereference:
                    m->stored_type = cast<ReferenceType>(m->operand->type)->elem;
                    m->is_lvalue = true;
                    break;

                /// Only generated by sema. Convert an lvalue to a reference.
                case CastKind::ReferenceBinding:
                    m->stored_type = new (mod) ReferenceType(m->operand->type, m->location);
                    break;

                /// Only generated by sema. No-op here. Currently, there
                /// is no implicit cast that yields an lvalue.
                case CastKind::Implicit: break;
            }
        } break;

        /// Member access into a type.
        case Expr::Kind::MemberAccessExpr: {
            auto m = cast<MemberAccessExpr>(e);
            if (not Analyse(m->object)) return e->sema.set_errored();

            /// Convert the object to an rvalue.
            InsertLValueToRValueConversion(m->object);

            /// A slice type has a `data` and a `size` member.
            ///
            /// Neither of these are lvalues since slices are
            /// supposed to be pretty much immutable, and you
            /// should create a new one rather than changing
            /// the size or the data pointer.
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
            if (d->decl->sema.errored) d->sema.set_errored();
            else {
                d->stored_type = d->decl->type;
                d->is_lvalue = isa<VarDecl, ParamDecl>(d->decl);
            }
        } break;

        /// Make sure the type is valid.
        case Expr::Kind::ParamDecl: {
            auto param = cast<ParamDecl>(e);
            if (not AnalyseAsType(param->stored_type) or not MakeDeclType(param->stored_type))
                return e->sema.set_errored();
            param->is_lvalue = true;
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
            /// is an initialiser, it must be convertible to the type of the variable;
            /// if the variable is a reference, the initialiser may require dereferencing
            /// or reference binding.
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
            if (not var->sema.errored) {
                curr_scope->declare(var->name, var);
                var->is_lvalue = true;
            }
        } break;

        /// Unary expressions.
        case Expr::Kind::UnaryPrefixExpr: {
            auto u = cast<UnaryPrefixExpr>(e);
            if (not Analyse(u->operand)) return e->sema.set_errored();
            switch (u->op) {
                default: Unreachable("Invalid unary prefix operator");

                /// Explicit dereference.
                case Tk::Star: {
                    auto ref = dyn_cast<ReferenceType>(u->operand->type);
                    if (not ref) return Error(
                        u,
                        "Cannot dereference value of non-reference type '{}'",
                        u->operand->type
                    );

                    /// The type is the element type of the reference.
                    u->stored_type = ref->elem;
                } break;
            }
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
                    /// Operands are rvalues of non-reference type.
                    InsertLValueToRValueConversion(b->lhs);
                    InsertLValueToRValueConversion(b->rhs);

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
                    /// don’t have the same size. Integer conversions from a
                    /// smaller to a larger type can never fail, which is why
                    /// we assert rather than error here.
                    if (not Type::Equal(b->lhs->type, b->rhs->type)) {
                        auto lsz = b->lhs->type.size(mod->context);
                        auto rsz = b->rhs->type.size(mod->context);
                        if (lsz >= rsz) Assert(Convert(b->rhs, b->lhs->type));
                        else Assert(Convert(b->lhs, b->rhs->type));
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
                    /// Look through rvalues, but allow comparing references if
                    /// their addresses are taken.
                    InsertLValueToRValueConversion(b->lhs);
                    InsertLValueToRValueConversion(b->rhs);

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

                /// Value assignment. The LHS has to be an lvalue.
                case Tk::Assign: {
                    InsertLValueReduction(b->lhs);
                    InsertLValueToRValueConversion(b->rhs);
                    if (not b->lhs->is_lvalue) return Error(
                        b,
                        "Left-hand side of `=` must be an lvalue"
                    );

                    /// The RHS must be convertible to the LHS.
                    if (not Convert(b->rhs, b->lhs->type)) return Error(
                        b,
                        "Cannot assign '{}' to '{}'",
                        b->rhs->type,
                        b->lhs->type
                    );

                    /// The type of the expression is the type of the LHS.
                    b->stored_type = b->lhs->type;
                    b->is_lvalue = true;
                } break;

                /// Reference assignment.
                ///
                /// This one is more complicated. The LHS must be a reference,
                /// and the RHS must either be an lvalue whose type is exactly
                /// the element type of the LHS, or another reference of the
                /// same type as the LHS.
                ///
                /// If the LHS is a multi-level reference, and the conversion
                /// fails, a level of indirection is successively removed until
                /// either no level remains or the conversion succeeds.
                case Tk::RDblArrow: {
                    /// Better error message if the LHS isn’t even a reference
                    /// to begin with.
                    if (not isa<ReferenceType>(b->lhs->type)) return Error(
                        b,
                        "LHS of reference binding must be a reference, but was '{}'",
                        b->lhs->type
                    );

                    /// For error reporting *only*.
                    auto ltype_saved = b->lhs->type;
                    auto rtype_saved = b->rhs->type;

                    /// If both operands are references, dereference them until
                    /// they have the same level of indirection.
                    if (isa<ReferenceType>(b->rhs->type)) {
                        auto lvl_l = b->lhs->type.ref_depth;
                        auto lvl_r = b->rhs->type.ref_depth;
                        if (lvl_l != lvl_r) InsertImplicitDereference(
                            lvl_l > lvl_r ? b->lhs : b->rhs,
                            std::abs<isz>(lvl_l - lvl_r)
                        );
                    }

                    /// If the references are now of the same type, then the
                    /// rebinding is now simply an assignment.
                    if (Type::Equal(b->lhs->type, b->rhs->type))
                        b->stored_type = b->lhs->type;

                    /// Otherwise, if the RHS is an lvalue whose type is the
                    /// element type of the LHS, then this is reference binding.
                    else if (
                        not isa<ReferenceType>(b->rhs->type) and
                        Type::Equal(b->rhs->type, cast<ReferenceType>(b->lhs->type)->elem)
                    ) {
                        b->rhs = new (mod) CastExpr(
                            CastKind::ReferenceBinding,
                            b->rhs,
                            Type::Unknown,
                            e->location
                        );

                        Analyse(b->rhs);
                        b->stored_type = b->lhs->type;
                    }

                    /// Otherwise, this is not a valid rebinding operation.
                    else {
                        return Error(
                            b,
                            "No valid reference binding to '{}' from '{}'",
                            ltype_saved,
                            rtype_saved
                        );
                    }
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
