#include <source/Frontend/Sema.hh>

/// ===========================================================================
///  Helpers
/// ===========================================================================
bool src::Sema::Convert(Expr*& e, Expr* type) {
    /// Sanity checks.
    if (e->sema.errored or type->sema.errored) return false;
    Assert(type->sema.ok, "Cannot convert to unanalysed type");
    Assert(isa<Type>(type));

    /// If the types are equal, then they’re convertible to one another.
    if (Type::Equal(e->type, type)) return true;

    /// No other conversions are supported yet.
    return Error(e, "Cannot convert from '{}' to '{}'", e->type, type);
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

                /// TODO: Rewrite to var decl.
                Assert(false, "Sorry, variable declarations are not supported yet");
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
            d->scope->visit(d->name, false, [&] (auto&& decls) {
                Assert(not decls.empty(), "Ill-formed symbol table entry");

                /// If there are multiple declarations, this is an error.
                if (d->decl or decls.size() > 1) {
                    Error(e, "Ambiguous reference to '{}'", d->name);
                    return;
                }

                /// Otherwise, we’ve found the declaration.
                Analyse(decls.front());
                d->decl = decls.front();
            });

            /// If we didn’t find anything, then this is an error.
            if (not d->decl) {
                e->sema.set_errored();
                Error(e, "Unknown symbol '{}'", d->name);
            }

            /// The type of this is the type of the referenced expression.
            /// TODO: This is an lvalue.
            d->stored_type = d->decl->type;
        } break;

        /// Make sure the type is valid.
        case Expr::Kind::ParamDecl: {
            auto param = cast<ParamDecl>(e);
            if (not AnalyseAsType(param->stored_type) or not MakeDeclType(param->stored_type))
                e->sema.set_errored();
            /// TODO: Check for redeclaration?
        } break;

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
    /// Analyse all functions.
    for (auto& f : mod->functions) AnalyseProcedure(f);
}
