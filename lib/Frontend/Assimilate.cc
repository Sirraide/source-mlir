#include <source/Frontend/AST.hh>

namespace src {
/// A bit like Clang’s ASTImporter, except that it mutates the imported
/// expressions rather than cloning them, hence ‘assimilating’ them.
class Assimilator {
    Module* const to;
    Module* const from;
    BlockExpr* const from_global_scope = from->global_scope;
    BlockExpr* const to_global_scope = to->global_scope;
    DenseSet<Expr*> visited;

    void Assimilate(Type& e) {
        Assimilate(e.ptr);
    }

    template <std::derived_from<Expr> Expression>
    requires (not std::is_same_v<Expr, Expression>)
    void Assimilate(Expression*& expr) {
        if (not expr) return;
        Expr* e = expr;
        Assimilate(e);
        expr = cast<Expression>(e);
    }

    void Assimilate(Expr*& e) {
        if (not e) return;
        if (not visited.contains(e)) {
            visited.insert(e);
            AssimilateChildren(e);
        }
        DoAssimilate(e);
    }

    void AssimilateChildren(Expr* e) {
        if (auto te = dyn_cast<TypedExpr>(e)) Assimilate(te->stored_type);
        switch (e->kind) {
            case Expr::Kind::BoolLitExpr:
            case Expr::Kind::BuiltinType:
            case Expr::Kind::EmptyExpr:
            case Expr::Kind::FieldDecl:
            case Expr::Kind::IntType:
            case Expr::Kind::IntLitExpr:
            case Expr::Kind::Nil:
            case Expr::Kind::StrLitExpr:
                break;

            case Expr::Kind::TypeofType:
                Assimilate(cast<TypeofType>(e)->expr);
                break;

            case Expr::Kind::ArrayType:
            case Expr::Kind::ClosureType:
            case Expr::Kind::OptionalType:
            case Expr::Kind::ReferenceType:
            case Expr::Kind::ScopedPointerType:
            case Expr::Kind::ScopedType:
            case Expr::Kind::SliceType:
            case Expr::Kind::SugaredType: {
                auto ty = cast<SingleElementTypeBase>(e);
                Assimilate(ty->elem);
            } break;

            case Expr::Kind::ProcType: {
                auto p = cast<ProcType>(e);
                Assimilate(p->ret_type);
                for (auto& t : p->parameters) Assimilate(t.type);
                Assimilate(p->static_chain_parent);
                Assimilate(p->smp_parent);
            } break;

            case Expr::Kind::OpaqueType: {
                auto o = cast<OpaqueType>(e);
                DoAssimilate(o->module);
            } break;

            case Expr::Kind::TupleType: {
                auto st = cast<TupleType>(e);
                for (auto& f : st->all_fields) Assimilate(f);
            } break;

            case Expr::Kind::StructType: {
                auto st = cast<StructType>(e);
                for (auto& f : st->all_fields) Assimilate(f);
                for (auto& init : st->initialisers) Assimilate(init);
                for (auto& [_, procs] : st->member_procs)
                    for (auto m : procs)
                        Assimilate(m);
                Assimilate(st->deleter);
                Assimilate(st->scope);
                DoAssimilate(st->module);
            } break;

            case Expr::Kind::WhileExpr: {
                auto w = cast<WhileExpr>(e);
                Assimilate(w->body);
                Assimilate(w->cond);
            } break;

            case Expr::Kind::WithExpr: {
                auto w = cast<WithExpr>(e);
                Assimilate(w->object);
                Assimilate(w->body);
            } break;

            case Expr::Kind::ForInExpr: {
                auto f = cast<ForInExpr>(e);
                Assimilate(f->body);
                Assimilate(f->iter);
                Assimilate(f->index);
                Assimilate(f->range);
            } break;

            case Expr::Kind::AssertExpr: {
                auto a = cast<AssertExpr>(e);
                Assimilate(a->cond);
                Assimilate(a->msg);
                Assimilate(a->cond_str);
                Assimilate(a->file_str);
            } break;

            case Expr::Kind::DeferExpr: {
                auto d = cast<DeferExpr>(e);
                Assimilate(d->expr);
            } break;

            case Expr::Kind::ExportExpr: {
                auto x = cast<ExportExpr>(e);
                Assimilate(x->expr);
            } break;

            case Expr::Kind::LabelExpr: {
                auto l = cast<LabelExpr>(e);
                Assimilate(l->expr);
                Assimilate(l->parent);
                Assimilate(l->parent_full_expression);
            } break;

            case Expr::Kind::ModuleRefExpr: {
                auto m = cast<ModuleRefExpr>(e);
                DoAssimilate(m->module);
            } break;

            case Expr::Kind::AliasExpr: {
                auto a = cast<AliasExpr>(e);
                Assimilate(a->expr);
            } break;

            case Expr::Kind::MaterialiseTemporaryExpr: {
                auto a = cast<MaterialiseTemporaryExpr>(e);
                Assimilate(a->ctor);
            } break;

            case Expr::Kind::ConstructExpr: {
                auto c = cast<ConstructExpr>(e);
                for (auto& arg : c->args_and_init()) Assimilate(arg);
            } break;

            case Expr::Kind::OverloadSetExpr: {
                auto o = cast<OverloadSetExpr>(e);
                for (auto& f : o->overloads) Assimilate(f);
            } break;

            case Expr::Kind::ReturnExpr: {
                auto r = cast<ReturnExpr>(e);
                Assimilate(r->value);
            } break;

            case Expr::Kind::GotoExpr: {
                auto g = cast<GotoExpr>(e);
                for (auto& u : g->unwind) Assimilate(u);
                Assimilate(g->parent_full_expression);
                Assimilate(g->target);
            } break;

            case Expr::Kind::LoopControlExpr: {
                auto l = cast<LoopControlExpr>(e);
                for (auto& u : l->unwind) Assimilate(u);
                Assimilate(l->parent_full_expression);
                Assimilate(l->target);
            } break;

            /// Note: the symbol table should not contain anything that
            /// needs to be replaced (as the top-level function and the
            /// global scope should never be in there), so we don’t bother
            /// iterating over its entries here.
            case Expr::Kind::BlockExpr: {
                auto b = cast<BlockExpr>(e);
                for (auto& x : b->exprs) Assimilate(x);
                for (auto& u : b->unwind) Assimilate(u);
                Assimilate(b->parent);
                Assimilate(b->parent_full_expression);
                DoAssimilate(b->module);
            } break;

            case Expr::Kind::ImplicitThisExpr: {
                auto i = cast<ImplicitThisExpr>(e);
                Assimilate(i->init);
            } break;

            case Expr::Kind::InvokeExpr: {
                auto i = cast<InvokeExpr>(e);
                for (auto& arg : i->args) Assimilate(arg);
                for (auto& arg : i->init_args) Assimilate(arg);
                Assimilate(i->callee);
            } break;

            case Expr::Kind::InvokeBuiltinExpr: {
                auto i = cast<InvokeBuiltinExpr>(e);
                for (auto& arg : i->args) Assimilate(arg);
            } break;

            case Expr::Kind::ConstExpr: {
                auto c = cast<ConstExpr>(e);
                Assimilate(c->expr);
            } break;

            case Expr::Kind::CastExpr: {
                auto c = cast<CastExpr>(e);
                Assimilate(c->operand);
            } break;

            /// Note: field will be analysed when we analyse the
            /// corresponding struct.
            case Expr::Kind::MemberAccessExpr: {
                auto m = cast<MemberAccessExpr>(e);
                Assimilate(m->object);
            } break;

            case Expr::Kind::ScopeAccessExpr: {
                auto s = cast<ScopeAccessExpr>(e);
                Assimilate(s->object);
                Assimilate(s->resolved);
            } break;

            case Expr::Kind::UnaryPrefixExpr: {
                auto u = cast<UnaryPrefixExpr>(e);
                Assimilate(u->operand);
            } break;

            case Expr::Kind::IfExpr: {
                auto i = cast<IfExpr>(e);
                Assimilate(i->cond);
                Assimilate(i->then);
                Assimilate(i->else_);
            } break;

            case Expr::Kind::BinaryExpr: {
                auto b = cast<BinaryExpr>(e);
                Assimilate(b->lhs);
                Assimilate(b->rhs);
            } break;

            case Expr::Kind::AssignExpr: {
                auto a = cast<AssignExpr>(e);
                Assimilate(a->lvalue);
                Assimilate(a->ctor);
            } break;

            case Expr::Kind::EnumType: {
                auto t = cast<EnumType>(e);
                DoAssimilate(t->module);
                Assimilate(t->elem);
                for (auto& elem : t->enumerators) Assimilate(elem);
            } break;

            case Expr::Kind::EnumeratorDecl: {
                auto b = cast<EnumeratorDecl>(e);
                Assimilate(b->stored_type);
                Assimilate(b->initialiser);
            } break;

            case Expr::Kind::TupleExpr: {
                auto t = cast<TupleExpr>(e);
                for (auto& elem : t->elements) Assimilate(elem);
            } break;

            case Expr::Kind::TupleIndexExpr: {
                auto t = cast<TupleIndexExpr>(e);
                Assimilate(t->object);
                Assimilate(t->field);
            } break;

            case Expr::Kind::DeclRefExpr: {
                auto d = cast<DeclRefExpr>(e);
                Assimilate(d->scope);
                Assimilate(d->decl);
            } break;

            case Expr::Kind::LocalRefExpr: {
                auto l = cast<LocalRefExpr>(e);
                Assimilate(l->parent);
                Assimilate(l->decl);
            } break;

            case Expr::Kind::ParenExpr: {
                auto p = cast<ParenExpr>(e);
                Assimilate(p->expr);
            } break;

            case Expr::Kind::SubscriptExpr: {
                auto s = cast<SubscriptExpr>(e);
                Assimilate(s->object);
                Assimilate(s->index);
            } break;

            case Expr::Kind::ArrayLitExpr: {
                auto a = cast<ArrayLitExpr>(e);
                for (auto& elem : a->elements) Assimilate(elem);
                Assimilate(a->result_object);
            } break;

            case Expr::Kind::LocalDecl:
            case Expr::Kind::ParamDecl: {
                auto l = cast<LocalDecl>(e);
                for (auto& i : l->init_args) Assimilate(i);
                Assimilate(l->parent);
                Assimilate(l->ctor);
            } break;

            case Expr::Kind::ProcDecl: {
                auto p = cast<ProcDecl>(e);
                for (auto& x : p->params) Assimilate(x);
                for (auto& [k, v] : p->labels) Assimilate(v);
                for (auto& x : p->captured_locals) Assimilate(x);
                Assimilate(p->parent);
                Assimilate(p->body);
                Assimilate(p->captured_locals_type);
                DoAssimilate(p->module);
            } break;
        }
    }

    void DoAssimilate(Module*& m) {
        if (m == from) m = to;
    }

    void DoAssimilate(Expr*& e) {
        if (e == from->top_level_func) e = to->top_level_func;
        else if (e == from_global_scope) e = to_global_scope;
    }

public:
    explicit Assimilator(Module* to, Module* from) : to(to), from(from) {}

    void operator()(Expr* e) {
        Assimilate(e);
    }

    void finalise() {
        /// Copy over top-level symbols.
        for (auto& [k, v] : from_global_scope->symbol_table)
            utils::append(to_global_scope->symbol_table[k], v);

        /// Move over top-level expressions.
        utils::append(to_global_scope->exprs, from_global_scope->exprs);
    }
};
} // namespace src

void src::Module::assimilate(Module* other) {
    Assimilator a{this, other};

    /// AST merging after Sema sounds like a nightmare.
    Assert(
        not other->top_level_func->sema.analysed,
        "Sorry, assimilating analysed modules is not implemented"
    );

    for (auto& i : other->imports)
        if (not utils::contains(imports, i))
            imports.push_back(i);

    /// Move over all functions, but exclude the top-level
    /// function of the assimilated module.
    Assert(other->functions.front() == other->top_level_func);
    utils::append(functions, ArrayRef<ProcDecl*>(other->functions).drop_front());

    /// Do NOT call Assimilate() here as we do NOT want to actually
    /// replace any of these, just their children.
    for (auto s : other->exprs) a(s);
    for (auto& [k, v] : other->exports) utils::append(exports[k], v);
    utils::append(named_structs, other->named_structs);
    utils::append(exprs, other->exprs);
    a.finalise();

    /// Move over identifier table and allocator.
    owned_objects.emplace_back(other->alloc.release());
    owned_objects.emplace_back(other->tokens.release());

    /// Make sure we don’t attempt to delete expressions twice.
    other->exprs.clear();
}
