
// DAG names.
def args;

// An expression in the AST.
class Expr {
    // Parent class.
    Expr parent;

    // Expression fields that are also constructor parameters.
    dag fields = (args);

    // Extra fields that are not constructor parameters.
    dag extra_fields = (args);

    // Whether the constructor is trivial.
    bit trivial_constructor = true;

    // Extra code to add to the constructor.
    code constructor_body = "";

    // Extra code added to the class.
    code class_body = "";
}

/// ===========================================================================
///  Untyped Expressions.
/// ===========================================================================
def UnwindExpr : Expr {
    dag extra_fields = (args
        "Vector<Expr*>":$unwind,
        "Expr*":$parent_full_expression
    );
}

def AssertExpr : Expr {
    let fields = (args "Expr*":$cond, "Expr*":$msg);
    let extra_fields = (args "String":$message_string);
}

def DeferExpr : Expr {
    let fields = (args "Expr*":$expr);
}

def WhileExpr : Expr {
    let fields = (args "Expr*":$cond, "BlockExpr*":$body);
    let extra_fields = (args
        "mlir::Block*":$cond_block,
        "mlir::Block*":$join_block
    );
}

def ReturnExpr : Expr {
    let parent = UnwindExpr;
    let fields = (args "Expr*":$value);
}

def LoopControlExpr : Expr {
    let parent = UnwindExpr;
    let extra_fields = (args "WhileExpr*":$target);
    let fields = (args
        /// Resolved expression. This is set to the parent
        /// if there is no label. This is resolved in Sema.
        "String":$label,
        "bool":$is_continue
    );
}

def LabelExpr : Expr {
    let trivial_constructor = false;

    let fields = (args
        "ProcDecl*":$in_procedure,
        "std::string":$label,
        "Expr*":$expr
    );

    let extra_fields = (args
        "Expr*":$parent_full_expression,

        /// Parent scope. This is required for forward gotos.
        "BlockExpr*":$parent,

        /// Block that is represented by this label.
        "mlir::Block*":$block,

        /// Whether this label is ever branched to.
        "bool":$used
    );
}

def GotoExpr : Expr {
    let parent = UnwindExpr;
    let fields = (args "std::string":$label);
    let extra_fields = (args "LabelExpr*":$target);
}

def EmptyExpr : Expr {
    let constructor_body = "sema.set_done()";
}

def ExportExpr : Expr {
    let fields = (args "Expr*":$expr);
}

def ModuleRefExpr : Expr {
    let fields = (args "Module*":$module);
    let constructor_body = "sema.set_done()";
}

def OverloadSetExpr : Expr {
    let fields = (args "Vector<ProcDecl*>":$overloads);
    let constructor_body = "sema.set_done()";
}

/// ===========================================================================
///  Typed Expressions.
/// ===========================================================================
def TypedExpr : Expr {
    let fields = (args "Expr*":$stored_type);
}

def BlockExpr : Expr {
    let parent = TypedExpr;

    let fields = (args
        "Module*":$module,
        "BlockExpr*":$parent,
        "bool":$implicit
    );

    let extra_fields = (args
        "Vector<Expr*>":$exprs,
        "StringMap<Vector<Expr*, 1>>":$symbol_table,
        "Expr*":$parent_full_expression,
        "Vector<Expr*>":$unwind,
        "mlir::Operation*":$scope_op
    );

    let class_body = [{
        /// Get the nearest parent scope that is a function scope.
        readonly_decl(BlockExpr*, enclosing_function_scope);

    private:
        /// What kind of scope this is.
        ScopeKind scope_kind = ScopeKind::Block;

    public:
        /// Whether this is a function scope.
        readonly(bool, is_function, return scope_kind == ScopeKind::Function);

        /// Whether this is a struct scope.
        readonly(bool, is_struct, return scope_kind == ScopeKind::Struct);

        /// Declare a symbol in this scope.
        void declare(StringRef name, Expr* value) {
            symbol_table[name].push_back(value);
        }

        /// Mark this scope as a function scope. This cannot be undone.
        void set_function_scope() {
            Assert(scope_kind == ScopeKind::Block);
            scope_kind = ScopeKind::Function;
        }

        /// Mark this scope as a struct scope. This cannot be undone.
        void set_struct_scope() {
            Assert(scope_kind == ScopeKind::Block);
            scope_kind = ScopeKind::Struct;
        }

        /// Visit each symbol with the given name.
        template <typename Func>
        void visit(StringRef name, bool this_scope_only, Func f) {
            if (auto sym = symbol_table.find(name); sym != symbol_table.end())
                if (std::invoke(f, sym->second) == utils::StopIteration)
                    return;
            if (parent and not this_scope_only) parent->visit(name, false, f);
        }

        /// Find the NCA of two blocks in a function. Returns nullptr
        /// if the blocks are not in the same function.
        static auto NCAInFunction(BlockExpr* a, BlockExpr* b) -> BlockExpr*;
    }];
}