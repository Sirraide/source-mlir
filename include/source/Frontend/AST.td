
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