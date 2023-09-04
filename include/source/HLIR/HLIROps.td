#ifndef HLIR_OPS_TD
#define HLIR_OPS_TD

include "HLIRDialect.td"

def FuncOp : HLIR_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "Function";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
        )>
    ];

    let extraClassDeclaration = [{
        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def PrintOp : HLIR_Op<"print"> {
    let summary = "Print";
    let arguments = (ins HLIR_Type:$value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
    let hasVerifier = 1;
}

def StringOp : HLIR_Op<"string"> {
    let summary = "String literal";
    let arguments = (ins StrAttr:$value);
    let results = (outs HLIR_StringType);

    let builders = [
        OpBuilder<(ins "::mlir::StringAttr":$value), [{
            /// Call auto-generated function.
            build($_builder, $_state, value.getType(), value);
        }]>,

        OpBuilder<(ins "::llvm::StringRef":$value), [{
            /// Call auto-generated function.
            build($_builder, $_state, StringType::get($_builder.getContext()), value);
        }]>
    ];

    let assemblyFormat = "$value attr-dict";
    let hasVerifier = 1;
}

#endif // HLIR_OPS_TD
