#ifndef HLIR_OPS_TD
#define HLIR_OPS_TD

include "HLIRDialect.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMEnums.td"

def FuncOp : HLIR_Op<"func", [
    AffineScope, AutomaticAllocationScope, CallableOpInterface,
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
    let summary = "Function";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        LinkageAttr:$linkage,
        CConv:$cc,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "::mlir::LLVM::Linkage":$linkage,
            "::mlir::LLVM::CConv":$cc,
            "FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
        ), [{
            buildWithEntryBlock($_builder, $_state, name, type, attrs, type.getInputs());
            $_state.addAttribute(getCcAttrName($_state.name), mlir::LLVM::CConvAttr::get($_state.getContext(), cc));
            $_state.addAttribute(getLinkageAttrName($_state.name), mlir::LLVM::LinkageAttr::get($_state.getContext(), linkage));
        }]>
    ];

    /// Most of this is copied from the func dialect’s FuncOp.
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

        /// Returns the argument attributes for all callable region arguments or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableArgAttrs() {
          return getArgAttrs().value_or(nullptr);
        }

        /// Returns the result attributes for all callable region results or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableResAttrs() {
          return getResAttrs().value_or(nullptr);
        }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        //===------------------------------------------------------------------===//
        // OpAsmOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Allow the dialect prefix to be omitted.
        static StringRef getDefaultDialect() { return "hlir"; }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() { return isExternal(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def CallOp : HLIR_Op<"call", [
    CallOpInterface,
    MemRefsNormalizable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "call operation";
    let description = [{ Function call }];
    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        BoolAttr:$inline_call,
        CConvAttr:$cc,
        Variadic<HLIR_Type>:$args
    );
    let results = (outs Optional<HLIR_Type>:$yield);

    let builders = [
        OpBuilder<(ins
            "::mlir::hlir::FuncOp":$callee,
            "bool":$inline_call,
            "::mlir::LLVM::CConv":$cc,
            CArg<"ValueRange", "{}">:$operands
        ), [{
          $_state.addOperands(operands);
          $_state.addAttribute(getCalleeAttrName($_state.name), SymbolRefAttr::get(callee));
          $_state.addAttribute(getCcAttrName($_state.name), mlir::LLVM::CConvAttr::get($_state.getContext(), cc));
          $_state.addAttribute(getInlineCallAttrName($_state.name), mlir::BoolAttr::get($_state.getContext(), inline_call));
          $_state.addTypes(callee.getFunctionType().getResults());
        }]>
    ];

    let extraClassDeclaration = [{
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }
    }];
}

def ReturnOp : HLIR_Op<"return", [
    Pure, HasParent<"FuncOp">,
    MemRefsNormalizable, ReturnLike, Terminator
]> {
    let summary = "Function return operation";
    let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";
    let arguments = (ins Optional<HLIR_Type>:$operand);
    let hasVerifier = 0;
}

def StringOp : HLIR_Op<"string"> {
    let summary = "String literal";
    let arguments = (ins StrAttr:$value, IndexAttr:$index);

    let builders = [
        OpBuilder<(ins "::mlir::StringAttr":$value, "::mlir::IntegerAttr":$index), [{
            build($_builder, $_state, {}, value, index);
        }]>,

        OpBuilder<(ins "::llvm::StringRef":$value, "::llvm::APInt":$index), [{
            build(
                $_builder,
                $_state,
                {},
                value,
                index
            );
        }]>,
    ];

    let assemblyFormat = "$index `,` $value attr-dict";
    let hasVerifier = 0;
}

def SliceDataOp : HLIR_Op<"slice_data"> {
    let summary = "Get the data pointer of a slice";
    let arguments = (ins HLIR_SliceType:$operand);
    let results = (outs HLIR_ReferenceType:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand), [{
            build(
                $_builder,
                $_state,
                ReferenceType::get($_state.getContext(), operand.getType().cast<SliceType>().getElem()),
                operand
            );
        }]>
    ];
}

def SliceSizeOp : HLIR_Op<"slice_size"> {
    let summary = "Get the size of a slice";
    let arguments = (ins HLIR_SliceType:$operand);
    let results = (outs HLIR_Int:$res);
    let assemblyFormat = "$operand attr-dict `:` type($operand)";
    let hasVerifier = 0;
}

def GlobalRefOp : HLIR_Op<"global_ref"> {
    let summary = "Get a reference to a global variable";
    let arguments = (ins SymbolRefAttr:$name);
    let results = (outs HLIR_ReferenceType:$res);
    let assemblyFormat = "$name attr-dict `:` type($res)";
    let hasVerifier = 0;
}

def LoadOp : HLIR_Op<"load"> {
    let summary = "Load a value from a reference";
    let arguments = (ins HLIR_ReferenceType:$operand);
    let results = (outs HLIR_Type:$type);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand), [{
            build(
                $_builder,
                $_state,
                operand.getType().cast<ReferenceType>().getElem(),
                operand
            );
        }]>
    ];
}

def StoreOp : HLIR_Op<"store"> {
    let summary = "Store a value to a reference";
    let arguments = (ins
        HLIR_ReferenceType:$addr,
        HLIR_Type:$value,
        Builtin_IntegerAttr:$alignment
    );
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$addr,
            "::mlir::Value":$value,
            "std::int64_t":$alignment
        ), [{
            build(
                $_builder,
                $_state,
                addr,
                value,
                $_builder.getI64IntegerAttr(alignment)
            );
        }]>
    ];
}

def LiteralOp : HLIR_Op<"literal"> {
    let summary = "Literal value";
    let arguments = (ins Variadic<HLIR_Type>:$args);
    let results = (outs HLIR_Type:$value);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::hlir::SliceType":$type,
            "::mlir::Value":$data,
            "::mlir::Value":$size
        ), [{
            build(
                $_builder,
                $_state,
                type,
                {data, size}
            );
        }]>
    ];
}

def ZeroinitialiserOp : HLIR_Op<"zeroinitialiser"> {
    let summary = "Zeroinitialiser";
    let arguments = (ins HLIR_ReferenceType:$operand, Builtin_IntegerAttr:$bytes);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand, "::std::int64_t":$type_size_in_bytes), [{
            build(
                $_builder,
                $_state,
                {},
                operand,
                $_builder.getI64IntegerAttr(type_size_in_bytes)
            );
        }]>
    ];
}

def ArrayDecayOp : HLIR_Op<"array_decay"> {
    let summary = "Decay a reference to an array to a reference to its first element";
    let arguments = (ins HLIR_ReferenceType:$operand);
    let results = (outs HLIR_ReferenceType:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$operand), [{
            auto type = ReferenceType::get(
                $_state.getContext(),
                operand
                    .getType()
                    .cast<ReferenceType>()
                    .getElem()
                    .cast<ArrayType>()
                    .getElem()
            );

            build(
                $_builder,
                $_state,
                type,
                operand
            );
        }]>
    ];
}

def HLIR_LocalOp : HLIR_Op<"local"> {
    let summary = "Local variable";
    let arguments = (ins Builtin_IntegerAttr:$alignment);
    let results = (outs HLIR_ReferenceType:$res);
    let hasVerifier = 0;
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Type":$type, "std::int64_t":$alignment), [{
            $_state.addTypes(ReferenceType::get($_state.getContext(), type));
            $_state.addAttribute("alignment", $_builder.getI64IntegerAttr(alignment));
        }]>
    ];
}

def ExpIOp : HLIR_Op<
    "expi", [
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<VectorUnrollOpInterface>,
        NoMemoryEffect
    ] # ElementwiseMappable.traits> {
    let summary = "Exponential function";
    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike:$res);
    let hasVerifier = 0;
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$lhs,
            "::mlir::Value":$rhs
        ), [{
            build(
                $_builder,
                $_state,
                lhs.getType(),
                {lhs, rhs}
            );
        }]>
    ];
}

def StructGEPOp : HLIR_Op<"structgep"> {
    let summary = "Get a pointer to a struct element by offsetting from a struct pointer";
    let arguments = (ins HLIR_ReferenceType:$struct_ref, Builtin_IntegerAttr:$idx);
    let results = (outs HLIR_ReferenceType:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$struct_ref,
            "::std::int32_t":$idx
        ), [{
            auto ty = struct_ref
                .getType().cast<ReferenceType>()
                .getElem().cast<LLVM::LLVMStructType>()
                .getBody()[idx];

            build(
                $_builder,
                $_state,
                hlir::ReferenceType::get(ty),
                struct_ref,
                $_builder.getI32IntegerAttr(idx)
            );
        }]>
    ];
}

def ChainExtractLocalOp : HLIR_Op<"extractlocal"> {
    let summary = "Extract a local variable from a static chain area";
    let arguments = (ins HLIR_ReferenceType:$struct_ref, Builtin_IntegerAttr:$idx);
    let results = (outs HLIR_Type:$res);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 0;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$struct_ref,
            "::std::int32_t":$idx
        ), [{
            auto ty = struct_ref
                .getType().cast<ReferenceType>()
                .getElem().cast<LLVM::LLVMStructType>()
                .getBody()[idx];

            build(
                $_builder,
                $_state,
                ty,
                struct_ref,
                $_builder.getI32IntegerAttr(idx)
            );
        }]>
    ];
}

def MakeClosureOp : HLIR_Op<"makeclosure"> {
    let summary = "Make a closure";
    let arguments = (ins
        HLIR_Type:$closure,
        Optional<HLIR_Type>:$env
    );
    let results = (outs HLIR_ClosureType);

    let hasCustomAssemblyFormat = 1;
}

def InvokeClosureOp : HLIR_Op<"invokeclosure"> {
    let summary = "Invoke a closure";
    let arguments = (ins
        HLIR_ClosureType:$closure,
        Variadic<HLIR_Type>:$args
    );
    let results = (outs HLIR_Type);

    let hasCustomAssemblyFormat = 1;
}

def NewOp : HLIR_Op<"new"> {
    let summary = "Allocate an object on the heap";
    let results = (outs HLIR_ReferenceType);
    let hasCustomAssemblyFormat = 1;
}

def DeleteOp : HLIR_Op<"delete"> {
    let summary = "Free an object allocated on the heap";
    let arguments = (ins HLIR_ReferenceType:$object);
    let hasCustomAssemblyFormat = 1;
}

// def LocalEscapeOp : HLIR_Op<"localescape"> {
//     let summary = "Make a local variables visible in nested functions";
//     let arguments = (ins Variadic<HLIR_ReferenceType>:$vars);
//     let hasVerifier = 0;
//     let assemblyFormat = "`(` $vars `)` attr-dict `:` type($vars)";
//
//     let builders = [
//         OpBuilder<(ins "::mlir::ValueRange":$vars), [{
//             build(
//                 $_builder,
//                 $_state,
//                 {},
//                 vars
//             );
//         }]>
//     ];
// }
//
// def LocalRecoverOp : HLIR_Op<"localrecover"> {
//     let summary = "Recover a local variable from a frame pointer";
//     let arguments = (ins FlatSymbolRefAttr:$func, HLIR_StaticChainPointerType:$fp, Builtin_IntegerAttr:$idx);
//     let results = (outs HLIR_ReferenceType:$var);
//     let hasVerifier = 0;
//     let assemblyFormat = "$fp `:` type($var) attr-dict";
//
//     let builders = [
//         OpBuilder<(ins
//             "::mlir::Type":$type,
//             "::llvm::StringRef":$func,
//             "::mlir::Value":$fp,
//             "::std::int32_t":$idx
//         ), [{
//             build(
//                 $_builder,
//                 $_state,
//                 ::hlir::ReferenceType::get($_state.getContext(), type),
//                 SymbolRefAttr::get($_state.getContext(), func),
//                 fp,
//                 $_builder.getI32IntegerAttr(idx)
//             );
//         }]>
//     ];
// }
//
// def LocalAddressOp : HLIR_Op<"localaddress"> {
//     let summary = "Get a frame pointer suitable for passing to hlir.localrecover";
//     let hasVerifier = 0;
//     let assemblyFormat = "attr-dict";
//     let results = (outs HLIR_StaticChainPointerType);
//
//     let builders = [
//         OpBuilder<(ins), [{
//             build(
//                 $_builder,
//                 $_state,
//                 ::hlir::StaticChainPointerType::get($_state.getContext())
//             );
//         }]>
//     ];
// }

#endif // HLIR_OPS_TD
