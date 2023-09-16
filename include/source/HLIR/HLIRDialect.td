#ifndef HLIR_DIALECT_TD
#define HLIR_DIALECT_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def HLIR_Dialect : Dialect {
    let name = "hlir";
    let summary = "Source High-level Intermediate Representation";
    let cppNamespace = "::mlir::hlir";
    let description = [{
        High-Level IR for the Source compiler.
    }];

    /// Use default printer/parser for types and attrs.
    //let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

class HLIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<HLIR_Dialect, mnemonic, traits>;

/// ============================================================================
///  Types
/// ============================================================================
def HLIR_Void : TypeAlias<NoneType, "void">;
def HLIR_Bool : TypeAlias<I1, "bool">;
def HLIR_Int : TypeAlias<Index, "int">;

class HLIR_Type<string name, string type_mnemonic, list<Trait> traits = []> :
    TypeDef<HLIR_Dialect, name, traits> {
    let mnemonic = type_mnemonic;
}

def HLIR_SliceType : HLIR_Type<"Slice", "slice"> {
    let parameters = (ins "Type":$elem);
    let assemblyFormat = "$elem `[` `]`";
    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$elem), [{
            return $_get(elem.getContext(), elem);
        }]>
    ];
}

def HLIR_Type : AnyTypeOf<[
    Builtin_None,
    Builtin_Index,
    Builtin_Integer,
    HLIR_SliceType,
]>;

#endif // HLIR_DIALECT_TD