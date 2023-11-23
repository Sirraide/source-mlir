#ifndef HLIR_DIALECT_TD
#define HLIR_DIALECT_TD

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
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
    let useDefaultTypePrinterParser = 0;

    let extraClassDeclaration = [{
        void printType(Type, DialectAsmPrinter &) const override;
    }];
}

class HLIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<HLIR_Dialect, mnemonic, traits>;

class HLIR_Attr<
    string name,
    string attrMnemonic,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Attribute"
> : AttrDef<HLIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

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

class HLIR_SingleElementType<
    string name,
    string type_mnemonic,
    list<Trait> traits = []
> : HLIR_Type<
    name,
    type_mnemonic,
    traits # [
        DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>
    ]
> {
    let parameters = (ins "Type":$elem);
    let hasCustomAssemblyFormat = 1;
    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$elem), [{
            return $_get(elem.getContext(), elem);
        }]>
    ];
}

/// Used only so we can certain operations (e.g. `DeferOp`s) around.
def HLIR_TokenType : HLIR_Type<"Token", "token">;

def HLIR_SliceType : HLIR_SingleElementType<"Slice", "slice">;
def HLIR_ScopedPointerType : HLIR_SingleElementType<"ScopedPointer", "scp">;
def HLIR_ReferenceType : HLIR_SingleElementType<"Reference", "ref">;
def HLIR_ClosureType : HLIR_SingleElementType<"Closure", "closure">;

def HLIR_ArrayType : HLIR_SingleElementType<"Array", "array"> {
    let parameters = (ins "Type":$elem, "std::size_t":$size);
    let hasCustomAssemblyFormat = 1;
    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$elem, "std::size_t":$size), [{
            return $_get(elem.getContext(), elem, size);
        }]>
    ];
}


def HLIR_Type : AnyTypeOf<[
    Builtin_Integer,
    FunctionType,
    LLVM_AnyStruct,
    HLIR_Void,
    HLIR_Int,
    HLIR_ClosureType,
    HLIR_Bool,
    HLIR_SliceType,
    HLIR_ArrayType,
    HLIR_ReferenceType,
    HLIR_ScopedPointerType,
    HLIR_TokenType,
]>;

/// ============================================================================
///  Enums
/// ============================================================================
def LocalInitUninit : I32EnumAttrCase<"Uninit", 0, "uninit">;
def LocalInitZeroinit : I32EnumAttrCase<"Zeroinit", 1, "zeroinit">;
def LocalInitInit : I32EnumAttrCase<"Init", 2, "init">;

def LocalInitEnum : I32EnumAttr<"LocalInit", "LocalOp initialiser kind",
    [LocalInitUninit, LocalInitZeroinit, LocalInitInit]> {
    let cppNamespace = "::mlir::hlir";
    let stringToSymbolFnName = "ToString";
}

def LocalInit : DialectAttr<
    HLIR_Dialect,
    CPred<"::llvm::isa<::mlir::hlir::LocalInitAttr>($_self)">,
    "LocalOp initialisation option attribute"> {
  let storageType = "::mlir::hlir::LocalInitAttr";
  let returnType = "::mlir::hlir::LocalInit";
  let convertFromStorage = "$_self.getValue()";
  let constBuilderCall = "::mlir::hlir::LocalInitAttr::get($_builder.getContext(), $0)";
}

def LocalInitAttr : HLIR_Attr<"LocalInitKind", "local_init"> {
  let parameters = (ins "LocalInit":$LocalInit);
  let assemblyFormat = "`<` $LocalInit `>`";
}

#endif // HLIR_DIALECT_TD