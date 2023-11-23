add_mlir_dialect(HLIROps hlir)
add_mlir_doc(HLIRDialect HLIRDialect HLIR/ -gen-dialect-doc)
add_mlir_doc(HLIROps HLIROps HLIR/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS HLIRDialect.td)
mlir_tablegen(HLIREnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(HLIREnumAttr.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHLIRDialectEnumIncGen)
add_dependencies(mlir-headers MLIRHLIRDialectEnumIncGen)
