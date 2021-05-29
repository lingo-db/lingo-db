
set(LLVM_TARGET_DEFINITIONS UtilOps.td)
mlir_tablegen(UtilOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(UtilOps.h.inc -gen-op-decls)
mlir_tablegen(UtilOps.cpp.inc -gen-op-defs)
mlir_tablegen(UtilOpsTypes.h.inc --gen-typedef-decls)
mlir_tablegen(UtilOpsTypes.cpp.inc --gen-typedef-defs)
add_public_tablegen_target(MLIRUtilOpsIncGen)
#add_mlir_doc(UtilDialect -gen-dialect-doc UtilDialect Util/)
#add_mlir_doc(UtilOps -gen-op-doc UtilOps Util/)

