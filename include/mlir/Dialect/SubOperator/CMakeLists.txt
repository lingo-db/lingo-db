
set(LLVM_TARGET_DEFINITIONS SubOperatorOps.td)
mlir_tablegen(SubOperatorOpsDialect.h.inc -gen-dialect-decls --dialect=subop)
mlir_tablegen(SubOperatorOpsDialect.cpp.inc -gen-dialect-defs --dialect=subop)
mlir_tablegen(SubOperatorOpsTypes.h.inc --gen-typedef-decls --typedefs-dialect=subop)
mlir_tablegen(SubOperatorOpsTypes.cpp.inc --gen-typedef-defs --typedefs-dialect=subop)
mlir_tablegen(SubOperatorOpsAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect=subop)
mlir_tablegen(SubOperatorOpsAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect=subop)
mlir_tablegen(SubOperatorOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SubOperatorOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SubOperatorOps.h.inc -gen-op-decls)
mlir_tablegen(SubOperatorOps.cpp.inc -gen-op-defs)
set(LLVM_TARGET_DEFINITIONS SubOpInterfaces.td)
mlir_tablegen(SubOperatorOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SubOperatorOpsInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(SubOperatorOpsTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(SubOperatorOpsTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRSubOperatorOpsIncGen)
add_dependencies(build_includes MLIRSubOperatorOpsIncGen)
add_mlir_doc(SubOperatorOps SubOperatorDialect SubOperator/ -gen-dialect-doc --dialect=subop)
