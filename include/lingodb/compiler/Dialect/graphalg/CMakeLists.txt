#include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
#include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)

set(LLVM_TARGET_DEFINITIONS GraphAlgAttr.td)
mlir_tablegen(GraphAlgEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(GraphAlgEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(GraphAlgAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(GraphAlgAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS GraphAlgOps.td)
mlir_tablegen(GraphAlgOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(GraphAlgOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(GraphAlgOps graphalg)

set(LLVM_TARGET_DEFINITIONS GraphAlgPasses.td)
mlir_tablegen(GraphAlgPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRGraphAlgPassesIncGen)

set(LLVM_TARGET_DEFINITIONS GraphAlgInterfaces.td)
mlir_tablegen(GraphAlgTypeInterfaces.h.inc --gen-type-interface-decls)
mlir_tablegen(GraphAlgTypeInterfaces.cpp.inc --gen-type-interface-defs)
mlir_tablegen(GraphAlgOpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(GraphAlgOpInterfaces.cpp.inc --gen-op-interface-defs)
add_public_tablegen_target(MLIRGraphAlgInterfacesIncGen)

add_dependencies(build_dialect_includes MLIRGraphAlgInterfacesIncGen)