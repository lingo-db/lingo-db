#include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
#include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)

set(LLVM_TARGET_DEFINITIONS GARelAttr.td)
mlir_tablegen(GARelEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(GARelEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(GARelAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(GARelAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS GARelOps.td)
add_mlir_dialect(GARelOps garel)

set(LLVM_TARGET_DEFINITIONS GARelPasses.td)
mlir_tablegen(GARelPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRGARelPassesIncGen)
add_dependencies(build_dialect_includes MLIRGARelPassesIncGen)
