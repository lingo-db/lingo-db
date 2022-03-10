set(LLVM_TARGET_DEFINITIONS Transforms/EliminateNulls.td)
mlir_tablegen(EliminateNulls.inc -gen-rewriters)
add_public_tablegen_target(MLIRDBEliminateNullsIncGen)
set(LLVM_TARGET_DEFINITIONS Transforms/SimplifyToArith.td)
mlir_tablegen(SimplifyToArith.inc -gen-rewriters)
add_public_tablegen_target(MLIRDBSimplifyToArithIncGen)
add_mlir_dialect_library(MLIRDB
        DBDialect.cpp
        DBOps.cpp
        DBOpsEnums.cpp
        DBTypes.cpp
        Transforms/EliminateNulls.cpp
        Transforms/SimplifyToArith.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/DB

        DEPENDS
        MLIRDBOpsIncGen
        MLIRDBEliminateNullsIncGen
        MLIRDBSimplifyToArithIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRStandardOpsTransforms
        MLIRUtil
        )
