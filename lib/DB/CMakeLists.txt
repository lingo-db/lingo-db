set(LLVM_TARGET_DEFINITIONS Transforms/EliminateNulls.td)
mlir_tablegen(EliminateNulls.inc -gen-rewriters)
add_public_tablegen_target(MLIRDBEliminateNullsIncGen)
set(LLVM_TARGET_DEFINITIONS Transforms/SimplifyToArith.td)
mlir_tablegen(SimplifyToArith.inc -gen-rewriters)
add_public_tablegen_target(MLIRDBSimplifyToArithIncGen)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mlir_dialect_library(MLIRDB
        DBDialect.cpp
        DBOps.cpp
        DBOpsEnums.cpp
        DBTypes.cpp
        RuntimeFunctions.cpp
        Transforms/EliminateNulls.cpp
        Transforms/SimplifyToArith.cpp
        Transforms/OptimizeRuntimeFunctions.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/DB
        ${CMAKE_CURRENT_BINARY_DIR}

        DEPENDS
        MLIRDBOpsIncGen
        MLIRDBEliminateNullsIncGen
        MLIRDBSimplifyToArithIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncTransforms
        MLIRUtil
        )
