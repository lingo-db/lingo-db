set(LLVM_TARGET_DEFINITIONS Transforms/EliminateNulls.td)
mlir_tablegen(EliminateNulls.inc -gen-rewriters)
add_public_tablegen_target(MLIRDBEliminateNullsIncGen)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(build_includes MLIRDBEliminateNullsIncGen)


add_mlir_dialect_library(MLIRDB
        DBDialect.cpp
        DBOps.cpp
        DBOpsEnums.cpp
        DBTypes.cpp
        RuntimeFunctions/RuntimeFunctions.cpp
        Transforms/EliminateNulls.cpp
        Transforms/OptimizeRuntimeFunctions.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/DB
        ${CMAKE_CURRENT_BINARY_DIR}

        DEPENDS
        dump-rt-defs
        string-rt-defs
        date-rt-defs
        integer-rt-defs
        float-rt-defs
        decimal-rt-defs
        timing-rt-defs
        MLIRDBOpsIncGen
        MLIRDBEliminateNullsIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncTransforms
        MLIRUtil
        runtime_funcs_ptr
)
