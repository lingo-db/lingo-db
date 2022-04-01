
add_mlir_dialect_library(MLIRDSA
        DSADialect.cpp
        DSAOps.cpp
        DSAOpsEnums.cpp
        DSATypes.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/DSA

        DEPENDS
        MLIRDSAOpsIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncTransforms
        MLIRUtil
        )
