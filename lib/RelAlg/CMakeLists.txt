add_mlir_dialect_library(MLIRRelAlg
        RelAlgDialect.cpp
        RelAlgOps.cpp
        RelAlgOpsEnums.cpp
        OperatorInterfaceImpl.cpp
        ExtractNestedOperators.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/RelAlg

        DEPENDS
        MLIRRelAlgOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
