add_mlir_dialect_library(MLIRSubOperator
        SubOperatorDialect.cpp
        SubOperatorOps.cpp
        SubOperatorTypes.cpp
        SubOperatorAttributes.cpp
        SubOpInterfaces.cpp
        Transforms/SubOpDependencyAnalysis.cpp
        Transforms/Passes.cpp
        Transforms/EnforceOrder.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/SubOperator

        DEPENDS
        MLIRSubOperatorOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        )
