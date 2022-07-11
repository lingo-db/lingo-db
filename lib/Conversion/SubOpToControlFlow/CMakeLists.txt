add_mlir_conversion_library(MLIRSubOpToControlFlow
        SubOpToControlFlow.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/SubOpToControlFlow

        DEPENDS
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDB
        MLIRDSA
        MLIRSubOperator
        )
