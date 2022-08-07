add_mlir_conversion_library(MLIRSubOpToControlFlow
        SubOpToControlFlow.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/SubOpToControlFlow

        DEPENDS
        vec-rt-defs
        join-ht-rt-defs
        ht-rt-defs

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDB
        MLIRDSA
        MLIRSubOperator
        )
