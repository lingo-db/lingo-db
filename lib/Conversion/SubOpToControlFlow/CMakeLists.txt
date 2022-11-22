gen_rt_def(grow-buf-rt-defs "GrowingBuffer.h")
gen_rt_def(buf-rt-defs "Buffer.h")

add_mlir_conversion_library(MLIRSubOpToControlFlow
        SubOpToControlFlow.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/SubOpToControlFlow

        DEPENDS
        vec-rt-defs
        join-ht-rt-defs
        ht-rt-defs
        ds-it-rt-defs
        grow-buf-rt-defs
        buf-rt-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDB
        MLIRDSA
        MLIRSubOperator
        )
