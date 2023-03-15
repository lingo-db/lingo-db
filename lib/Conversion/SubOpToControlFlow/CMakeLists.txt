gen_rt_def(grow-buf-rt-defs "GrowingBuffer.h")
gen_rt_def(buf-rt-defs "Buffer.h")
gen_rt_def(stv-rt-defs "SegmentTreeView.h")
gen_rt_def(heap-rt-defs "Heap.h")
gen_rt_def(idx-rt-defs "ExternalHashIndex.h")
gen_rt_def(hmm-rt-defs "HashMultiMap.h")

add_mlir_conversion_library(MLIRSubOpToControlFlow
        SubOpToControlFlow.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/SubOpToControlFlow

        DEPENDS
        join-ht-rt-defs
        ht-rt-defs
        hmm-rt-defs
        ds-it-rt-defs
        grow-buf-rt-defs
        buf-rt-defs
        stv-rt-defs
        heap-rt-defs
        idx-rt-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDB
        MLIRDSA
        MLIRSubOperator
        )
