

add_mlir_conversion_library(MLIRSubOpToControlFlow
        SubOpToControlFlow.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../lingodb/compiler/Conversion/SubOpToControlFlow

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
        tls-rt-defs
        st-rt-defs
        paht-rt-defs
        db-rt-defs
        tracing-rt-defs
        lock-rt-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDB
        MLIRArrow
        MLIRSubOperator
        runtime_funcs_ptr
)
