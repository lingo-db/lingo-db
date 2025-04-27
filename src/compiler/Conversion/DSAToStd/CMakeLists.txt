

add_mlir_conversion_library(MLIRArrowToStd
        LowerToStd.cpp
        DSAToStdPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../lingodb/compiler/Conversion/ArrowToStd

        DEPENDS
        ds-it-rt-defs
        join-ht-rt-defs
        ht-rt-defs
        ec-rt-defs
        tb-arrow-column-defs
        tb-arrow-table-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRArrow
        MLIRSCFTransforms
        runtime_funcs_ptr
)
