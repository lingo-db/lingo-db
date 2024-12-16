

add_mlir_conversion_library(MLIRDSAToStd
        LowerToStd.cpp
        DSAToStdPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../lingodb/compiler/Conversion/DSAToStd

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
        MLIRDSA
        MLIRSCFTransforms
        runtime_funcs_ptr
)
