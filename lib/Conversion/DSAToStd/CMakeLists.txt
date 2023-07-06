

add_mlir_conversion_library(MLIRDSAToStd
        CollectionIterators.cpp
        LowerToStd.cpp
        DSAToStdPatterns.cpp
        CollectionsToStdPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/DSAToStd

        DEPENDS
        tb-rt-defs
        ds-it-rt-defs
        join-ht-rt-defs
        ht-rt-defs
        ec-rt-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDSA
        MLIRSCFTransforms
        runtime_funcs_ptr
        )
