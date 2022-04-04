gen_rt_def(tb-rt-defs "TableBuilder.h")
gen_rt_def(ds-it-rt-defs "DataSourceIteration.h")
gen_rt_def(vec-rt-defs "Vector.h")
gen_rt_def(join-ht-rt-defs "LazyJoinHashtable.h")
gen_rt_def(ht-rt-defs "Hashtable.h")

add_mlir_conversion_library(MLIRDSAToStd
        CollectionIterators.cpp
        LowerToStd.cpp
        ScalarDSAToStdPatterns.cpp
        DSAToStdPatterns.cpp
        CollectionsToStdPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/DSAToStd

        DEPENDS
        tb-rt-defs
        ds-it-rt-defs
        vec-rt-defs
        join-ht-rt-defs
        ht-rt-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDSA
        MLIRSCFTransforms
        )
