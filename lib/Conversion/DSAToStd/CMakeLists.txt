add_mlir_conversion_library(MLIRDSAToStd
        CollectionIterators.cpp
        LowerToStd.cpp
        FunctionRegistry.cpp
        ScalarDSAToStdPatterns.cpp
        RuntimeSpecificScalarToStdPatterns.cpp
        DSAToStdPatterns.cpp
        CollectionsToStdPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/DSAToStd

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDSA
        MLIRSCFTransforms
        )
