add_mlir_conversion_library(MLIRDBToArrowStd
        CollectionIterators.cpp
        LowerToStd.cpp
        FunctionRegistry.cpp
        ScalarDBToStdPatterns.cpp
        RuntimeSpecificScalarToStdPatterns.cpp
        BuildersToStdPatterns.cpp
        CollectionsToStdPatterns.cpp
        ArrowTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/DBToArrowStd

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDB
        MLIRSCFTransforms
        )
