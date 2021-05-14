add_mlir_conversion_library(MLIRDBToArrowStd
        CollectionIterators.cpp
        LowerToStd.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/DBToArrowStd

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDB
        )
