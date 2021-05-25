add_mlir_conversion_library(MLIRRelAlgToDB
        LowerToDB.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/RelAlgToDB

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDB
        )
