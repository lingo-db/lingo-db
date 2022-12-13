add_mlir_conversion_library(MLIRFuncToCranelift
        LowerToCranelift.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/CraneliftConversions/

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRCranelift
        )
