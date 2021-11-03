add_mlir_conversion_library(MLIRUtilToLLVM
        LowerToLLVM.cpp
        UtilTypeConversion.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/UtilToLLVM

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRUtil
        )
