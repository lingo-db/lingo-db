add_mlir_conversion_library(MLIGraphAlgToRelAlg
        GraphAlgCoreToRelAlg.cpp
        GraphAlgToGraphAlgCore.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../lingodb/compiler/Conversion/GraphAlgToRelAlg

        DEPENDS
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRRelAlg
        MLIRGraphAlg
        runtime_funcs_ptr
)
