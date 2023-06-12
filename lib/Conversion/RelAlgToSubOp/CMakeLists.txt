add_mlir_conversion_library(MLIRelAlgToSubOp
        RelAlgToSubOp.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/RelAlgToSubOp

        DEPENDS
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRRelAlg
        MLIRSubOperator
        runtime_funcs_ptr
        )
