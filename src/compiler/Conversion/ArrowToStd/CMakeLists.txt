

add_mlir_conversion_library(MLIRArrowToStd
        ArrowToStd.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../lingodb/compiler/Conversion/ArrowToStd

        DEPENDS
        tb-arrow-column-defs
        tb-arrow-table-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRArrow
        MLIRSCFTransforms
        runtime_funcs_ptr
)
