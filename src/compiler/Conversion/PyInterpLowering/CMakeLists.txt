

add_mlir_conversion_library(MLIRPyInterpToStd
        PyInterpLowering.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../lingodb/compiler/Conversion/PyInterpLowering

        DEPENDS
        py-rt-defs
        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRArrow
        MLIRSCFTransforms
        runtime_funcs_ptr
)
