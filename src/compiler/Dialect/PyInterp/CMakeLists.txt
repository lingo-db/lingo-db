
add_mlir_dialect_library(MLIRPyInterp
        PyInterpDialect.cpp
        PyInterpOps.cpp
        PyInterpTypes.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/PyInterp

        DEPENDS
        MLIRPyInterpOpsIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncTransforms
)
