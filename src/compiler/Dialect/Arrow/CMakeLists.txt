
add_mlir_dialect_library(MLIRArrow
        ArrowDialect.cpp
        ArrowOps.cpp
        ArrowTypes.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/Arrow

        DEPENDS
        MLIRArrowOpsIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncTransforms
        MLIRUtil
        runtime_funcs_ptr
)
