add_mlir_library(MLIRCustomTransforms
        SinkOp.cpp
        SimplifyMemrefs.cpp



        LINK_LIBS PUBLIC
        MLIRAnalysis
        MLIRCopyOpInterface
        MLIRLoopLikeInterface
        MLIRPass
        MLIRSupport
        MLIRTransformUtils
        )