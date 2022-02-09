add_mlir_library(MLIRCustomTransforms
        SinkOp.cpp




        LINK_LIBS PUBLIC
        MLIRAnalysis
        MLIRCopyOpInterface
        MLIRLoopLikeInterface
        MLIRPass
        MLIRSupport
        MLIRTransformUtils
        )