add_mlir_dialect_library(MLIRGraphAlg
        # IR
        GraphAlgAttr.cpp
        GraphAlgCanonicalize.cpp
        GraphAlgCast.cpp
        GraphAlgDialect.cpp
        GraphAlgInterfaces.cpp
        GraphAlgOps.cpp
        GraphAlgInferDensity.cpp
        GraphAlgTypes.cpp
        SemiringTypes.cpp
        # Passes
        GraphAlgExplicateSparsity.cpp
        GraphAlgLoopAggregate.cpp
        GraphAlgPrepareInlinePass.cpp
        GraphAlgScalarizeApply.cpp
        GraphAlgSetDimensions.cpp
        GraphAlgSplitAggregate.cpp
        GraphAlgToCore.cpp
        GraphAlgToCorePipeline.cpp
        GraphAlgVerifyDimensions.cpp

        ADDITIONAL_HEADER_DIRS
        ../../include/graphalg/Dialect/GraphAlg

        DEPENDS
        MLIRGraphAlgOpsIncGen
        MLIRGraphAlgPassesIncGen
        MLIRGraphAlgInterfacesIncGen

        LINK_LIBS PUBLIC
        GraphAlgAnalysis
        MLIRFuncDialect
        MLIRFuncInlinerExtension
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRSupport
        MLIRTransforms
)

# Apply specific flags to the target created by the macro
target_compile_options(MLIRGraphAlg PRIVATE -Wno-dangling-assignment-gsl)
if(NOT GRAPHALG_ENABLE_RTTI)
  target_compile_options(MLIRGraphAlg PUBLIC -fno-rtti)
endif()

add_subdirectory(analysis)
add_subdirectory(evaluate)
add_subdirectory(parse)
