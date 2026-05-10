add_mlir_dialect_library(MLIRSubOperator
        SubOperatorDialect.cpp
        SubOperatorOps.cpp
        SubOperatorTypes.cpp
        SubOperatorAttributes.cpp
        SubOpInterfaces.cpp
        Transforms/SubOpStateUsageTransformer.cpp
        Transforms/ReuseLocalPass.cpp
        Transforms/SpecializeSubOpPass.cpp
        Transforms/FoldColumnsPass.cpp
        Transforms/Passes.cpp
        Transforms/ColumnUsageAnalysis.cpp
        Transforms/ColumnCreationAnalysis.cpp
        Transforms/PullGatherUpPass.cpp
        Transforms/NormalizeSubOpPass.cpp
        Transforms/ParallelizePass.cpp
        Transforms/SpecializeParallelPass.cpp
        Transforms/SplitIntoNestedExecutionStepsPass.cpp
        Transforms/OrganizeExecutionStepsPass.cpp
        Transforms/StepGraphUtils.cpp
        Transforms/InlineNestedMapPass.cpp
        Transforms/PrepareForLowering.cpp
        Transforms/CommonPipelineElimination.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/SubOperator

        DEPENDS
        MLIRSubOperatorOpsIncGen
        MLIRArrowOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRArrow
        MLIRCompilerHelper
        runtime_funcs_ptr
)
