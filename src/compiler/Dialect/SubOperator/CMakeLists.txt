add_mlir_dialect_library(MLIRSubOperator
        SubOperatorDialect.cpp
        SubOperatorOps.cpp
        SubOperatorTypes.cpp
        SubOperatorAttributes.cpp
        SubOpInterfaces.cpp
        Transforms/GlobalOpt.cpp
        Transforms/SubOpDependencyAnalysis.cpp
        Transforms/SubOpStateUsageTransformer.cpp
        Transforms/ReuseLocalPass.cpp
        Transforms/SpecializeSubOpPass.cpp
        Transforms/FoldColumnsPass.cpp
        Transforms/Passes.cpp
        Transforms/EnforceOrder.cpp
        Transforms/ColumnUsageAnalysis.cpp
        Transforms/ColumnCreationAnalysis.cpp
        Transforms/PullGatherUpPass.cpp
        Transforms/NormalizeSubOpPass.cpp
        Transforms/ParallelizePass.cpp
        Transforms/SpecializeParallelPass.cpp
        Transforms/SplitIntoExecutionSteps.cpp
        Transforms/InlineNestedMapPass.cpp
        Transforms/FinalizePass.cpp
        Transforms/PrepareForLowering.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/SubOperator

        DEPENDS
        MLIRSubOperatorOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        runtime_funcs_ptr
)
