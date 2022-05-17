add_mlir_dialect_library(MLIRRelAlg
        RelAlgDialect.cpp
        RelAlgOps.cpp
        RelAlgOpsEnums.cpp
        OperatorInterfaceImpl.cpp
        Transforms/MetaData.cpp
        Transforms/ExtractNestedOperators.cpp
        Transforms/SimplifyAggregations.cpp
        Transforms/DecomposeLambdas.cpp
        Transforms/CombinePredicates.cpp
        Transforms/OptimizeImplementations.cpp
        Transforms/PropagateConstraints.cpp
        Transforms/IntroduceTmp.cpp
        ColumnManager.cpp
        Transforms/Unnesting.cpp
        Transforms/Pushdown.cpp
        Transforms/ImplicitToExplicitJoins.cpp
        Transforms/OptimizeJoinOrder.cpp
        Transforms/queryopt/utils.cpp
        Transforms/queryopt/QueryGraphBuilder.cpp
        Transforms/queryopt/DPhyp.cpp
        Transforms/queryopt/GOO.cpp
        Transforms/queryopt/QueryGraph.cpp
        Passes.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/RelAlg

        DEPENDS
        MLIRRelAlgOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        )
