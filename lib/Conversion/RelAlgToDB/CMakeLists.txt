add_mlir_conversion_library(MLIRRelAlgToDB
        LowerToDB.cpp
        BaseTableOp.cpp
        AggregationOp.cpp
        ConstRelationOp.cpp
        CrossProductOp.cpp
        MapOp.cpp
        MaterializeOp.cpp
        SelectionOp.cpp
        SortOp.cpp
        InnerJoinOp.cpp
        OuterJoinOp.cpp
        SingleJoinOp.cpp
        MarkJoinOp.cpp
        SemiJoinOp.cpp
        AntiSemiJoinOp.cpp
        RenamingOp.cpp
        ProjectionOp.cpp
        LimitOp.cpp
        TmpOp.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/RelAlgToDB

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDB
        )
