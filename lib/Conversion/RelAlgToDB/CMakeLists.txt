add_mlir_conversion_library(MLIRRelAlgToDB
        Pipeline.cpp
        Translator.cpp
        JoinTranslator.cpp
        NLJoinTranslator.cpp
        HashJoinTranslator.cpp
        LowerToDB.cpp
        Translators/BaseTableOp.cpp
        Translators/AggregationOp.cpp
        Translators/ConstRelationOp.cpp
        Translators/CrossProductOp.cpp
        Translators/MapOp.cpp
        Translators/MaterializeOp.cpp
        Translators/SelectionOp.cpp
        Translators/SortOp.cpp
        Translators/InnerJoinOp.cpp
        Translators/OuterJoinOp.cpp
        Translators/SingleJoinOp.cpp
        Translators/MarkJoinOp.cpp
        Translators/CollectionJoinOp.cpp
        Translators/SemiJoinOp.cpp
        Translators/AntiSemiJoinOp.cpp
        Translators/RenamingOp.cpp
        Translators/ProjectionOp.cpp
        Translators/LimitOp.cpp
        Translators/TmpOp.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../mlir/Conversion/RelAlgToDB

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        MLIRDB
        )
