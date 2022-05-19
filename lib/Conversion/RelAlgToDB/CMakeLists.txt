add_mlir_conversion_library(MLIRRelAlgToDB
        Translator.cpp
        JoinTranslator.cpp
        NLJoinTranslator.cpp
        HashJoinTranslator.cpp
        LowerToDB.cpp
        Translators/BaseTableOp.cpp
        Translators/AggregationOp.cpp
        Translators/ConstRelationOp.cpp
        Translators/MapOp.cpp
        Translators/MaterializeOp.cpp
        Translators/SelectionOp.cpp
        Translators/SortOp.cpp
        Translators/Joins.cpp
        Translators/SetOps.cpp
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
