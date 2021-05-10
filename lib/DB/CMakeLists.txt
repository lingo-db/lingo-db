add_mlir_dialect_library(MLIRDB
        DBDialect.cpp
        DBOps.cpp
        DBOpsEnums.cpp
        DBTypes.cpp
        LowerToStd.cpp
        CollectionIterators.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/DB

        DEPENDS
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        )
