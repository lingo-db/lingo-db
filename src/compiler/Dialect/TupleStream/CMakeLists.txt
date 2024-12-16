add_mlir_dialect_library(MLIRTupleStream
        TupleStreamDialect.cpp
        TupleStreamOps.cpp
        ColumnManager.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/TupleStream

        DEPENDS
        MLIRTupleStreamOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
)
