add_mlir_dialect_library(MLIRTupleStream
        TupleStreamDialect.cpp
        TupleStreamOps.cpp
        ColumnManager.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/TupleStream

        DEPENDS
        MLIRTupleStreamOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        )
