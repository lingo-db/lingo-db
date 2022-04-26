gen_rt_def(db-rt-defs "Database.h")
gen_rt_def(ec-rt-defs "ExecutionContext.h")

add_mlir_library(MLIRSQLFrontend
        Parser.cpp
        DEPENDS
        db-rt-defs
        ec-rt-defs
        LINK_LIBS PUBLIC
        ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        )