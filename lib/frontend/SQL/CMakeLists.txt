add_mlir_library(MLIRSQLFrontend
        Parser.cpp
        DEPENDS

        LINK_LIBS PUBLIC
        ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        )