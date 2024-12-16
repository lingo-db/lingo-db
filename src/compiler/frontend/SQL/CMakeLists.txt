
add_library(MLIRSQLFrontend
        Parser.cpp
)
add_dependencies(MLIRSQLFrontend ec-rt-defs)
target_link_libraries(MLIRSQLFrontend PUBLIC ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        runtime_funcs_ptr)