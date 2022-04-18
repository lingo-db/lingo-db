add_executable(sql-to-mlir sql-to-mlir.cpp)
llvm_update_compile_flags(sql-to-mlir)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend pg_query::pg_query runtime)