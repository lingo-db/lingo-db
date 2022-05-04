#mlir-to-sql
add_executable(mlir-to-sql mlir-to-sql.cpp)
llvm_update_compile_flags(mlir-to-sql)
target_link_libraries(mlir-to-sql PUBLIC runtime ${LIBS})
set_target_properties(mlir-to-sql PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# sql-to-mlir
add_executable(sql-to-mlir sql-to-mlir.cpp)
llvm_update_compile_flags(sql-to-mlir)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend pg_query::pg_query runtime)
set_target_properties(sql-to-mlir PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# run-mlir
add_executable(run-mlir run-mlir.cpp)
target_link_libraries(run-mlir runner runtime mlir-support)
target_link_options(run-mlir PUBLIC -Wl,--export-dynamic)
set_target_properties(run-mlir PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# run-sql
add_executable(run-sql run-sql.cpp)
target_link_libraries(run-sql runner runtime mlir-support)
target_link_options(run-sql PUBLIC -Wl,--export-dynamic)
set_target_properties(run-sql PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


# mlir-db-opt
add_executable(mlir-db-opt mlir-opt.cpp)
llvm_update_compile_flags(mlir-db-opt)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} runtime mlir-support PRIVATE arrow_shared)
set_target_properties(mlir-db-opt PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
