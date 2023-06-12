#mlir-to-sql
add_executable(mlir-to-sql mlir-to-sql.cpp)
llvm_update_compile_flags(mlir-to-sql)
target_link_libraries(mlir-to-sql PUBLIC runtime utility ${LIBS}  PRIVATE arrow_static)
set_target_properties(mlir-to-sql PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( mlir-to-sql PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)

#mlir-to-json
add_executable(mlir-to-json mlir-to-json.cpp)
llvm_update_compile_flags(mlir-to-json)
target_link_libraries(mlir-to-json  PUBLIC runtime utility runner mlir-support ${LIBS} PRIVATE arrow_static)
target_link_options(mlir-to-json PUBLIC -Wl,--export-dynamic)
set_target_properties(mlir-to-json PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( mlir-to-json PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)


# sql-to-mlir
add_executable(sql-to-mlir sql-to-mlir.cpp)
llvm_update_compile_flags(sql-to-mlir)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend pg_query::pg_query runtime utility PRIVATE arrow_static)
set_target_properties(sql-to-mlir PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( sql-to-mlir PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)


# run-mlir
add_executable(run-mlir run-mlir.cpp)
target_link_libraries(run-mlir PUBLIC runner runtime utility mlir-support  PRIVATE arrow_static)
target_link_options(run-mlir PUBLIC -Wl,--export-dynamic)
set_target_properties(run-mlir PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( run-mlir PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)


# run-sql
add_executable(run-sql run-sql.cpp)
target_link_libraries(run-sql PUBLIC runner runtime utility mlir-support  PRIVATE arrow_static)
target_link_options(run-sql PUBLIC -Wl,--export-dynamic)
set_target_properties(run-sql PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( run-sql PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)

add_executable(run-sql-batched run-sql-batched.cpp)
target_link_libraries(run-sql-batched PUBLIC runner runtime utility mlir-support PRIVATE arrow_static)
target_link_options(run-sql-batched PUBLIC -Wl,--export-dynamic)
set_target_properties(run-sql-batched PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( run-sql-batched PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)

# mlir-db-opt
add_executable(mlir-db-opt mlir-opt.cpp)
llvm_update_compile_flags(mlir-db-opt)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} runner runtime utility mlir-support  PRIVATE arrow_static)
set_target_properties(mlir-db-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
