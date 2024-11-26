
#mlir-to-json
add_executable(mlir-to-json mlir-relalg-to-json.cpp)
llvm_update_compile_flags(mlir-to-json)
target_link_libraries(mlir-to-json  PUBLIC runtime utility runner mlir-support ${LIBS} PRIVATE arrow)
set_target_properties(mlir-to-json PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( mlir-to-json PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)



add_executable(mlir-subop-to-json mlir-subop-to-json.cpp)
llvm_update_compile_flags(mlir-subop-to-json)
target_link_libraries(mlir-subop-to-json PUBLIC runtime utility runner mlir-support ${LIBS} PRIVATE arrow)
set_target_properties(mlir-subop-to-json PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories(mlir-subop-to-json PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)

# sql-to-mlir
add_executable(sql-to-mlir sql-to-mlir.cpp)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend pg_query::pg_query runtime utility runner PRIVATE arrow)
set_target_properties(sql-to-mlir PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( sql-to-mlir PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)

# run-mlir
add_executable(run-mlir run-mlir.cpp)
target_link_libraries(run-mlir PUBLIC runner runtime utility mlir-support  PRIVATE arrow)
set_target_properties(run-mlir PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( run-mlir PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)


# run-sql
add_executable(run-sql run-sql.cpp)
target_link_libraries(run-sql PUBLIC runner runtime utility mlir-support  PRIVATE arrow)
set_target_properties(run-sql PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( run-sql PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)


# mlir-db-opt
add_executable(mlir-db-opt mlir-opt.cpp)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} runner runtime utility mlir-support  PRIVATE arrow)
set_target_properties(mlir-db-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


#mlir-analyze-snapshots
add_executable(mlir-analyze-snapshots analyze-snapshots.cpp)
llvm_update_compile_flags(mlir-analyze-snapshots)
target_link_libraries(mlir-analyze-snapshots  PUBLIC runtime utility runner mlir-support ${LIBS} PRIVATE arrow)
set_target_properties(mlir-analyze-snapshots PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_directories( mlir-analyze-snapshots PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)

