add_executable(sql sql.cpp)
target_link_libraries(sql runner utility runtime mlir-support scheduler linenoise-ng)
set_target_properties(sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(sqlite-tester sqlite-tester.cpp)
target_link_libraries(sqlite-tester runner runtime utility mlir-support scheduler)
set_target_properties(sqlite-tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# mlir-db-opt
add_executable(mlir-db-opt mlir-opt.cpp)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} runner runtime utility mlir-support scheduler)
set_target_properties(mlir-db-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# sql-to-mlir
add_executable(sql-to-mlir sql-to-mlir.cpp)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend runtime utility runner scheduler)
set_target_properties(sql-to-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# run-mlir
add_executable(run-mlir run-mlir.cpp)
target_link_libraries(run-mlir PUBLIC runner runtime utility mlir-support scheduler)
set_target_properties(run-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# run-sql
add_executable(run-sql run-sql.cpp)
target_link_libraries(run-sql PUBLIC runner runtime utility mlir-support scheduler)
set_target_properties(run-sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


