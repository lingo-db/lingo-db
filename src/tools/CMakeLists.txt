set(COMPILE_DEFS "")
if (ENABLE_BASELINE_BACKEND)
    list(APPEND COMPILE_DEFS " BASELINE_ENABLED=1")
else ()
    list(APPEND COMPILE_DEFS " BASELINE_ENABLED=0")
endif (ENABLE_BASELINE_BACKEND)

if (ENABLE_GPU_BACKEND)
    list(APPEND COMPILE_DEFS " GPU_ENABLED=1")
else ()
    list(APPEND COMPILE_DEFS " GPU_ENABLED=0")
endif (ENABLE_GPU_BACKEND)

add_executable(sql sql.cpp)
target_link_libraries(sql runner utility runtime mlir-support scheduler linenoise-ng)
set_target_properties(sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sql PRIVATE ${COMPILE_DEFS})

add_executable(sqlite-tester sqlite-tester.cpp)
target_link_libraries(sqlite-tester runner runtime utility mlir-support scheduler)
set_target_properties(sqlite-tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sqlite-tester PRIVATE ${COMPILE_DEFS})

# mlir-db-opt
add_executable(mlir-db-opt mlir-opt.cpp)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} runner runtime utility mlir-support scheduler)
set_target_properties(mlir-db-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(mlir-db-opt PRIVATE ${COMPILE_DEFS})

# sql-to-mlir
add_executable(sql-to-mlir sql-to-mlir.cpp)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend runtime utility runner scheduler)
set_target_properties(sql-to-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sql-to-mlir PRIVATE ${COMPILE_DEFS})

# run-mlir
add_executable(run-mlir run-mlir.cpp)
target_link_libraries(run-mlir PUBLIC runner runtime utility mlir-support scheduler)
set_target_properties(run-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(run-mlir PRIVATE ${COMPILE_DEFS})

# run-sql
add_executable(run-sql run-sql.cpp)
target_link_libraries(run-sql PUBLIC runner runtime utility mlir-support scheduler)
set_target_properties(run-sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(run-sql PRIVATE ${COMPILE_DEFS})

# cpython-wasm-test
add_executable(cpython-wasm-test cpython-wasm-test.cpp ${UNCOMMON_SHARED_SOURCE})
target_link_libraries(cpython-wasm-test PUBLIC cpython-wasm)
