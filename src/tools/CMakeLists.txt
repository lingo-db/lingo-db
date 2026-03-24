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
if (ENABLE_MIMALLOC)
    list(APPEND COMPILE_DEFS " MI_MALLOC_OVERRIDE=1")
endif ()
# Common libraries used by most tools
set(COMMON_LIBS runner utility runtime mlir-support scheduler)


if (ENABLE_MIMALLOC)
    list(APPEND COMMON_LIBS mimalloc-static)
endif ()
add_executable(sql sql.cpp)
# sql needs linenoise-ng in addition to the common libs
target_link_libraries(sql PRIVATE ${COMMON_LIBS} linenoise-ng)
set_target_properties(sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sql PRIVATE ${COMPILE_DEFS})


add_executable(sqlite-tester sqlite-tester.cpp)
target_link_libraries(sqlite-tester PRIVATE ${COMMON_LIBS})
set_target_properties(sqlite-tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sqlite-tester PRIVATE ${COMPILE_DEFS})

# mlir-db-opt
add_executable(mlir-db-opt mlir-opt.cpp)
# mlir-db-opt also needs the ${LIBS} variable
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} ${COMMON_LIBS})
set_target_properties(mlir-db-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(mlir-db-opt PRIVATE ${COMPILE_DEFS})

# sql-to-mlir
add_executable(sql-to-mlir sql-to-mlir.cpp)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend ${COMMON_LIBS} runner scheduler)
set_target_properties(sql-to-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sql-to-mlir PRIVATE ${COMPILE_DEFS})

# run-mlir
add_executable(run-mlir run-mlir.cpp)
target_link_libraries(run-mlir PUBLIC ${COMMON_LIBS})
set_target_properties(run-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(run-mlir PRIVATE ${COMPILE_DEFS})


# run-sql
add_executable(run-sql run-sql.cpp)
target_link_libraries(run-sql PUBLIC ${COMMON_LIBS})
set_target_properties(run-sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(run-sql PRIVATE ${COMPILE_DEFS})
