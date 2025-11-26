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

# If CPYTHON is enabled, add Python to the common libs and expose link dir
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    list(APPEND COMMON_LIBS python3.14)
    set(PY_LINK_DIR ${PY_HOME}/lib)
endif()
if (ENABLE_MIMALLOC)
    list(APPEND COMMON_LIBS mimalloc-static)
endif ()
add_executable(sql sql.cpp)
# sql needs linenoise-ng in addition to the common libs
target_link_libraries(sql PRIVATE ${COMMON_LIBS} linenoise-ng)
set_target_properties(sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sql PRIVATE ${COMPILE_DEFS})
# If CPYTHON is enabled, ensure the python link dir and compile def are provided
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(sql PRIVATE ${PY_LINK_DIR})
    target_compile_definitions(sql PRIVATE USE_CPYTHON_RUNTIME=1)
endif()

add_executable(sqlite-tester sqlite-tester.cpp)
target_link_libraries(sqlite-tester PRIVATE ${COMMON_LIBS})
set_target_properties(sqlite-tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sqlite-tester PRIVATE ${COMPILE_DEFS})
# If CPYTHON is enabled, ensure the python link dir and compile def are provided
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(sqlite-tester PRIVATE ${PY_LINK_DIR})
    target_compile_definitions(sqlite-tester PRIVATE USE_CPYTHON_RUNTIME=1)
endif()
# mlir-db-opt
add_executable(mlir-db-opt mlir-opt.cpp)
# mlir-db-opt also needs the ${LIBS} variable
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} ${COMMON_LIBS})
set_target_properties(mlir-db-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(mlir-db-opt PRIVATE ${COMPILE_DEFS})
# If CPYTHON is enabled, ensure the python link dir and compile def are provided
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(mlir-db-opt PRIVATE ${PY_LINK_DIR})
    target_compile_definitions(mlir-db-opt PRIVATE USE_CPYTHON_RUNTIME=1)
endif()
# sql-to-mlir
add_executable(sql-to-mlir sql-to-mlir.cpp)
target_link_libraries(sql-to-mlir PUBLIC ${LIBS} MLIRSQLFrontend ${COMMON_LIBS} runner scheduler)
set_target_properties(sql-to-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(sql-to-mlir PRIVATE ${COMPILE_DEFS})
# If CPYTHON is enabled, ensure the python link dir and compile def are provided
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(sql-to-mlir PRIVATE ${PY_LINK_DIR})
    target_compile_definitions(sql-to-mlir PRIVATE USE_CPYTHON_RUNTIME=1)
endif()
# run-mlir
add_executable(run-mlir run-mlir.cpp)
target_link_libraries(run-mlir PUBLIC ${COMMON_LIBS})
set_target_properties(run-mlir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(run-mlir PRIVATE ${COMPILE_DEFS})
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(run-mlir PRIVATE ${PY_LINK_DIR})
    target_compile_definitions(run-mlir PRIVATE USE_CPYTHON_RUNTIME=1)
endif()

# run-sql
add_executable(run-sql run-sql.cpp)
target_link_libraries(run-sql PUBLIC ${COMMON_LIBS})
set_target_properties(run-sql PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_definitions(run-sql PRIVATE ${COMPILE_DEFS})
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(run-sql PRIVATE ${PY_LINK_DIR})
    target_compile_definitions(run-sql PRIVATE USE_CPYTHON_RUNTIME=1)
endif()

if(ENABLE_PYTHON STREQUAL "WASM")
# cpython-wasm-test
add_executable(cpython-wasm-test cpython-wasm-test.cpp ${UNCOMMON_SHARED_SOURCE})
target_link_libraries(cpython-wasm-test PUBLIC cpython-wasm)
endif()
