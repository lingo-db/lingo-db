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

# Add WAMR
# https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/doc/build_wamr.md#wamr-vmcore-cmake-building-configurations
if (APPLE)
    set (WAMR_BUILD_PLATFORM "darwin")
    set (WAMR_BUILD_TARGET "AARCH64")
else ()
    set (WAMR_BUILD_PLATFORM "linux")
    set (WAMR_BUILD_TARGET "X86_64")
endif ()
set (WAMR_BUILD_INTERP 0)
set (WAMR_BUILD_FAST_INTERP 0)
set (WAMR_BUILD_AOT 1)
set (WAMR_BUILD_LIBC_BUILTIN 1)
set (WAMR_BUILD_LIBC_WASI 1)
set (WAMR_BUILD_SIMD 1)
set (WAMR_BUILD_REF_TYPES 1)
set (WAMR_ROOT_DIR ${PROJECT_SOURCE_DIR}/vendored/wasm-micro-runtime)
include (${WAMR_ROOT_DIR}/build-scripts/runtime_lib.cmake)
add_library(vmlib ${WAMR_RUNTIME_LIB_SOURCE})
include (${SHARED_DIR}/utils/uncommon/shared_uncommon.cmake)
# /Add WAMR

# Check if python.wasm exists
if (NOT DEFINED PYTHON_BUILD_DIR)
    set(PYTHON_BUILD_DIR "${CMAKE_SOURCE_DIR}/build/cpython-wasm/Python-3.14.0/cross-build/wasm32-wasip1" CACHE PATH "Path to the Python build directory")
    message(WARNING "PYTHON_BUILD_DIR was not set. Defaulting to ${PYTHON_BUILD_DIR}")
endif()

if (NOT EXISTS "${PYTHON_BUILD_DIR}/python.wasm")
    message(FATAL_ERROR "${PYTHON_BUILD_DIR}/python.wasm does not exist. Run lingo-db/tools/scripts/build_cpython_wasm.sh before cmake configure.")
endif()

## Compile python.wasm to python.aot using wamrc
# Define the build directory for wamr-compiler
set(WAMR_COMPILER_DIR "${PROJECT_SOURCE_DIR}/vendored/wasm-micro-runtime/wamr-compiler")
set(WAMR_COMPILER_BUILD_DIR "${WAMR_COMPILER_DIR}/build")
set(WAMR_COMPILER_EXECUTABLE "${WAMR_COMPILER_BUILD_DIR}/wamrc")

# Add custom command to build wamr-compiler
add_custom_command(
        OUTPUT "${WAMR_COMPILER_EXECUTABLE}"
        COMMAND ${CMAKE_COMMAND} -S "${WAMR_COMPILER_DIR}" -B "${WAMR_COMPILER_BUILD_DIR}"
        COMMAND make -C "${WAMR_COMPILER_BUILD_DIR}" -j4
        COMMENT "Building wamr-compiler using cmake and make"
)

# Add custom command to generate python.aot from python.wasm
add_custom_command(
        OUTPUT "${PYTHON_BUILD_DIR}/python.aot"
        COMMAND ${WAMR_COMPILER_EXECUTABLE} -o "${PYTHON_BUILD_DIR}/python.aot" "${PYTHON_BUILD_DIR}/python.wasm"
        DEPENDS "${PYTHON_BUILD_DIR}/python.wasm" "${WAMR_COMPILER_EXECUTABLE}"
        COMMENT "Generating python.aot from python.wasm using wamrc"
)

# Create the cpython-wasm-test executable
add_executable(cpython-wasm-test cpython-wasm-test.cpp ${UNCOMMON_SHARED_SOURCE})
target_link_libraries(cpython-wasm-test PUBLIC vmlib -lm)
set_target_properties(cpython-wasm-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Our runtime code depends on reading this file, make sure it exists
add_custom_target(
        GeneratePythonAOT ALL
        DEPENDS "${PYTHON_BUILD_DIR}/python.aot"
)
add_dependencies(cpython-wasm-test GeneratePythonAOT)
