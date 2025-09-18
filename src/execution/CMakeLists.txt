set(EXECUTION_FILES ResultProcessor.cpp Frontend.cpp LLVMBackends.cpp Execution.cpp BackendPasses.cpp CEmitter.cpp CBackend.cpp BareFunctions.cpp Instrumentation.cpp DecomposeTuplePass.cpp)

if (ENABLE_GPU_BACKEND)
    find_library(CUDA_RUNTIME_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

    add_mlir_library(mlir_cuda_runtime
            SHARED
            ${LLVM_BINARY_DIR}/src/CudaRuntimeWrappers.cpp
            EXCLUDE_FROM_LIBMLIR
            PARTIAL_SOURCES_INTENDED
    )
    target_include_directories(mlir_cuda_runtime
            PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(mlir_cuda_runtime
            PRIVATE
            ${CUDA_RUNTIME_LIBRARY}
    )
    get_target_property(CUDA_RT_OUTPUT_DIR mlir_cuda_runtime LIBRARY_OUTPUT_DIRECTORY)
endif (ENABLE_GPU_BACKEND)

add_library(runner ${EXECUTION_FILES})
llvm_update_compile_flags(runner)
target_compile_options(runner PRIVATE -fexceptions)
target_link_libraries(runner PRIVATE ${LIBS} MLIRSQLFrontend utility)
set(COMPILE_DEFS "SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
if (ENABLE_BASELINE_BACKEND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/baseline ${CMAKE_CURRENT_BINARY_DIR}/baseline)
    list(APPEND COMPILE_DEFS " BASELINE_ENABLED=1")
else ()
    list(APPEND COMPILE_DEFS " BASELINE_ENABLED=0")
endif (ENABLE_BASELINE_BACKEND)

if (ENABLE_GPU_BACKEND)
    list(APPEND COMPILE_DEFS " GPU_ENABLED=1")
    list(APPEND COMPILE_DEFS " CUDA_RUNTIME_LIB=\"${CUDA_RT_OUTPUT_DIR}/libmlir_cuda_runtime.so\"")
    add_dependencies(runner mlir_cuda_runtime)
else ()
    list(APPEND COMPILE_DEFS " GPU_ENABLED=0")
endif (ENABLE_GPU_BACKEND)
if (COMPILE_FOR_PERF)
    list(APPEND COMPILE_DEFS " RUN_QUERIES_WITH_PERF=1")
else ()
    list(APPEND COMPILE_DEFS " RUN_QUERIES_WITH_PERF=0")
endif (COMPILE_FOR_PERF)
message("compile defs: ${COMPILE_DEFS}")
target_compile_definitions(runner PRIVATE ${COMPILE_DEFS})
