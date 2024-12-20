find_package(CUDAToolkit REQUIRED)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

list(GET INSTALLED_GPU_CCS_3 0 FIRST_CUDA_ARCH) # we can have many GPUs, but we do not support multi-gpu, pick one.
string(REPLACE "." "" CUDA_ARCH "${FIRST_CUDA_ARCH}")
string(REPLACE "+PTX" "" CUDA_ARCH "${CUDA_ARCH}")

# Exposed to C++
add_library(CUDA-lingodb-rt STATIC
        MemoryPool.cu
        Vector.cu
        FlexibleBuffer.cu
        LazyJoinHashtable.cu
        PreAggregationHashTable.cu
        Properties.cu
)

# Exposed to MLIR
set(CUDA_DEVICE_RT
        Vector.cu
        FlexibleBuffer.cu
        LazyJoinHashtable.cu
        PreAggregationHashTable.cu
        RTDeviceFuncsIface.cu
)


set_target_properties(CUDA-lingodb-rt PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(CUDA-lingodb-rt PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_compile_features(CUDA-lingodb-rt PUBLIC cuda_std_17)
target_compile_options(CUDA-lingodb-rt PRIVATE -O3 --ptxas-options=-v -Wall)
target_link_libraries(CUDA-lingodb-rt PRIVATE CUDA::cudart CUDA::cuda_driver)

# Problem:
# `GPU arch sm_75 is supported by CUDA versions between 10.0 and 10.1 (inclusive), but installation at /usr/local/cuda is 7.0.#
#   Use --cuda-path to specify a different CUDA install, pass a different GPU arch with --cuda-gpu-arch, or pass --no-cuda-version-check.`
# Solution: Check that you use latest clang version. For example, CUDA 12.0 works with clang-16
foreach(CUDA_FILE ${CUDA_DEVICE_RT})
    string(REPLACE ".cu" ".bc" BITCODE_FILE "${CUDA_FILE}")
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${BITCODE_FILE}
            COMMAND clang++
            ARGS
            -I ../../../../../../include
            --cuda-device-only
            --cuda-gpu-arch=sm_${CUDA_ARCH}
            -emit-llvm
            -std=c++17
            -Ofast
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_FILE}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${BITCODE_FILE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_FILE}
    )
    list(APPEND BITCODE_FILES ${CMAKE_CURRENT_BINARY_DIR}/${BITCODE_FILE})
endforeach()

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RTDeviceFuncs.bc
        COMMAND llvm-link ${BITCODE_FILES} -o ${CMAKE_CURRENT_BINARY_DIR}/RTDeviceFuncs.bc
        DEPENDS ${BITCODE_FILES}
        VERBATIM
)

add_custom_target(RTDeviceFuncsBc ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/RTDeviceFuncs.bc
)


add_custom_target(rt_gpu_bc)
add_dependencies(rt_gpu_bc RTDeviceFuncsBc)