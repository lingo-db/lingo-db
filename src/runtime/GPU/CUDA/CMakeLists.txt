find_package(CUDAToolkit REQUIRED)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

list(GET INSTALLED_GPU_CCS_3 0 FIRST_CUDA_ARCH) # we can have many GPUs, but we do not support multi-gpu, pick one.
string(REPLACE "." "" CUDA_ARCH "${FIRST_CUDA_ARCH}")

# Exposed to C++
add_library(CUDA-lingodb-rt STATIC

        Properties.cu
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

