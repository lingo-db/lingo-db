add_library(GPU-rt INTERFACE)

if (ENABLE_GPU_BACKEND)
    add_subdirectory(CUDA)
    target_link_libraries(GPU-rt INTERFACE CUDA-lingodb-rt)
endif(ENABLE_GPU_BACKEND)
