# Do NOT use object library for CUDA
add_library(GPU STATIC
        DeviceMemoryManager.cpp
        DeviceRTWrapper.cpp
)
target_compile_definitions(GPU PUBLIC ${COMPILE_DEFS} )
add_library(GPU-rt INTERFACE)
target_link_libraries(GPU-rt INTERFACE GPU)

if (ENABLE_GPU_BACKEND)
    add_subdirectory(CUDA)
    target_link_libraries(GPU-rt INTERFACE CUDA-lingodb-rt)
    add_dependencies(GPU-rt rt_gpu_bc)
endif(ENABLE_GPU_BACKEND)
