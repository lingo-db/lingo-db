#ifndef LINGODB_RUNTIME_GPU_CUDA_DEFINITIONHELPER_CUH
#define LINGODB_RUNTIME_GPU_CUDA_DEFINITIONHELPER_CUH

/*
    We have to include some CUDA headers (.cuh) in C++ code 
    for things like get class size. This means that
    all decorators as well as CUDA-specific code
    should be hidden from what is included in "normal" C++.
    
    __CUDACC__ is automatically defined for .cu files.
*/
#ifdef __CUDACC__
#define CUDA_HOST __host__ 
#define CUDA_DEVICE __device__ 
#define CUDA_HOST_DEVICE CUDA_HOST CUDA_DEVICE
#define CUDA_KERNEL __global__ 
#else
#define CUDA_HOST  
#define CUDA_DEVICE  
#define CUDA_HOST_DEVICE
#define CUDA_KERNEL 
#endif

#endif // LINGODB_RUNTIME_GPU_CUDA_DEFINITIONHELPER_CUH
