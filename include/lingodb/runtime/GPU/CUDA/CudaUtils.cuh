#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H
#include<iostream>

#define CHECK_CUDA_ERROR_ALLOC(err) \
    if (err != cudaSuccess) { \
        if (err == cudaErrorMemoryAllocation) { \
            throw std::bad_alloc(); \
        } \
        std::cout << err << " != " << cudaErrorMemoryAllocation << "\n"; \
        std::cout << "CUDA ALLOC Error " << (int64_t)err << ": " << cudaGetErrorString(err) << "  occurred at line: " << __LINE__ << " in file: " << __FILE__ << "\n"; \
        exit(-1); \
    }

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cout<< "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
        std::cout << "Error occurred at line: " << __LINE__ << " in file: " << __FILE__ << "\n"; \
        exit(-1); \
    }

#endif // CUDA_UTIL_H