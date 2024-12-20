#ifndef LINGODB_RUNTIME_GPU_CUDA_DEVICEHEAP_CUH
#define LINGODB_RUNTIME_GPU_CUDA_DEVICEHEAP_CUH
#include <cassert>
#include <stdint.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace cudaRT {
__device__ static void* heapAlloc(uint64_t numBytes) {
   void* result = nullptr;
#ifdef GALLATIN_ENABLED
   result = gallatin::allocators::global_malloc(numBytes);
#else
   result = malloc(numBytes);
#endif
   // if(!result){
   //     printf("[ERROR] memAlloc returned nullptr for %llu bytes alloc, already allocated %llu bytes!\n", numBytes, totallyAllocated);
   // }
   return result;
}

__device__ static void freeHeapPtr(void* ptr) {
#ifdef GALLATIN_ENABLED
   gallatin::allocators::global_free(ptr);
#else
   free(ptr);
#endif
}
}
#else
[[maybe_unused]] void* heapAlloc(uint64_t numBytes) {
   assert(0 && "You shouldn't call it outside CUDA code!");
   return nullptr;
};
[[maybe_unused]] void freeHeapPtr(void* ptr) { assert(0 && "You shouldn't call it outside CUDA code!"); };

#endif

#endif //LINGODB_RUNTIME_GPU_CUDA_DEVICEHEAP_CUH
