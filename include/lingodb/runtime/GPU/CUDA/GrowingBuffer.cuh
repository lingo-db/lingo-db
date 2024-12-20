#ifndef LINGODB_RUNTIME_GPU_CUDA_GROWINGBUFFER_CUH
#define LINGODB_RUNTIME_GPU_CUDA_GROWINGBUFFER_CUH

#include "FlexibleBuffer.cuh"

namespace cudaRT{
class GrowingBuffer {
   FlexibleBuffer values;
   public:
   CUDA_HOST_DEVICE GrowingBuffer(int32_t cap, int32_t typeSize, bool allocateInit) : values(cap, typeSize, allocateInit) {}
   CUDA_HOST_DEVICE GrowingBuffer(int32_t typeSize) : values(typeSize) {}

   CUDA_DEVICE uint8_t* insert(const int32_t numElems){return values.insert(numElems);};
   CUDA_DEVICE uint8_t* insertWarpOpportunistic(){return values.insertWarpOpportunistic();}

   CUDA_HOST_DEVICE int32_t getLen() const {return values.getLen();};
   CUDA_HOST_DEVICE int32_t getTypeSize() const {return values.getTypeSize();};
   //    __device__ Buffer sort(runtime::ExecutionContext*, bool (*compareFn)(uint8_t*, uint8_t*));
   //    __device__ Buffer asContinuous(ExecutionContext* executionContext);
   //  __device__ static void destroy(GrowingBuffer* vec);
   CUDA_DEVICE void merge(GrowingBuffer* other) { values.merge(&other->values);};
   CUDA_HOST_DEVICE FlexibleBuffer* getValuesPtr() { return &values; }
};

}// namespace cudaRT
#endif //LINGODB_RUNTIME_GPU_CUDA_GROWINGBUFFER_CUH
