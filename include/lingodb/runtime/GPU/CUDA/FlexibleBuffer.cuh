#ifndef LINGODB_RUNTIME_GPU_CUDA_FLEXIBLEBUFFER_CUH
#define LINGODB_RUNTIME_GPU_CUDA_FLEXIBLEBUFFER_CUH
#include "DefinitionHelper.cuh"
#include "Vector.cuh"

#include <stdio.h>

namespace cudaRT {
class FlexibleBuffer {
   Vec buffers;
   int32_t totalLen{0};
   int32_t currCapacity{128};
   int32_t typeSize{0};
   int32_t lock{0};
   CUDA_DEVICE uint8_t* allocCurrentCapacity();
   CUDA_DEVICE void nextBuffer(int numElems) {
      while (currCapacity < numElems) {
         currCapacity *= 2;
      }
      // Cut off at 4MB to prevent underutilization of large blocks at fine locality levels
      if (currCapacity * typeSize < 4 * 1024 * 1024) {
         currCapacity *= 2;
      }
      uint8_t* ptr = allocCurrentCapacity();
      // printf("[nextBuffer][%p] capacity=%d, ptr = %p\n ", this, currCapacity, ptr );
      buffers.push_back(Buffer{0, ptr});
   }

   public:
   CUDA_DEVICE FlexibleBuffer() {}
   CUDA_HOST_DEVICE FlexibleBuffer(int32_t typeSize) : typeSize(typeSize) {}
   CUDA_DEVICE FlexibleBuffer(int32_t initialCapacity, int32_t typeSize, bool initialAlloc);
   CUDA_DEVICE ~FlexibleBuffer() {}
   CUDA_DEVICE void destroy();
   CUDA_DEVICE int32_t* getLock() { return &lock; }
   CUDA_DEVICE uint8_t* insertWarpOpportunistic();
   CUDA_DEVICE uint8_t* insert(const int32_t numElems) {
      if (buffers.size() == 0 || buffers.back().numElements + numElems >= currCapacity) {
         nextBuffer(numElems);
      }
      uint8_t* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
      // printf("[insert][%p] buffers.back().ptr = %p, buffers.back().numElements = %ld += %d,  typeSize=%d\n ", this, buffers.back().ptr, buffers.back().numElements, numElems, typeSize);
      buffers.back().numElements += numElems;
      totalLen += numElems;
      return res;
   }
   CUDA_DEVICE void merge(FlexibleBuffer* other) {
      // printf("[GPU][FlexibleBuffer::merge] totalLen = %d, other->getLen() = %d, %p <- %p, myTypeSize=%d, other->typeSize()=%d\n", totalLen, other->getLen(), this, other, typeSize, other->getTypeSize());
      buffers.merge(&other->getBuffers());
      currCapacity = (currCapacity > other->getCapacity()) ? currCapacity : other->getCapacity();
      totalLen += other->getLen();
   }
   CUDA_HOST_DEVICE int32_t getLen() const { return totalLen; }
   CUDA_HOST_DEVICE int32_t buffersSize() { return buffers.numElems; }
   CUDA_HOST_DEVICE int32_t getTypeSize() const { return typeSize; }
   CUDA_HOST_DEVICE int32_t getCapacity() const { return currCapacity; }
   CUDA_HOST_DEVICE Vec& getBuffers() { return buffers; }
   CUDA_DEVICE void printMetaInfo(void (*printEntry)(uint8_t*) = nullptr) {
      printf("--------------------FlexibleBuffer [%p]--------------------\n", this);
      printf("totalLen=%d, currCapacity=%d, typeSize=%d, buffersLen=%d\n", totalLen, currCapacity, typeSize, buffers.size());
      for (int i = 0; i < buffers.size(); i++) {
         printf("-  Buffer %d has %ld elements\n", i, buffers[i].numElements);
         if (printEntry) {
            for (int elIdx = 0; elIdx < buffers[i].numElements; elIdx++) {
               printEntry(&buffers[i].ptr[elIdx * typeSize]);
               printf("\n");
            }
         }
      }
      printf("-----------------------------------------------------------\n");
   }
};

class FlexibleBufferIterator {
   FlexibleBuffer* parent;
   int32_t start{0};
   int32_t stride{0};
   int32_t bufferIndex{0};
   int32_t localIndex{0};

   CUDA_DEVICE uint8_t* currentPointer() {
      if (bufferIndex < parent->getBuffers().size()) {
         return &parent->getBuffers()[bufferIndex].ptr[localIndex * parent->getTypeSize()];
      } else {
         return nullptr;
      }
   }

   public:
   CUDA_DEVICE FlexibleBufferIterator(FlexibleBuffer* parent, int32_t start, int32_t stride) : parent(parent), start(start), stride(stride) {}
   CUDA_DEVICE uint8_t* initialize() { // Given start, find position across buffers
      int32_t numBuffers = parent->getBuffers().size();
      int32_t remaining = start;
      uint8_t* res{nullptr};
      for (bufferIndex = 0; bufferIndex < numBuffers; ++bufferIndex) {
         if (remaining < parent->getBuffers()[bufferIndex].numElements) {
            localIndex = remaining;
            res = currentPointer();
            break;
         }
         remaining -= parent->getBuffers()[bufferIndex].numElements;
      }
      return res;
   }

   CUDA_DEVICE uint8_t* step() {
      int32_t numBuffers = parent->getBuffers().size();
      uint8_t* res{nullptr};
      if (bufferIndex < numBuffers) {
         localIndex += stride;
         while (bufferIndex < numBuffers && localIndex >= parent->getBuffers()[bufferIndex].numElements) {
            localIndex -= parent->getBuffers()[bufferIndex].numElements;
            bufferIndex++;
         }
         res = currentPointer();
      }
      return res;
   }
};

} // namespace cudaRT

#endif //LINGODB_RUNTIME_GPU_CUDA_FLEXIBLEBUFFER_CUH
