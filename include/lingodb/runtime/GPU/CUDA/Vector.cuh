#ifndef LINGODB_RUNTIME_GPU_CUDA_VECTOR_CUH
#define LINGODB_RUNTIME_GPU_CUDA_VECTOR_CUH

#include "DefinitionHelper.cuh"

#include <cstdint>
#include <iostream>
namespace cudaRT {

struct Buffer {
   int64_t numElements{0};
   uint8_t* ptr;
};

struct BufferIterator {
   virtual bool isValid() = 0;
   virtual void next() = 0;
   virtual Buffer getCurrentBuffer() = 0;
   virtual void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void*) = 0;
   static bool isIteratorValid(BufferIterator* iterator);
   static void iteratorNext(BufferIterator* iterator);

   static Buffer iteratorGetCurrentBuffer(BufferIterator* iterator);
   static void destroy(BufferIterator* iterator);
   static void iterate(BufferIterator* iterator, bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) {
      iterator->iterateEfficient(parallel, forEachChunk, contextPtr);
   }
   virtual ~BufferIterator() {}
};

class OneBufferIterator : public BufferIterator {
   Buffer buf;

   public:
   OneBufferIterator(Buffer buffer) : buf(buffer) {}
   bool isValid() override { return true; }
   void next() override {}
   Buffer getCurrentBuffer() override {
      return buf;
   }
   void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
      forEachChunk(buf, contextPtr);
   }
};

struct Vec {
   Buffer* payLoad{nullptr};
   int32_t numElems{0};
   int32_t capacity{64};
   CUDA_DEVICE Buffer* allocCurrentCapacity();
   CUDA_HOST_DEVICE Vec() {}
   CUDA_DEVICE ~Vec() {} // should call destroy explicitly instead of relying on destructor!
   CUDA_DEVICE void destroy();
   CUDA_DEVICE void grow();
   CUDA_DEVICE void push_back(const Buffer& elem); // NOLINT(readability-identifier-naming)
   CUDA_DEVICE void merge(Vec* other);
   CUDA_DEVICE Buffer& operator[](const int32_t index) { return payLoad[index]; }
   CUDA_DEVICE Buffer& back() { return payLoad[numElems - 1]; }
   CUDA_DEVICE int32_t size() const { return numElems; }
};
} // namespace cudaRT
#endif //LINGODB_RUNTIME_GPU_CUDA_VECTOR_CUH