#ifndef CUDA_RUNTIME_LAZYJOINHASHTABLE_H
#define CUDA_RUNTIME_LAZYJOINHASHTABLE_H
#include <stdio.h>

#include "DefinitionHelper.cuh"
#include "GrowingBuffer.cuh"
namespace cudaRT{

class HashIndexedView {
   public:
   struct Entry {
      Entry* next;
      uint64_t hashValue;
      //kv follows
   };
   Entry** ht{nullptr};
   uint64_t htMask{0}; //NOLINT(clang-diagnostic-unused-private-field)
   CUDA_HOST HashIndexedView(uint8_t* ht, uint64_t htMask) : ht(reinterpret_cast<Entry**>(ht)), htMask(htMask) {};
   CUDA_HOST HashIndexedView(uint64_t htSize) : htMask(htSize-1) {};
   CUDA_DEVICE static void destroy(HashIndexedView*);
   CUDA_DEVICE void printMetaInfo();
};

extern "C" HashIndexedView* buildHashIndexedView(GrowingBuffer* buffer, uint8_t* stream); // C++ interface

}
#endif // CUDA_RUNTIME_LAZYJOINHASHTABLE_H
