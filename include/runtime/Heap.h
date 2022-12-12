#ifndef RUNTIME_HEAP_H
#define RUNTIME_HEAP_H
#include "Buffer.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace runtime {
class Heap {
   using CmpFn = std::add_pointer<bool(uint8_t* left, uint8_t* right)>::type;

   CmpFn cmpFn;
   size_t typeSize;
   size_t maxElements;
   size_t currElements;
   uint8_t* data;
   void bubbleDown(size_t idx, size_t end);
   void buildHeap();
   bool isLt(size_t l, size_t r) {
      return cmpFn(&data[l * typeSize], &data[r * typeSize]);
   }
   void swap(size_t l, size_t r) {
      std::swap_ranges(&data[l * typeSize], &data[l * typeSize + typeSize], &data[r * typeSize]);
   }

   public:
   Heap(size_t maxElements, size_t typeSize, CmpFn cmpFn) : cmpFn(cmpFn), typeSize(typeSize), maxElements(maxElements), currElements(0), data(new uint8_t[typeSize * (maxElements + 1)]) {
   }
   static Heap* create(size_t maxElements, size_t typeSize, bool (*cmpFn)(unsigned char*, unsigned char*));
   void insert(uint8_t* currData);
   Buffer getBuffer();
   static void destroy(Heap*);
   ~Heap() {
      delete[] data;
   }
};
} // namespace runtime
#endif // RUNTIME_HEAP_H
