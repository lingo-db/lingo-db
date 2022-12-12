#include "runtime/Heap.h"
#include <cstring>

void runtime::Heap::bubbleDown(size_t idx,size_t end) {
   size_t leftChild = idx * 2;
   if (leftChild > end) return;
   size_t rightChild = idx * 2 + 1;
   size_t maxChild = leftChild;
   if (rightChild <= end && isLt(leftChild, rightChild)) {
      maxChild = rightChild;
   }
   if (isLt(idx, maxChild)) {
      swap(idx, maxChild);
      bubbleDown(maxChild,end);
   }
}
void runtime::Heap::buildHeap() {
   for (int i = currElements; i > 0; i--) {
      bubbleDown(i,currElements);
   }
}

void runtime::Heap::insert(uint8_t* currData) {
   if (currElements < maxElements) {
      memcpy(&data[typeSize * (currElements + 1)], currData, typeSize);
      currElements++;
      if (currElements == maxElements) {
         buildHeap();
      }
      return;
   }
   auto* lastData = &data[typeSize];
   if (cmpFn(currData, lastData)) {
      memcpy(&data[typeSize], currData, typeSize);
      bubbleDown(1,currElements);
   }
}
runtime::Heap* runtime::Heap::create(size_t maxElements, size_t typeSize, bool (*cmpFn)(unsigned char*, unsigned char*)) {
   return new Heap(maxElements, typeSize, cmpFn);
}
runtime::Buffer runtime::Heap::getBuffer() {
   if (currElements < maxElements) {
      buildHeap();
   }
   for(size_t i=0;i<currElements;i++){
      swap(1,currElements-i);
      bubbleDown(1,currElements-i-1);
   }

   return runtime::Buffer{currElements * std::max(1ul, typeSize), &data[typeSize]};
}
void runtime::Heap::destroy(Heap* h) {
   delete h;
}