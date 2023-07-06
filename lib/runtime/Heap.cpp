#include "runtime/Heap.h"
#include "utility/Tracer.h"
#include <cstring>
namespace {
utility::Tracer::Event mergeHeapEvent("Heap", "merge");
} //end namespace

void runtime::Heap::bubbleDown(size_t idx, size_t end) {
   size_t leftChild = idx * 2;
   if (leftChild > end) return;
   size_t rightChild = idx * 2 + 1;
   size_t maxChild = leftChild;
   if (rightChild <= end && isLt(leftChild, rightChild)) {
      maxChild = rightChild;
   }
   if (isLt(idx, maxChild)) {
      swap(idx, maxChild);
      bubbleDown(maxChild, end);
   }
}
void runtime::Heap::buildHeap() {
   for (int i = currElements; i > 0; i--) {
      bubbleDown(i, currElements);
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
      bubbleDown(1, currElements);
   }
}
runtime::Heap* runtime::Heap::create(runtime::ExecutionContext* executionContext, size_t maxElements, size_t typeSize, bool (*cmpFn)(unsigned char*, unsigned char*)) {
   auto* heap = new Heap(maxElements, typeSize, cmpFn);
   executionContext->registerState({heap, [](void* ptr) { delete reinterpret_cast<Heap*>(ptr); }});
   return heap;
}
runtime::Buffer runtime::Heap::getBuffer() {
   if (currElements < maxElements) {
      buildHeap();
   }
   for (size_t i = 0; i < currElements; i++) {
      swap(1, currElements - i);
      bubbleDown(1, currElements - i - 1);
   }

   return runtime::Buffer{currElements * std::max(1ul, typeSize), &data[typeSize]};
}
void runtime::Heap::destroy(Heap* h) {
   delete h;
}

runtime::Heap* runtime::Heap::merge(runtime::ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(mergeHeapEvent);
   Heap* first = nullptr;
   for (auto* ptr : threadLocal->getTls()) {
      auto* current = reinterpret_cast<Heap*>(ptr);
      if (!first) {
         first = current;
      } else {
         first->mergeWithOther(current);
      }
   }
   return first;
}