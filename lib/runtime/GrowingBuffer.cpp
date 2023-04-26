#include "runtime/GrowingBuffer.h"
#include "utility/Tracer.h"
#include <algorithm>
#include <cstring>
#include <iostream>
namespace {
static utility::Tracer::Event createEvent("GrowingBuffer", "create");
static utility::Tracer::Event mergeEvent("GrowingBuffer", "merge");
} // end namespace
runtime::GrowingBuffer* runtime::GrowingBuffer::create(runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) {
   utility::Tracer::Trace trace(createEvent);
   auto* res = new GrowingBuffer(initialCapacity, sizeOfType);
   executionContext->registerState({res, [](void* ptr) { delete reinterpret_cast<GrowingBuffer*>(ptr); }});
   trace.stop();
   return res;
}

uint8_t* runtime::GrowingBuffer::insert() {
   return values.insert();
}
size_t runtime::GrowingBuffer::getLen() const {
   return values.getLen();
}

size_t runtime::GrowingBuffer::getTypeSize() const {
   return values.getTypeSize();
}
runtime::Buffer runtime::GrowingBuffer::sort(runtime::ExecutionContext* executionContext, bool (*compareFn)(uint8_t*, uint8_t*)) {
   std::vector<uint8_t*> toSort;
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });
   size_t typeSize = values.getTypeSize();
   size_t len = values.getLen();
   std::sort(toSort.begin(), toSort.end(), [&](uint8_t* left, uint8_t* right) {
      return compareFn(left, right);
   });
   uint8_t* sorted = new uint8_t[typeSize * len];
   executionContext->registerState({sorted, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = sorted + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }
   return Buffer{typeSize * len, sorted};
}
runtime::Buffer runtime::GrowingBuffer::asContinuous(runtime::ExecutionContext* executionContext) {
   //todo make more performant...
   std::vector<uint8_t*> toSort;
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });
   size_t typeSize = values.getTypeSize();
   size_t len = values.getLen();
   uint8_t* continuous = new uint8_t[typeSize * len];
   executionContext->registerState({continuous, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = continuous + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }
   return Buffer{typeSize * len, continuous};
}
void runtime::GrowingBuffer::destroy(GrowingBuffer* vec) {
   delete vec;
}

runtime::GrowingBuffer* runtime::GrowingBuffer::merge(runtime::ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(mergeEvent);
   GrowingBuffer* first = nullptr;
   for (auto* ptr : threadLocal->getTls()) {
      auto* current = reinterpret_cast<GrowingBuffer*>(ptr);
      if (!first) {
         first = current;
      } else {
         first->values.merge(current->values); //todo: cleanup
      }
   }
   trace.stop();
   return first;
}
runtime::BufferIterator* runtime::GrowingBuffer::createIterator() {
   return values.createIterator();
}

runtime::Buffer runtime::Buffer::createZeroed(runtime::ExecutionContext* executionContext, size_t bytes) {
   auto* ptr = new uint8_t[bytes];
   std::memset(ptr, 0, bytes);
   executionContext->registerState({ptr, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   return Buffer{bytes, ptr};
}