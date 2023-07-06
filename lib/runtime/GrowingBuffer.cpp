#include "runtime/GrowingBuffer.h"
#include "runtime/helpers.h"
#include "utility/Tracer.h"
#include <algorithm>
#include <cstring>
#include <iostream>
namespace {
static utility::Tracer::Event createEvent("GrowingBuffer", "create");
static utility::Tracer::Event mergeEvent("GrowingBuffer", "merge");
static utility::Tracer::Event sortEvent("GrowingBuffer", "sort");
static utility::Tracer::Event rawSortEvent("GrowingBuffer", "rawSort");

class DefaultAllocator : public runtime::GrowingBufferAllocator {
   public:
   runtime::GrowingBuffer* create(runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) override {
      utility::Tracer::Trace trace(createEvent);
      auto* res = new runtime::GrowingBuffer(initialCapacity, sizeOfType);
      executionContext->registerState({res, [](void* ptr) { delete reinterpret_cast<runtime::GrowingBuffer*>(ptr); }});
      trace.stop();
      return res;
   }
};

class GroupAllocator : public runtime::GrowingBufferAllocator {
   std::vector<runtime::GrowingBuffer*> buffers;

   public:
   runtime::GrowingBuffer* create(runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) override {
      auto* res = new runtime::GrowingBuffer(initialCapacity, sizeOfType);
      buffers.push_back(res);
      return res;
   }
   ~GroupAllocator() {
      for (auto* buf : buffers) {
         delete buf;
      }
   }
};

} // end namespace

runtime::GrowingBuffer* runtime::GrowingBuffer::create(runtime::GrowingBufferAllocator* allocator, runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) {
   return allocator->create(executionContext, sizeOfType, initialCapacity);
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
   utility::Tracer::Trace trace(sortEvent);
   std::vector<uint8_t*> toSort;
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });
   size_t typeSize = values.getTypeSize();
   size_t len = values.getLen();
   utility::Tracer::Trace trace2(rawSortEvent);
   tbb::parallel_sort(toSort.begin(), toSort.end(), compareFn);
   trace2.stop();
   uint8_t* sorted = new uint8_t[typeSize * len];
   executionContext->registerState({sorted, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   tbb::parallel_for(tbb::blocked_range<size_t>(0ul, len), [&](tbb::blocked_range<size_t> range) {
      for (size_t i = range.begin(); i < range.end(); i++) {
         uint8_t* ptr = sorted + (i * typeSize);
         memcpy(ptr, toSort[i], typeSize);
      }
   });

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
   auto* ptr = FixedSizedBuffer<uint8_t>::createZeroed(bytes);
   executionContext->registerState({ptr, [bytes](void* ptr) { FixedSizedBuffer<uint8_t>::deallocate((uint8_t*) ptr, bytes); }});
   return Buffer{bytes, ptr};
}

runtime::GrowingBufferAllocator* runtime::GrowingBufferAllocator::getDefaultAllocator() {
   static DefaultAllocator defaultAllocator;
   return &defaultAllocator;
}

runtime::GrowingBufferAllocator* runtime::GrowingBufferAllocator::getGroupAllocator(runtime::ExecutionContext* executionContext, size_t groupId) {
   auto& state = executionContext->getAllocator(groupId);
   if (state.ptr) {
      return static_cast<GrowingBufferAllocator*>(state.ptr);
   } else {
      auto* newAllocator = new GroupAllocator;
      state.ptr = newAllocator;
      state.freeFn = [](void* ptr) { delete static_cast<GroupAllocator*>(ptr); };
      return newAllocator;
   }
}