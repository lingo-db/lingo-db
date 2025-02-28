#include "lingodb/runtime/GrowingBuffer.h"
#include "lingodb/runtime/Sorting.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/utility/Tracer.h"
#include <algorithm>
#include <cstring>
#include <iostream>
namespace {
static utility::Tracer::Event createEvent("GrowingBuffer", "create");
static utility::Tracer::Event mergeEvent("GrowingBuffer", "merge");
static utility::Tracer::Event sortEvent("GrowingBuffer", "sort");

class DefaultAllocator : public lingodb::runtime::GrowingBufferAllocator {
   public:
   lingodb::runtime::GrowingBuffer* create(lingodb::runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) override {
      utility::Tracer::Trace trace(createEvent);
      auto* res = new lingodb::runtime::GrowingBuffer(initialCapacity, sizeOfType);
      executionContext->registerState({res, [](void* ptr) { delete reinterpret_cast<lingodb::runtime::GrowingBuffer*>(ptr); }});
      trace.stop();
      return res;
   }
};

class GroupAllocator : public lingodb::runtime::GrowingBufferAllocator {
   std::vector<lingodb::runtime::GrowingBuffer*> buffers;

   public:
   lingodb::runtime::GrowingBuffer* create(lingodb::runtime::ExecutionContext* executionContext, size_t sizeOfType, size_t initialCapacity) override {
      auto* res = new lingodb::runtime::GrowingBuffer(initialCapacity, sizeOfType);
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

lingodb::runtime::GrowingBuffer* lingodb::runtime::GrowingBuffer::create(lingodb::runtime::GrowingBufferAllocator* allocator, size_t sizeOfType, size_t initialCapacity) {
   lingodb::runtime::ExecutionContext* executionContext= runtime::getCurrentExecutionContext();
   return allocator->create(executionContext, sizeOfType, initialCapacity);
}

uint8_t* lingodb::runtime::GrowingBuffer::insert() {
   return values.insert();
}
size_t lingodb::runtime::GrowingBuffer::getLen() const {
   return values.getLen();
}

size_t lingodb::runtime::GrowingBuffer::getTypeSize() const {
   return values.getTypeSize();
}
lingodb::runtime::Buffer lingodb::runtime::GrowingBuffer::sort(bool (*compareFn)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace trace(sortEvent);
   if (canParallelSort(values.getLen())) {
      lingodb::runtime::Buffer result = parallelSort(values, compareFn);
      trace.stop();
      return result;
   }
   auto* executionContext= runtime::getCurrentExecutionContext();
   std::vector<uint8_t*> toSort;
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });
   size_t typeSize = values.getTypeSize();
   size_t len = values.getLen();
   std::sort(toSort.begin(), toSort.end(), compareFn);
   trace.stop();
   uint8_t* sorted = new uint8_t[typeSize * len];
   executionContext->registerState({sorted, [](void* ptr) { delete[] reinterpret_cast<uint8_t*>(ptr); }});
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = sorted + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }

   return Buffer{typeSize * len, sorted};
}
lingodb::runtime::Buffer lingodb::runtime::GrowingBuffer::asContinuous() {
   auto* executionContext= runtime::getCurrentExecutionContext();
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
void lingodb::runtime::GrowingBuffer::destroy(GrowingBuffer* vec) {
   delete vec;
}

lingodb::runtime::GrowingBuffer* lingodb::runtime::GrowingBuffer::merge(lingodb::runtime::ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(mergeEvent);
   GrowingBuffer* first = nullptr;
   for (auto* current : threadLocal->getThreadLocalValues<GrowingBuffer>()) {
      if(!current) continue;
      if (!first) {
         first = current;
      } else {
         first->values.merge(current->values); //todo: cleanup
      }
   }
   trace.stop();
   return first;
}
lingodb::runtime::BufferIterator* lingodb::runtime::GrowingBuffer::createIterator() {
   return values.createIterator();
}

lingodb::runtime::Buffer lingodb::runtime::Buffer::createZeroed(size_t bytes) {
   auto* executionContext= runtime::getCurrentExecutionContext();
   auto* ptr = FixedSizedBuffer<uint8_t>::createZeroed(bytes);
   executionContext->registerState({ptr, [bytes](void* ptr) { FixedSizedBuffer<uint8_t>::deallocate((uint8_t*) ptr, bytes); }});
   return Buffer{bytes, ptr};
}

lingodb::runtime::GrowingBufferAllocator* lingodb::runtime::GrowingBufferAllocator::getDefaultAllocator() {
   static DefaultAllocator defaultAllocator;
   return &defaultAllocator;
}

lingodb::runtime::GrowingBufferAllocator* lingodb::runtime::GrowingBufferAllocator::getGroupAllocator(size_t groupId) {
   lingodb::runtime::ExecutionContext* executionContext= runtime::getCurrentExecutionContext();
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