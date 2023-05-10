#include "runtime/Buffer.h"
#include "utility/Tracer.h"
#include <iostream>
#include <oneapi/tbb.h>
namespace {
static utility::Tracer::Event iterateEvent("FlexibleBuffer", "iterateParallel");
static utility::Tracer::Event bufferIteratorEvent("BufferIterator", "iterate");

} // end namespace
bool runtime::BufferIterator::isIteratorValid(runtime::BufferIterator* iterator) {
   return iterator->isValid();
}
void runtime::BufferIterator::iteratorNext(runtime::BufferIterator* iterator) {
   iterator->next();
}
runtime::Buffer runtime::BufferIterator::iteratorGetCurrentBuffer(runtime::BufferIterator* iterator) {
   return iterator->getCurrentBuffer();
}
void runtime::BufferIterator::destroy(runtime::BufferIterator* iterator) {
   delete iterator;
}
void runtime::FlexibleBuffer::iterateBuffersParallel(const std::function<void(Buffer)>& fn) {
   tbb::parallel_for_each(buffers.begin(), buffers.end(), [&](Buffer buffer, tbb::feeder<Buffer>& feeder) {
      if (buffer.numElements <= 20000) {
         utility::Tracer::Trace trace(iterateEvent);
         trace.setMetaData(buffer.numElements);
         fn(buffer);
         trace.stop();
      } else {
         for (size_t i = 0; i < buffer.numElements; i += 20000) {
            size_t begin = i;
            size_t end = std::min(i + 20000, buffer.numElements);
            size_t len = end - begin;
            feeder.add({len, buffer.ptr + begin * std::max(1ul, getTypeSize())});
         }
      }
   });
}
class FlexibleBufferIterator : public runtime::BufferIterator {
   runtime::FlexibleBuffer& flexibleBuffer;
   size_t currBuffer;

   public:
   FlexibleBufferIterator(runtime::FlexibleBuffer& flexibleBuffer) : flexibleBuffer(flexibleBuffer), currBuffer(0) {}
   bool isValid() override {
      return currBuffer < flexibleBuffer.getBuffers().size();
   }
   void next() override {
      currBuffer++;
   }
   runtime::Buffer getCurrentBuffer() override {
      runtime::Buffer orig = flexibleBuffer.getBuffers().at(currBuffer);
      return runtime::Buffer{orig.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), orig.ptr};
   }
   void iterateEfficient(bool parallel, void (*forEachChunk)(runtime::Buffer, void*), void* contextPtr) override {
      if (parallel) {
         flexibleBuffer.iterateBuffersParallel([&](runtime::Buffer buffer) {
            buffer = runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            forEachChunk(buffer, contextPtr);
         });
      } else {
         for (auto buffer : flexibleBuffer.getBuffers()) {
            buffer = runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            forEachChunk(buffer, contextPtr);
         }
      }
   }
};

runtime::BufferIterator* runtime::FlexibleBuffer::createIterator() {
   return new FlexibleBufferIterator(*this);
}
size_t runtime::FlexibleBuffer::getLen() const {
   return totalLen;
}

void runtime::BufferIterator::iterate(runtime::BufferIterator* iterator, bool parallel, void (*forEachChunk)(runtime::Buffer, void*), void* contextPtr) {
   utility::Tracer::Trace trace(bufferIteratorEvent);
   iterator->iterateEfficient(parallel, forEachChunk, contextPtr);
}

void runtime::Buffer::iterate(bool parallel, runtime::Buffer buffer, size_t typeSize, void (*forEachChunk)(runtime::Buffer, size_t, size_t, void*), void* contextPtr) {
   size_t len = buffer.numElements / typeSize;

   auto range = tbb::blocked_range<size_t>(0, len);

   if (parallel) {
      tbb::parallel_for(range, [&](tbb::blocked_range<size_t> r) {
         utility::Tracer::Trace trace(iterateEvent);
         trace.setMetaData(r.size());
         forEachChunk(buffer, r.begin(), r.end(), contextPtr);
         trace.stop();
      });
   } else {
      forEachChunk(buffer, 0, buffer.numElements / typeSize, contextPtr);
   }
}