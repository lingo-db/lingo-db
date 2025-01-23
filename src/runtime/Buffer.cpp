#include "lingodb/runtime/Buffer.h"
#include "lingodb/utility/Tracer.h"
#include <iostream>
namespace {
static utility::Tracer::Event iterateEvent("FlexibleBuffer", "iterateParallel");
static utility::Tracer::Event bufferIteratorEvent("BufferIterator", "iterate");

} // end namespace
bool lingodb::runtime::BufferIterator::isIteratorValid(lingodb::runtime::BufferIterator* iterator) {
   return iterator->isValid();
}
void lingodb::runtime::BufferIterator::iteratorNext(lingodb::runtime::BufferIterator* iterator) {
   iterator->next();
}
lingodb::runtime::Buffer lingodb::runtime::BufferIterator::iteratorGetCurrentBuffer(lingodb::runtime::BufferIterator* iterator) {
   return iterator->getCurrentBuffer();
}
void lingodb::runtime::BufferIterator::destroy(lingodb::runtime::BufferIterator* iterator) {
   delete iterator;
}
void lingodb::runtime::FlexibleBuffer::iterateBuffersParallel(const std::function<void(Buffer)>& fn) {
   //todo: scheduler
   for (auto& buffer : buffers) {
      utility::Tracer::Trace trace(iterateEvent);
      fn(buffer);
      trace.stop();
   }
   /*
   tbb::parallel_for_each(buffers.begin(), buffers.end(), [&](Buffer buffer, tbb::feeder<Buffer>& feeder) {
      if (buffer.numElements <= 20000) {
         utility::Tracer::Trace trace(iterateEvent);
         //trace.setMetaData(buffer.numElements);
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
   */
}
class FlexibleBufferIterator : public lingodb::runtime::BufferIterator {
   lingodb::runtime::FlexibleBuffer& flexibleBuffer;
   size_t currBuffer;

   public:
   FlexibleBufferIterator(lingodb::runtime::FlexibleBuffer& flexibleBuffer) : flexibleBuffer(flexibleBuffer), currBuffer(0) {}
   bool isValid() override {
      return currBuffer < flexibleBuffer.getBuffers().size();
   }
   void next() override {
      currBuffer++;
   }
   lingodb::runtime::Buffer getCurrentBuffer() override {
      lingodb::runtime::Buffer orig = flexibleBuffer.getBuffers().at(currBuffer);
      return lingodb::runtime::Buffer{orig.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), orig.ptr};
   }
   void iterateEfficient(bool parallel, void (*forEachChunk)(lingodb::runtime::Buffer, void*), void* contextPtr) override {
      if (parallel) {
         flexibleBuffer.iterateBuffersParallel([&](lingodb::runtime::Buffer buffer) {
            buffer = lingodb::runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            forEachChunk(buffer, contextPtr);
         });
      } else {
         for (auto buffer : flexibleBuffer.getBuffers()) {
            buffer = lingodb::runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            forEachChunk(buffer, contextPtr);
         }
      }
   }
};

lingodb::runtime::BufferIterator* lingodb::runtime::FlexibleBuffer::createIterator() {
   return new FlexibleBufferIterator(*this);
}
size_t lingodb::runtime::FlexibleBuffer::getLen() const {
   return totalLen;
}

void lingodb::runtime::BufferIterator::iterate(lingodb::runtime::BufferIterator* iterator, bool parallel, void (*forEachChunk)(lingodb::runtime::Buffer, void*), void* contextPtr) {
   utility::Tracer::Trace trace(bufferIteratorEvent);
   iterator->iterateEfficient(parallel, forEachChunk, contextPtr);
}

void lingodb::runtime::Buffer::iterate(bool parallel, lingodb::runtime::Buffer buffer, size_t typeSize, void (*forEachChunk)(lingodb::runtime::Buffer, size_t, size_t, void*), void* contextPtr) {
   size_t len = buffer.numElements / typeSize;

   auto range = tbb::blocked_range<size_t>(0, len);
   //todo: scheduler
   if (parallel&&false) {
      tbb::parallel_for(range, [&](tbb::blocked_range<size_t> r) {
         utility::Tracer::Trace trace(iterateEvent);
         //trace.setMetaData(r.size());
         forEachChunk(buffer, r.begin(), r.end(), contextPtr);
         trace.stop();
      });
   } else {
      forEachChunk(buffer, 0, buffer.numElements / typeSize, contextPtr);
   }
}