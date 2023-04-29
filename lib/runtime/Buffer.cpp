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
   tbb::parallel_for_each(buffers.begin(), buffers.end(), [&](const Buffer& buffer) {
      utility::Tracer::Trace trace(iterateEvent);
      trace.setMetaData(buffer.numElements);
      fn(buffer);
      trace.stop();
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
   void iterateEfficient(bool parallel,void (*forEachChunk)(runtime::Buffer, void*), void* contextPtr) override{
      if(parallel){
         flexibleBuffer.iterateBuffersParallel([&](runtime::Buffer buffer){
            buffer=runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()),buffer.ptr};
            forEachChunk(buffer,contextPtr);
         });
      }else{
         for(auto buffer:flexibleBuffer.getBuffers()){
            buffer=runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()),buffer.ptr};
            forEachChunk(buffer,contextPtr);
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

void runtime::BufferIterator::iterate(runtime::BufferIterator* iterator,bool parallel, void (*forEachChunk)(runtime::Buffer, void*), void* contextPtr) {
   utility::Tracer::Trace trace(bufferIteratorEvent);
   iterator->iterateEfficient(parallel,forEachChunk,contextPtr);
}
