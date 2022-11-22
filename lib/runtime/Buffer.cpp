#include "runtime/Buffer.h"

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
};

runtime::BufferIterator* runtime::FlexibleBuffer::createIterator() {
   return new FlexibleBufferIterator(*this);
}
size_t runtime::FlexibleBuffer::getLen() const {
   return totalLen;
}
