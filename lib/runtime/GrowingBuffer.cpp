#include "runtime/GrowingBuffer.h"
#include <algorithm>
#include <cstring>
#include <iostream>

runtime::GrowingBuffer* runtime::GrowingBuffer::create(size_t sizeOfType, size_t initialCapacity) {
   return new GrowingBuffer(initialCapacity, sizeOfType);
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
runtime::Buffer runtime::GrowingBuffer::sort(bool (*compareFn)(uint8_t*, uint8_t*)) {
   std::vector<uint8_t*> toSort;
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });
   size_t typeSize=values.getTypeSize();
   size_t len=values.getLen();
   std::sort(toSort.begin(), toSort.end(), [&](uint8_t* left, uint8_t* right) {
      return compareFn(left, right);
   });
   uint8_t* sorted = new uint8_t[typeSize * len];
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = sorted + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }
   return Buffer{typeSize * len,sorted};
}
runtime::Buffer runtime::GrowingBuffer::asContinuous(){
   //todo make more performant...
   std::vector<uint8_t*> toSort;
   values.iterate([&](uint8_t* entryRawPtr) {
      toSort.push_back(entryRawPtr);
   });
   size_t typeSize=values.getTypeSize();
   size_t len=values.getLen();
   uint8_t* continuous = new uint8_t[typeSize * len];
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = continuous + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }
   return Buffer{typeSize * len, continuous};
}
void runtime::GrowingBuffer::destroy(GrowingBuffer* vec) {
   delete vec;
}
runtime::BufferIterator* runtime::GrowingBuffer::createIterator() {
   return values.createIterator();
}
