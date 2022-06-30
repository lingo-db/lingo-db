#include "runtime/Vector.h"
#include <algorithm>
runtime::Vector* runtime::Vector::create(size_t sizeOfType, size_t initialCapacity) {
   return new Vector(initialCapacity, sizeOfType);
}
void runtime::Vector::resize() {
   size_t newCapacity = cap * 2;
   ptr = runtime::MemoryHelper::resize(ptr, len * typeSize, newCapacity * typeSize);
   cap = newCapacity;
}
size_t runtime::Vector::getLen() const {
   return len;
}
size_t runtime::Vector::getCap() const {
   return cap;
}
uint8_t* runtime::Vector::getPtr() const {
   return ptr;
}
size_t runtime::Vector::getTypeSize() const {
   return typeSize;
}
void runtime::Vector::sort(bool (*compareFn)(uint8_t*, uint8_t*)) {
   std::vector<uint8_t*> toSort;
   for (size_t i = 0; i < len; i++) {
      toSort.push_back(ptrAt<uint8_t>(i));
   }
   std::sort(toSort.begin(), toSort.end(), [&](uint8_t* left, uint8_t* right) {
      return compareFn(left, right);
   });
   uint8_t* sorted = new uint8_t[typeSize * len];
   for (size_t i = 0; i < len; i++) {
      uint8_t* ptr = sorted + (i * typeSize);
      memcpy(ptr, toSort[i], typeSize);
   }
   memcpy(ptr, sorted, typeSize * len);
   delete[] sorted;
}
void runtime::Vector::destroy(Vector* vec) {
   delete vec;
}
