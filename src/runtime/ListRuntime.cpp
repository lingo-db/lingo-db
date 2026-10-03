#include "lingodb/runtime/ListRuntime.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
using namespace lingodb::runtime;

List* List::create(size_t sizeOfType) {
   return createRefCounted<List>(sizeOfType);
}
uint8_t* List::append() {
   if ((len + 1) * sizeOfType > values.size()) {
      values.resize(values.size() * 2);
   }
   auto* res = values.data() + len * sizeOfType;
   len++;
   return res;
}

uint8_t* List::pop(size_t pos) {
   if (len == 0) {
      throw std::runtime_error("pop from empty list");
   }
   if (pos >= len) {
      throw std::runtime_error("pop index out of range");
   }
   // move the element behind the (shortened) end, where it stays until the next append
   std::vector<uint8_t> element(values.data() + pos * sizeOfType, values.data() + (pos + 1) * sizeOfType);
   std::memmove(values.data() + pos * sizeOfType, values.data() + (pos + 1) * sizeOfType, (len - pos - 1) * sizeOfType);
   len--;
   auto* res = values.data() + len * sizeOfType;
   std::memcpy(res, element.data(), sizeOfType);
   return res;
}

Buffer List::getBuffer() {
   return Buffer(len * sizeOfType, values.data());
}
List* List::fromBuffer(size_t sizeOfType, Buffer buffer) {
   auto* res = createRefCounted<List>(sizeOfType);
   res->len = buffer.numElements / sizeOfType;
   res->values.resize(buffer.numElements);
   memcpy(res->values.data(), buffer.ptr, buffer.numElements);
   return res;
}
uint8_t* List::at(size_t pos) {
   if (pos >= len) {
      throw std::runtime_error("out of bounds");
   }
   return values.data() + pos * sizeOfType;
}

size_t List::size() {
   return len;
}

void List::cleanupUseCb(List* list, void (*cleanupFn)(List*)) {
   decRefCount<List>(list, cleanupFn);
}

void List::cleanupUse(List* list) {
   decRefCount<List>(list, [](List*) {});
}

void List::addUse(List* list) {
   incRefCount<List>(list);
}

void List::sort(bool (*compareFn)(const void*, const void*)) {
   std::vector<uint8_t*> toSort;
   toSort.reserve(len);
   for (size_t i = 0; i < len; i++) {
      toSort.push_back(values.data() + i * sizeOfType);
   }
   std::sort(toSort.begin(), toSort.end(), compareFn);
   std::vector<uint8_t> sorted(values.size());
   for (size_t i = 0; i < len; i++) {
      memcpy(sorted.data() + i * sizeOfType, toSort[i], sizeOfType);
   }
   values = std::move(sorted);
}
