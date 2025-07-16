#include "lingodb/runtime/ListRuntime.h"
using namespace lingodb::runtime;

List* List::create(size_t sizeOfType) {
   auto* list = new List(sizeOfType);
   getCurrentExecutionContext()->registerState({list, [](void* ptr) { delete reinterpret_cast<List*>(ptr); }});
   return list;
}
uint8_t* List::append() {
   if ((len + 1) * sizeOfType > values.size()) {
      values.resize(values.size() * 2);
   }
   auto* res = values.data() + len * sizeOfType;
   len++;
   return res;
}

Buffer List::getBuffer() {
   return Buffer(len * sizeOfType, values.data());
}
List* List::fromBuffer(size_t sizeOfType, Buffer buffer) {
   auto* res = new List(sizeOfType);
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
