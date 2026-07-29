#include "lingodb/runtime/ListRuntime.h"

#include <iostream>
using namespace lingodb::runtime;

List* List::create(size_t sizeOfType) {
   return createRefCounted<List>(sizeOfType, StorageClass::REFCOUNTED);
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
   getCurrentExecutionContext()->registerState({res, [](void* p) {
                                                   delete static_cast<List*>(p);
                                                }});
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
   if (list && list->storageClass == StorageClass::REFCOUNTED) {
      decRefCount<List>(list, cleanupFn);
   }
}

void List::cleanupUse(List* list) {
   if (list && list->storageClass == StorageClass::REFCOUNTED) {
      decRefCount<List>(list, [](List*) {});
   }
}

void List::addUse(List* list) {
   if (list && list->storageClass == StorageClass::REFCOUNTED) {
      incRefCount<List>(list);
   }
}
List* List::promoteToGlobal(List* list) {
   if (!list) return nullptr;
   if (list->storageClass == StorageClass::GLOBAL) return list;
   auto* newList = new List(list->sizeOfType, StorageClass::GLOBAL);
   newList->len = list->len;
   newList->values.resize(list->values.size());
   std::ranges::copy(list->values.begin(), list->values.end(), newList->values.begin());

   getCurrentExecutionContext()->registerState({newList, [](void* p) {
                                                   delete static_cast<List*>(p);
                                                }});
   if (list->storageClass == StorageClass::REFCOUNTED) {
      cleanupUse(list);
   }
   return newList;
}
