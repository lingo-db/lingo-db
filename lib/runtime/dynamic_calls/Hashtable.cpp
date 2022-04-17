#include "runtime/Hashtable.h"
runtime::Hashtable* runtime::Hashtable::create(size_t typeSize, size_t initialCapacity) {
   return new (malloc(sizeof(Hashtable) + typeSize)) Hashtable(initialCapacity, typeSize);
}
void runtime::Hashtable::resize() {
   size_t oldHtSize = values.getCap() * 2;
   size_t newHtSize = oldHtSize * 2;
   values.resize();
   ht.setNewSize(newHtSize);
   size_t hashMask = newHtSize - 1;
   for (size_t i = 0; i < values.getLen(); i++) {
      auto* entry = values.ptrAt<Entry>(i);
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = entry;
      entry->next = previousPtr;
   }
}
void runtime::Hashtable::destroy(runtime::Hashtable* ht) {
   ht->~Hashtable();
   free(ht);
}