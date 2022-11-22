#include "runtime/Hashtable.h"
runtime::Hashtable* runtime::Hashtable::create(size_t typeSize, size_t initialCapacity) {
   return new (malloc(sizeof(Hashtable) + typeSize)) Hashtable(initialCapacity, typeSize);
}
void runtime::Hashtable::resize() {
   size_t oldHtSize = hashMask + 1;
   size_t newHtSize = oldHtSize * 2;
   ht.setNewSize(newHtSize);
   hashMask = newHtSize - 1;
   values.iterate([&](uint8_t* entryRawPtr){
      auto* entry = (Entry*) entryRawPtr;
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = entry;
      entry->next = previousPtr;
   });
}
void runtime::Hashtable::destroy(runtime::Hashtable* ht) {
   ht->~Hashtable();
   free(ht);
}
runtime::Hashtable::Entry* runtime::Hashtable::insert(size_t hash) {
   if (values.getLen() > hashMask / 2) {
      resize();
   }
   Entry* res = (Entry*) values.insert();
   auto pos = hash & hashMask;
   auto* previousPtr = ht.at(pos);
   res->next = previousPtr;
   res->hashValue = hash;
   ht.at(pos) = res;
   return res;
}

runtime::BufferIterator* runtime::Hashtable::createIterator() {
   return values.createIterator();
}