#include "runtime/Hashtable.h"
runtime::Hashtable* runtime::Hashtable::create(size_t typeSize, size_t initialCapacity) {
   return new (malloc(sizeof(Hashtable) + typeSize)) Hashtable(initialCapacity, typeSize);
}
void runtime::Hashtable::resize() {
   size_t oldHtSize = hashMask + 1;
   size_t newHtSize = oldHtSize * 2;
   ht.setNewSize(newHtSize);
   hashMask = newHtSize - 1;
   values.iterate([&](uint8_t* entryRawPtr) {
      auto* entry = (Entry*) entryRawPtr;
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = runtime::tag(entry, previousPtr, entry->hashValue);
      entry->next = runtime::untag(previousPtr);
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
   ht.at(pos) = runtime::tag(res, previousPtr, hash);
   res->next = runtime::untag(previousPtr);
   res->hashValue = hash;
   return res;
}

runtime::BufferIterator* runtime::Hashtable::createIterator() {
   return values.createIterator();
}