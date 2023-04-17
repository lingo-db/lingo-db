#include "runtime/HashMultiMap.h"

runtime::HashMultiMap* runtime::HashMultiMap::create(runtime::ExecutionContext* executionContext,size_t entryTypeSize, size_t valueTypeSize, size_t initialCapacity) {
   auto* hmm= new HashMultiMap(initialCapacity, entryTypeSize, valueTypeSize);
   executionContext->registerState({hmm, [](void* ptr) { delete reinterpret_cast<runtime::HashMultiMap*>(ptr); }});
   return hmm;
}
void runtime::HashMultiMap::destroy(HashMultiMap* hashMultiMap) {
   delete hashMultiMap;
}
runtime::HashMultiMap::Entry* runtime::HashMultiMap::insertEntry(size_t hash) {
   if (entries.getLen() > hashMask / 2) {
      resize();
   }
   Entry* res = (Entry*) entries.insert();
   auto pos = hash & hashMask;
   auto* previousPtr = ht.at(pos);
   ht.at(pos) = runtime::tag(res, previousPtr, hash);
   res->next = runtime::untag(previousPtr);
   res->valueList = nullptr;
   res->hashValue = hash;
   return res;
}
runtime::HashMultiMap::Value* runtime::HashMultiMap::insertValue(Entry* entry) {
   Value* res = (Value*) values.insert();
   res->nextValue = entry->valueList;
   entry->valueList = res;
   return res;
}

void runtime::HashMultiMap::resize() {
   size_t oldHtSize = hashMask + 1;
   size_t newHtSize = oldHtSize * 2;
   ht.setNewSize(newHtSize);
   hashMask = newHtSize - 1;
   entries.iterate([&](uint8_t* entryRawPtr) {
      auto* entry = (Entry*) entryRawPtr;
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = runtime::tag(entry, previousPtr, entry->hashValue);
      entry->next = runtime::untag(previousPtr);
   });
}
runtime::BufferIterator* runtime::HashMultiMap::createIterator() {
   return entries.createIterator();
}