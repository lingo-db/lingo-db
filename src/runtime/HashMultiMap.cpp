#include "lingodb/runtime/HashMultiMap.h"

lingodb::runtime::HashMultiMap* lingodb::runtime::HashMultiMap::create(lingodb::runtime::ExecutionContext* executionContext,size_t entryTypeSize, size_t valueTypeSize, size_t initialCapacity) {
   auto* hmm= new HashMultiMap(initialCapacity, entryTypeSize, valueTypeSize);
   executionContext->registerState({hmm, [](void* ptr) { delete reinterpret_cast<lingodb::runtime::HashMultiMap*>(ptr); }});
   return hmm;
}
void lingodb::runtime::HashMultiMap::destroy(HashMultiMap* hashMultiMap) {
   delete hashMultiMap;
}
lingodb::runtime::HashMultiMap::Entry* lingodb::runtime::HashMultiMap::insertEntry(size_t hash) {
   if (entries.getLen() > hashMask / 2) {
      resize();
   }
   Entry* res = (Entry*) entries.insert();
   auto pos = hash & hashMask;
   auto* previousPtr = ht.at(pos);
   ht.at(pos) = lingodb::runtime::tag(res, previousPtr, hash);
   res->next = lingodb::runtime::untag(previousPtr);
   res->valueList = nullptr;
   res->hashValue = hash;
   return res;
}
lingodb::runtime::HashMultiMap::Value* lingodb::runtime::HashMultiMap::insertValue(Entry* entry) {
   Value* res = (Value*) values.insert();
   res->nextValue = entry->valueList;
   entry->valueList = res;
   return res;
}

void lingodb::runtime::HashMultiMap::resize() {
   size_t oldHtSize = hashMask + 1;
   size_t newHtSize = oldHtSize * 2;
   ht.setNewSize(newHtSize);
   hashMask = newHtSize - 1;
   entries.iterate([&](uint8_t* entryRawPtr) {
      auto* entry = (Entry*) entryRawPtr;
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = lingodb::runtime::tag(entry, previousPtr, entry->hashValue);
      entry->next = lingodb::runtime::untag(previousPtr);
   });
}
lingodb::runtime::BufferIterator* lingodb::runtime::HashMultiMap::createIterator() {
   return entries.createIterator();
}