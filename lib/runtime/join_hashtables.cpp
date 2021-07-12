#include "runtime/join_hashtables.h"
runtime::MarkableLazyMultiMap::MarkableLazyMultiMap(size_t dataSize) : entryBuffer(sizeof(Entry) + dataSize), dataSize(dataSize), entries(0) {}
runtime::LazyMultiMap::LazyMultiMap(size_t dataSize) : entryBuffer(sizeof(Entry) + dataSize), dataSize(dataSize), entries(0) {}

void runtime::LazyMultiMap::finalize() {
   size_t htSize = entries == 0 ? 1 : nextPow2(entries);
   hashTableMask = htSize - 1;
   hashTable.resize(htSize);
   hashTable.zero();
   for (auto& e : entryBuffer) {
      Entry* ptr = (Entry*) &e;
      size_t hash = (size_t) ptr->next;
      size_t pos = hash & hashTableMask;

      Entry* currentEntry = hashTable[pos];
      ptr->next = currentEntry;
      hashTable[pos] = ptr;
   }
}
void runtime::MarkableLazyMultiMap::finalize() {
   size_t htSize = entries == 0 ? 1 : nextPow2(entries);
   hashTableMask = htSize - 1;
   hashTable.resize(htSize);
   hashTable.zero();

   for (auto& e : entryBuffer) {
      Entry* ptr = (Entry*) &e;
      size_t hash = (size_t) ptr->next;
      size_t pos = hash & hashTableMask;
      ptr->mark = 0;
      Entry* currentEntry = hashTable[pos];
      ptr->next = currentEntry;
      hashTable[pos] = ptr;
   }
}