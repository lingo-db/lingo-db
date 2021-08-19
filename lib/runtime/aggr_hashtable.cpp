#include "runtime/aggr_hashtable.h"

runtime::AggrHashtableBuilder::AggrHashtableBuilder(size_t keySize, size_t valSize, size_t aggrSize, size_t combinedSize, compareFn compareFunction, updateFn updateFunction, uint8_t* initAggr) : hashTable(combinedSize), keySize(keySize), valSize(valSize), aggrSize(aggrSize), combinedSize(combinedSize), compareFunction(compareFunction), updateFunction(updateFunction) {
   this->initAggr.insert(this->initAggr.end(), initAggr, initAggr + aggrSize);
   this->padding = combinedSize - keySize - aggrSize;
   this->aggrOffset = keySize + padding;
}

runtime::SimpleHashTable::SimpleHashTable(size_t objSize) : hashTable(initialSize), buffer(sizeof(Entry) + objSize), entries(0) {
   hashTable.zero();
   hashTableMask = initialSize - 1;
}
void runtime::SimpleHashTable::resize() {
   hashTable.resize(hashTable.size() * 2);
   hashTable.zero();
   hashTableMask = hashTable.size() - 1;
   for (auto& entry : buffer) {
      size_t loc = entry.hash & hashTableMask;
      Entry* htBefore = hashTable[loc];
      hashTable[loc] = &entry;
      entry.next = htBefore;
   }
}
uint8_t* runtime::SimpleHashTable::insert(uint64_t hash) {
   entries++;
   double loadFactor = entries / (double) hashTable.size();
   if (loadFactor > 0.5) {
      resize();
   }
   size_t loc = hash & hashTableMask;
   Entry* htBefore = hashTable[loc];
   Entry* newEntry = buffer.alloc();
   newEntry->next = htBefore;
   hashTable[loc] = newEntry;
   newEntry->hash = hash;
   return newEntry->data;
}
runtime::AggrHashtableBuilder::~AggrHashtableBuilder() {}
