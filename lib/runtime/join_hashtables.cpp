#include "runtime/join_hashtables.h"
runtime::MarkableLazyMultiMap::MarkableLazyMultiMap(size_t dataSize) : dataSize(dataSize), entries(0), entryBuffer(sizeof(Entry) + dataSize) {}
runtime::LazyMultiMap::LazyMultiMap(size_t dataSize) : dataSize(dataSize), entries(0), entryBuffer(sizeof(Entry) + dataSize) {}

void runtime::LazyMultiMap::finalize() {
   size_t ht_size = entries == 0 ? 1 : NextPow2_64(entries);
   hash_table_mask_ = ht_size - 1;
   hash_table_.resize(ht_size);
   hash_table_.zero();
   for (auto& e : entryBuffer) {
      Entry* ptr = (Entry*) &e;
      size_t hash = (size_t) ptr->next;
      size_t pos = hash & hash_table_mask_;

      Entry* currentEntry = hash_table_[pos];
      ptr->next = currentEntry;
      hash_table_[pos] = ptr;
   }
}
void runtime::MarkableLazyMultiMap::finalize() {
   size_t ht_size = entries == 0 ? 1 : NextPow2_64(entries);
   hash_table_mask_ = ht_size - 1;
   hash_table_.resize(ht_size);
   hash_table_.zero();

   for (auto& e : entryBuffer) {
      Entry* ptr = (Entry*) &e;
      size_t hash = (size_t) ptr->next;
      size_t pos = hash & hash_table_mask_;
      ptr->mark = 0;
      Entry* currentEntry = hash_table_[pos];
      ptr->next = currentEntry;
      hash_table_[pos] = ptr;
   }
}