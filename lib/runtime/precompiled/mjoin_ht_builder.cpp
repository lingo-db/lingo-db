#include "runtime/helpers.h"
#include "string.h"
#include <vector>

class MarkableLazyMultiMap {
   inline uint64_t NextPow2_64(uint64_t v) {
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v |= v >> 32;
      v++;
      return v;
   }

   public:
   // Entry in the hash table
   struct Entry {
      // Pointer to the next element in the collision list
      uint8_t mark;
      Entry* next;
      uint8_t data[];
   };

   public:
   // End of the equal range

   MarkableLazyMultiMap(size_t dataSize) : dataSize(dataSize), entries(0), entryBuffer(sizeof(Entry) + dataSize) {}

   // Insert an element into the hash table
   //  * Gather all entries with insert and build the hash table with finalize.

   // Finalize the hash table
   //  * Get the next power of two, which is larger than the number of entries (include/imlab/infra/bits.h).
   //  * Resize the hash table to that size.
   //  * For each entry in entries_, calculate the hash and prepend it to the collision list in the hash table.
   void finalize() {
      size_t ht_size = entries == 0 ? 1 : NextPow2_64(entries);
      hash_table_mask_ = ht_size - 1;
      hash_table_.resize(ht_size);
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
   inline Entry* getIt(size_t hash) {
      size_t pos = hash & hash_table_mask_;
      Entry* curr = hash_table_[pos];
      return curr;
   }

   runtime::VarLenBuffer varLenBuffer;
   // Entries of the hash table.
   runtime::ObjectBuffer<Entry> entryBuffer;
   runtime::ObjectBuffer<Entry>& getBuffer(){
      return entryBuffer;
   }
   size_t dataSize;
   size_t entries;

   protected:
   // The hash table.
   // Use the next_ pointers in the entries to store the collision list of the hash table.
   //
   //      hash_table_     entries_
   //      +---+           +---+
   //      | * | --------> | x | --+
   //      | 0 |           |   |   |
   //      | 0 |           |   |   |
   //      | 0 |           | z | <-+
   //      +---+           +---+
   //
   std::vector<Entry*> hash_table_;
   // The hash table mask.
   uint32_t hash_table_mask_;
};

EXPORT MarkableLazyMultiMap* _mlir_ciface_mjoin_ht_builder_create(size_t dataSize) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new MarkableLazyMultiMap(dataSize);
}
EXPORT __attribute__((always_inline)) runtime::Bytes _mlir_ciface_mjoin_ht_builder_add_var_len(MarkableLazyMultiMap* builder, runtime::Bytes* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (builder)->varLenBuffer.persist(*data);
}
EXPORT __attribute__((always_inline)) runtime::Pair<bool, runtime::ByteRange> _mlir_ciface_mjoin_ht_builder_add_nullable_var_len(MarkableLazyMultiMap* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::ByteRange(nullptr, 0)};
   }
   return {false, (builder)->varLenBuffer.persist(*data)};
}

EXPORT __attribute__((always_inline)) uint8_t* _mlir_ciface_mjoin_ht_builder_merge(MarkableLazyMultiMap* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->entries++;
   return (uint8_t*) &(builder)->entryBuffer.alloc()->next;
}
EXPORT MarkableLazyMultiMap* _mlir_ciface_mjoin_ht_builder_build(MarkableLazyMultiMap* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->finalize();
   return builder;
}
//for lookups
EXPORT __attribute__((always_inline)) MarkableLazyMultiMap::Entry* _mlir_ciface_mjoin_ht_lookup_iterator_init(MarkableLazyMultiMap* hashmap, size_t hash) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (hashmap)->getIt(hash);
}
EXPORT __attribute__((always_inline)) MarkableLazyMultiMap::Entry* _mlir_ciface_mjoin_ht_lookup_iterator_next(MarkableLazyMultiMap::Entry* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return iterator->next;
}
EXPORT __attribute__((always_inline)) runtime::Pair<uint8_t*,uint8_t*> _mlir_ciface_mjoin_ht_lookup_iterator_curr(MarkableLazyMultiMap::Entry* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return {(iterator)->data,&(iterator)->mark};
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_mjoin_ht_lookup_iterator_valid(MarkableLazyMultiMap::Entry* iterator) {
   return iterator != nullptr;
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_mjoin_ht_lookup_iterator_free(MarkableLazyMultiMap::Entry* iterator) {
}
using iterator_t = runtime::ObjectBuffer<MarkableLazyMultiMap::Entry>::RangeIterator;

EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_mjoin_ht_iterator_init(MarkableLazyMultiMap* hashmap) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (hashmap)->getBuffer().beginPtr();
}
EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_mjoin_ht_iterator_next(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   iterator->operator++();
   return iterator;
}
EXPORT __attribute__((always_inline)) runtime::Pair<uint8_t*,uint8_t*> _mlir_ciface_mjoin_ht_iterator_curr(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return {(*iterator)->data,&(*iterator)->mark};
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_mjoin_ht_iterator_valid(iterator_t* iterator) {
   return iterator->valid();
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_mjoin_ht_iterator_free(iterator_t* iterator) {
   delete iterator;
}