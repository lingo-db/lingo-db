#include "runtime/helpers.h"
#include "string.h"
#include <vector>

class LazyMultiMap {
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
      Entry* next;
      uint8_t data[];
   };

   public:
   // End of the equal range

   LazyMultiMap(size_t dataSize) : dataSize(dataSize) {}

   // Insert an element into the hash table
   //  * Gather all entries with insert and build the hash table with finalize.

   // Finalize the hash table
   //  * Get the next power of two, which is larger than the number of entries (include/imlab/infra/bits.h).
   //  * Resize the hash table to that size.
   //  * For each entry in entries_, calculate the hash and prepend it to the collision list in the hash table.
   void finalize() {
      size_t ht_size = entries_.empty() ? 1 : NextPow2_64(entries_.size() / dataSize);
      hash_table_mask_ = ht_size - 1;
      hash_table_.resize(ht_size);
      for (size_t i = 0; i < entries_.size(); i += dataSize) {
         Entry* ptr = (Entry*) &entries_[i];
         size_t hash = (size_t) ptr->next;
         size_t pos = hash & hash_table_mask_;

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
   std::vector<uint8_t> entries_;
   size_t dataSize;

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

EXPORT runtime::Pointer<LazyMultiMap> _mlir_ciface_join_ht_builder_create(size_t dataSize) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new LazyMultiMap(dataSize);
}
EXPORT __attribute__((always_inline)) runtime::ByteRange _mlir_ciface_join_ht_builder_add_var_len(runtime::Pointer<LazyMultiMap>* builder, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*builder)->varLenBuffer.persist(*data);
}
EXPORT __attribute__((always_inline)) runtime::Pair<bool, runtime::ByteRange> _mlir_ciface_join_ht_builder_add_nullable_var_len(runtime::Pointer<LazyMultiMap>* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::ByteRange(nullptr, 0)};
   }
   return {false, (*builder)->varLenBuffer.persist(*data)};
}

EXPORT __attribute__((always_inline)) runtime::Pointer<uint8_t> _mlir_ciface_join_ht_builder_merge(runtime::Pointer<LazyMultiMap>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& values = (*builder)->entries_;
   size_t sizeBefore = values.size();
   values.resize(sizeBefore + (*builder)->dataSize);
   return &values[sizeBefore];
}
EXPORT runtime::Pointer<LazyMultiMap> _mlir_ciface_join_ht_builder_build(runtime::Pointer<LazyMultiMap>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (*builder)->finalize();
   return *builder;
}

EXPORT __attribute__((always_inline)) runtime::Pointer<LazyMultiMap::Entry> _mlir_ciface_join_ht_iterator_init(runtime::Pointer<LazyMultiMap>* hashmap, size_t hash) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*hashmap)->getIt(hash);
}
EXPORT __attribute__((always_inline)) runtime::Pointer<LazyMultiMap::Entry> _mlir_ciface_join_ht_iterator_next(runtime::Pointer<LazyMultiMap::Entry>* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*iterator)->next;
}
EXPORT __attribute__((always_inline)) runtime::Pointer<uint8_t> _mlir_ciface_join_ht_iterator_curr(runtime::Pointer<LazyMultiMap::Entry>* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (*iterator)->data;
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_join_ht_iterator_valid(runtime::Pointer<LazyMultiMap::Entry>* iterator) {
   return (*iterator).get() != nullptr;
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_join_ht_iterator_free(runtime::Pointer<LazyMultiMap::Entry>* iterator) {
}
