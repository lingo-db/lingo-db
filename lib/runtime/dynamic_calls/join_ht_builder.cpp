#include "runtime/helpers.h"
#include "string.h"
#include <iostream>
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
      std::byte data[];
   };
   // EqualRangeIterator for the hash table
   class EqualRangeIterator : public std::iterator<std::forward_iterator_tag, Entry> {
      public:
      // Default constructor
      EqualRangeIterator() : ptr_(nullptr) {}
      // Constructor
      explicit EqualRangeIterator(Entry* ptr) : ptr_(ptr) {}
      // Destructor
      ~EqualRangeIterator() = default;

      // Postfix increment
      EqualRangeIterator operator++(int) {
         EqualRangeIterator copy = *this;
         this->operator++();
         return copy;
      }
      // Prefix increment
      EqualRangeIterator& operator++() {
         ptr_ = ptr_->next;
         return *this;
      }
      // Reference
      Entry& operator*() { return *ptr_; }
      // Pointer
      Entry* operator->() { return ptr_; }
      // Equality
      bool operator==(const EqualRangeIterator& other) const { return ptr_ == other.ptr_; }
      // Inequality
      bool operator!=(const EqualRangeIterator& other) const { return ptr_ != other.ptr_; }

      bool hasNext() {
         return ptr_ != nullptr;
      }

      protected:
      // Entry pointer
      Entry* ptr_;
   };

   public:
   // End of the equal range
   EqualRangeIterator end() { return EqualRangeIterator(); }

   LazyMultiMap(size_t dataSize) : dataSize(dataSize) {}

   // Insert an element into the hash table
   //  * Gather all entries with insert and build the hash table with finalize.

   // Finalize the hash table
   //  * Get the next power of two, which is larger than the number of entries (include/imlab/infra/bits.h).
   //  * Resize the hash table to that size.
   //  * For each entry in entries_, calculate the hash and prepend it to the collision list in the hash table.
   void finalize() {
      size_t ht_size = entries_.empty() ? 1 : NextPow2_64(entries_.size()/dataSize);
      hash_table_mask_ = ht_size - 1;
      hash_table_.resize(ht_size);
      for (size_t i = 0; i < entries_.size(); i += dataSize) {
         Entry* ptr = (Entry*) &entries_[i];
         size_t hash = (size_t)ptr->next;
         size_t pos = hash & hash_table_mask_;

         Entry* currentEntry = hash_table_[pos];
         ptr->next = currentEntry;
         hash_table_[pos] = ptr;
      }
   }
   EqualRangeIterator getIt(size_t hash) {
      size_t pos = hash & hash_table_mask_;
      Entry* curr = hash_table_[pos];
      return EqualRangeIterator(curr);
   }
   // To find an element, calculate the hash (Key::Hash), and search this list until you reach a nullptr;
   std::pair<EqualRangeIterator, EqualRangeIterator> equal_range(size_t hash) {
      return std::make_pair(getIt(hash), EqualRangeIterator());
   }
   std::vector<std::byte> rawData;
   // Entries of the hash table.
   std::vector<std::byte> entries_;
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
EXPORT runtime::Triple<bool, uint64_t, uint64_t> _mlir_ciface_join_ht_builder_add_nullable_var_len(runtime::Pointer<LazyMultiMap>* builder, bool null, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, 0, 0};
   }
   auto& rawValues = (*builder)->rawData;
   size_t sizeBefore = rawValues.size();
   rawValues.resize(sizeBefore + (*data).getSize());
   memcpy(&rawValues[sizeBefore], (*data).getPtr(), (*data).getSize());
   return {false, sizeBefore, (*data).getSize()};
}
EXPORT void _mlir_ciface_join_ht_builder_add_var_len(runtime::Pair<uint64_t, uint64_t>* result, runtime::Pointer<LazyMultiMap>* builder, runtime::ByteRange* data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& rawValues = (*builder)->rawData;
   size_t sizeBefore = rawValues.size();
   rawValues.resize(sizeBefore + (*data).getSize());
   memcpy(&rawValues[sizeBefore], (*data).getPtr(), (*data).getSize());
   *result = {sizeBefore, (*data).getSize()};
}

EXPORT runtime::Pointer<std::byte> _mlir_ciface_join_ht_builder_merge(runtime::Pointer<LazyMultiMap>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto& values = (*builder)->entries_;
   size_t sizeBefore = values.size();
   values.resize(sizeBefore + (*builder)->dataSize);
   std::byte* ptr = &values[sizeBefore];
   return ptr;
}
EXPORT runtime::Pointer<LazyMultiMap> _mlir_ciface_join_ht_builder_build(runtime::Pointer<LazyMultiMap>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (*builder)->finalize();
   return *builder;
}
EXPORT runtime::ByteRange _mlir_ciface_join_ht_get_raw_data(runtime::Pointer<LazyMultiMap>* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return runtime::ByteRange((uint8_t*) &(*builder)->rawData[0], (*builder)->rawData.size());
}

EXPORT runtime::Pointer<LazyMultiMap::EqualRangeIterator> _mlir_ciface_join_ht_iterator_init(runtime::Pointer<LazyMultiMap>* hashmap, size_t hash) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return new LazyMultiMap::EqualRangeIterator((*hashmap)->getIt(hash));
}
EXPORT runtime::Pointer<LazyMultiMap::EqualRangeIterator> _mlir_ciface_join_ht_iterator_next(runtime::Pointer<LazyMultiMap::EqualRangeIterator>* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (*iterator)->operator++();
   return *iterator;
}
EXPORT runtime::Pointer<std::byte> _mlir_ciface_join_ht_iterator_curr(runtime::Pointer<LazyMultiMap::EqualRangeIterator>* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return &(*iterator)->operator*().data[0];
}

EXPORT bool _mlir_ciface_join_ht_iterator_valid(runtime::Pointer<LazyMultiMap::EqualRangeIterator>* iterator) {
   return (*iterator)->hasNext();
}
EXPORT void _mlir_ciface_join_ht_iterator_free(runtime::Pointer<LazyMultiMap::EqualRangeIterator>* iterator) {
   return delete (*iterator).get();
}
