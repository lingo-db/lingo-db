#include "runtime/helpers.h"
#include "string.h"
#include <vector>
using i8 = uint8_t;
using i64 = uint64_t;
typedef bool (*compareFn)(i8*, i8*);
typedef void (*updateFn)(i8*, i8*);
struct SimpleHashTable {
   size_t payloadSize;
   static constexpr size_t initialSize = 8;
   struct Entry {
      uint64_t hash;
      size_t location;
      Entry* next;
      uint8_t data[];
   };
   SimpleHashTable(size_t objSize) : hashTable(initialSize, nullptr), buffer(sizeof(Entry) + objSize) {
      hashTableMask = initialSize - 1;
   }
   std::vector<Entry*> hashTable;
   runtime::ObjectBuffer<Entry> buffer;
   size_t hashTableMask;
   size_t entries;
   void resize() {
      std::vector<Entry*> newHashTable(hashTable.size() * 2, nullptr);
      hashTable = newHashTable;
      hashTableMask = hashTable.size() - 1;
      for (auto& entry : buffer) {
         size_t loc = entry.hash & hashTableMask;
         Entry* htBefore = hashTable[loc];
         hashTable[loc] = &entry;
         entry.next = htBefore;
      }
   }
   uint8_t* insert(uint64_t hash) {
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
   // EqualRangeIterator for the hash table
   class EqualRangeIterator : public std::iterator<std::forward_iterator_tag, Entry> {
      public:
      // Default constructor
      EqualRangeIterator() : curr(nullptr) {}
      // Constructor
      explicit EqualRangeIterator(Entry* curr, size_t hash) : curr(curr), hash(hash) {}
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
         curr = curr->next;
         while (curr != nullptr && curr->hash != hash) {
            curr = curr->next;
         }
         return *this;
      }
      // Reference
      Entry& operator*() { return *curr; }
      // Pointer
      Entry* operator->() { return curr; }
      // Equality
      bool operator==(const EqualRangeIterator& other) const { return curr == other.curr; }
      // Inequality
      bool operator!=(const EqualRangeIterator& other) const { return curr != other.curr; }

      bool hasNext() {
         return curr != nullptr;
      }

      protected:
      Entry* curr;
      size_t hash;
   };
   EqualRangeIterator end() { return EqualRangeIterator(); }
   EqualRangeIterator getIt(size_t hash) {
      size_t pos = hash & hashTableMask;
      Entry* curr = hashTable[pos];
      while (curr != nullptr && curr->hash != hash) {
         curr = curr->next;
      }
      return EqualRangeIterator(curr, hash);
   }
   inline Entry* lookup(size_t hash) {
      size_t pos = hash & hashTableMask;
      Entry* curr = hashTable[pos];
      while (curr != nullptr && curr->hash != hash) {
         curr = curr->next;
      }
      return curr;
   }

   // To find an element, calculate the hash (Key::Hash), and search this list until you reach a nullptr;
   std::pair<EqualRangeIterator, EqualRangeIterator> equal_range(size_t hash) {
      return std::make_pair(getIt(hash), EqualRangeIterator());
   }
   runtime::ObjectBuffer<Entry>& getBuffer() {
      return buffer;
   }
};
static std::string string_to_hex(const std::string& input) {
   static const char hex_digits[] = "0123456789ABCDEF";

   std::string output;
   output.reserve(input.length() * 2);
   for (unsigned char c : input) {
      output.push_back(hex_digits[c >> 4]);
      output.push_back(hex_digits[c & 15]);
   }
   return output;
}
struct AggrHashtableBuilder {
   std::string test = "abc";
   SimpleHashTable hashTable;
   size_t entries = 0;
   size_t keySize;
   size_t valSize;
   size_t aggrSize;
   size_t combinedSize;

   size_t padding;
   size_t aggrOffset;
   compareFn compareFunction;
   updateFn updateFunction;
   std::vector<uint8_t> initAggr;
   runtime::VarLenBuffer varLenBuffer;
   AggrHashtableBuilder(size_t keySize, size_t valSize, size_t aggrSize, size_t combinedSize, compareFn compareFunction, updateFn updateFunction, uint8_t* initAggr) : hashTable(combinedSize), keySize(keySize), valSize(valSize), aggrSize(aggrSize), combinedSize(combinedSize), compareFunction(compareFunction), updateFunction(updateFunction) {
      this->initAggr.insert(this->initAggr.end(), initAggr, initAggr + aggrSize);
      this->padding = combinedSize - keySize - aggrSize;
      this->aggrOffset = keySize + padding;
   }
   SimpleHashTable& getHashTable() {
      return hashTable;
   }
   inline bool matches(uint8_t* key1, uint8_t* key2) {
      if (keySize == 0) {
         return true;
      } else {
         return compareFunction((uint8_t*) key1,(uint8_t*) key2);
      }
   }
   inline void update(uint8_t* aggr, uint8_t* val) {
      if (aggrSize > 0) {
         updateFunction((uint8_t*) aggr,(uint8_t*) val);
      }
   }
   inline runtime::Pair<bool, uint8_t*> lookup(uint64_t hash) {
      auto entry = hashTable.lookup(hash);
      if (entry == nullptr) {
         return {false, nullptr};
      } else {
         return {true, entry->data};
      }
   }
   size_t currVarLen = 0;
   size_t unnecessaryAlloc = 0;
   void removeUneccesaryVarLen() {
      unnecessaryAlloc += currVarLen;
      currVarLen = 0;
   }

   std::string hex(uint8_t* bytes, size_t len) {
      return string_to_hex(std::string((char*) bytes, len));
   }
   inline void insert(uint64_t hash, uint8_t* key, uint8_t* val) {
      auto range = hashTable.equal_range(hash);
      for (auto i = range.first; i != range.second; ++i) {
         auto& x = *i;
         uint8_t* entryKey = x.data;
         uint8_t* entryAggr = x.data + aggrOffset;
         if (matches(key, entryKey)) {
            update(entryAggr, val);
            removeUneccesaryVarLen();
            return;
         }
      }
      //no match found!
      uint8_t* ptr = hashTable.insert(hash);
      memcpy(ptr, key, keySize);
      memcpy(ptr + aggrOffset, &initAggr[0], aggrSize);
      update(ptr + aggrOffset, val);
   }
};

EXPORT AggrHashtableBuilder* _mlir_ciface_aggr_ht_builder_create(uint64_t keySize, uint64_t valSize, uint64_t aggrSize, uint64_t combinedSize, compareFn compareFunction, updateFn updateFunction, uint8_t* bytes) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto* ptr = new AggrHashtableBuilder(keySize, valSize, aggrSize, combinedSize, compareFunction, updateFunction, bytes);
   return ptr;
}
EXPORT __attribute__((always_inline)) runtime::Bytes _mlir_ciface_aggr_ht_builder_add_var_len(AggrHashtableBuilder* builder, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->currVarLen += (data).getSize();
   return (builder)->varLenBuffer.persist(data);
}
EXPORT __attribute__((always_inline)) runtime::Pair<bool, runtime::Bytes> _mlir_ciface_aggr_ht_builder_add_nullable_var_len(AggrHashtableBuilder* builder, bool null, runtime::Bytes data) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   if (null) {
      return {true, runtime::Bytes(nullptr, 0)};
   }
   (builder)->currVarLen += (data).getSize();
   return {false, (builder)->varLenBuffer.persist(data)};
}
EXPORT __attribute__((always_inline)) runtime::Pair<bool, uint8_t*> _mlir_ciface_aggr_ht_builder_fast_lookup(AggrHashtableBuilder* builder, uint64_t hash) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return (builder)->lookup(hash);
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_aggr_ht_builder_merge(AggrHashtableBuilder* builder, uint64_t hash, uint8_t* key, uint8_t* val) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (builder)->insert(hash, key, val);
}
EXPORT __attribute__((always_inline)) AggrHashtableBuilder* _mlir_ciface_aggr_ht_builder_build(AggrHashtableBuilder* builder) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return builder;
}
using iterator_t = runtime::ObjectBuffer<SimpleHashTable::Entry>::RangeIterator;
EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_aggr_ht_iterator_init(AggrHashtableBuilder* hashmap) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto x = (hashmap)->getHashTable().getBuffer().beginPtr();
   return x;
}
EXPORT __attribute__((always_inline)) iterator_t* _mlir_ciface_aggr_ht_iterator_next(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   (iterator)->operator++();
   return iterator;
}
EXPORT __attribute__((always_inline)) uint8_t* _mlir_ciface_aggr_ht_iterator_curr(iterator_t* iterator) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   uint8_t* ptr = (*iterator)->data;
   return ptr;
}

EXPORT __attribute__((always_inline)) bool _mlir_ciface_aggr_ht_iterator_valid(iterator_t* iterator) {
   return iterator->valid();
}
EXPORT __attribute__((always_inline)) void _mlir_ciface_aggr_ht_iterator_free(iterator_t* iterator) {
   delete iterator;
}