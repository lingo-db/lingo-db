#ifndef RUNTIME_AGGR_HASHTABLE_H
#define RUNTIME_AGGR_HASHTABLE_H
#include "runtime/helpers.h"
using i8 = uint8_t;
using i64 = uint64_t;
typedef bool (*compareFn)(i8*, i8*);
typedef void (*updateFn)(i8*, i8*);
namespace runtime {
struct SimpleHashTable {
   static constexpr size_t initialSize = 8;
   struct Entry {
      uint64_t hash;
      size_t location;
      Entry* next;
      uint8_t data[];
   };
   SimpleHashTable(size_t objSize);
   runtime::Vec<Entry*> hashTable;
   runtime::ObjectBuffer<Entry> buffer;
   size_t hashTableMask;
   size_t entries;
   void resize();
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
   std::pair<EqualRangeIterator, EqualRangeIterator> equalRange(size_t hash) {
      return std::make_pair(getIt(hash), EqualRangeIterator());
   }
   runtime::ObjectBuffer<Entry>& getBuffer() {
      return buffer;
   }
};

struct AggrHashtableBuilder {
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
   AggrHashtableBuilder(size_t keySize, size_t valSize, size_t aggrSize, size_t combinedSize, compareFn compareFunction, updateFn updateFunction, uint8_t* initAggr);
   SimpleHashTable& getHashTable() {
      return hashTable;
   }
   inline bool matches(uint8_t* key1, uint8_t* key2) {
      if (keySize == 0) {
         return true;
      } else {
         return compareFunction((uint8_t*) key1, (uint8_t*) key2);
      }
   }
   inline void update(uint8_t* aggr, uint8_t* val) {
      if (aggrSize > 0) {
         updateFunction((uint8_t*) aggr, (uint8_t*) val);
      }
   }
   inline runtime::Pair<bool, uint8_t*> lookup(uint64_t hash) {
      auto* entry = hashTable.lookup(hash);
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


   inline void insert(uint64_t hash, uint8_t* key, uint8_t* val) {
      auto range = hashTable.equalRange(hash);
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
} // namespace runtime
#endif // RUNTIME_AGGR_HASHTABLE_H
