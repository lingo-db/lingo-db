#ifndef RUNTIME_JOIN_HASHTABLES_H
#define RUNTIME_JOIN_HASHTABLES_H
#include "runtime/helpers.h"
namespace runtime {
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

   LazyMultiMap(size_t dataSize);

   // Insert an element into the hash table
   //  * Gather all entries with insert and build the hash table with finalize.

   // Finalize the hash table
   //  * Get the next power of two, which is larger than the number of entries (include/imlab/infra/bits.h).
   //  * Resize the hash table to that size.
   //  * For each entry in entries_, calculate the hash and prepend it to the collision list in the hash table.
   void finalize();
   inline Entry* getIt(size_t hash) {
      size_t pos = hash & hash_table_mask_;
      Entry* curr = hash_table_[pos];
      return curr;
   }

   runtime::VarLenBuffer varLenBuffer;
   // Entries of the hash table.
   runtime::ObjectBuffer<Entry> entryBuffer;
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
   runtime::Vec<Entry*> hash_table_;
   // The hash table mask.
   uint32_t hash_table_mask_;
};
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

   MarkableLazyMultiMap(size_t dataSize);

   // Insert an element into the hash table
   //  * Gather all entries with insert and build the hash table with finalize.

   // Finalize the hash table
   //  * Get the next power of two, which is larger than the number of entries (include/imlab/infra/bits.h).
   //  * Resize the hash table to that size.
   //  * For each entry in entries_, calculate the hash and prepend it to the collision list in the hash table.
   void finalize();
   inline Entry* getIt(size_t hash) {
      size_t pos = hash & hash_table_mask_;
      Entry* curr = hash_table_[pos];
      return curr;
   }

   runtime::VarLenBuffer varLenBuffer;
   // Entries of the hash table.
   runtime::ObjectBuffer<Entry> entryBuffer;
   runtime::ObjectBuffer<Entry>& getBuffer() {
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
   runtime::Vec<Entry*> hash_table_;
   // The hash table mask.
   uint32_t hash_table_mask_;
};
}
#endif //RUNTIME_JOIN_HASHTABLES_H
