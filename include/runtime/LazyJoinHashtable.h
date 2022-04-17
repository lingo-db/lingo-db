#ifndef RUNTIME_LAZYJOINHASHTABLE_H
#define RUNTIME_LAZYJOINHASHTABLE_H
#include "runtime/Vector.h"
#include "runtime/helpers.h"
namespace runtime {
class LazyJoinHashtable {
   struct Entry {
      Entry* next;
      //kv follows
   };
   runtime::FixedSizedBuffer<Entry*> ht;
   size_t htMask;
   runtime::Vector values;
   LazyJoinHashtable(size_t initial, size_t typeSize) : ht(0), htMask(0), values(initial, typeSize) {}
   static uint64_t nextPow2(uint64_t v) {
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
   static LazyJoinHashtable* create(size_t typeSize);
   void finalize();
   void resize();
   static void destroy(LazyJoinHashtable*);
};
} // end namespace runtime
#endif // RUNTIME_LAZYJOINHASHTABLE_H
