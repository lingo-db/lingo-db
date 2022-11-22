#ifndef RUNTIME_HASHTABLE_H
#define RUNTIME_HASHTABLE_H
#include "runtime/Buffer.h"
#include "runtime/helpers.h"
namespace runtime {
class Hashtable {
   struct Entry {
      Entry* next;
      size_t hashValue;
      uint8_t content[];
      //kv follows
   };
   runtime::FixedSizedBuffer<Entry*> ht;
   size_t hashMask;
   runtime::FlexibleBuffer values;
   //initial value follows...
   Hashtable(size_t initialCapacity, size_t typeSize) : ht(initialCapacity * 2), hashMask(initialCapacity * 2 - 1), values(initialCapacity, typeSize) {}

   public:
   void resize();
   Entry* insert(size_t hash);
   static Hashtable* create(size_t typeSize, size_t initialCapacity);
   static void destroy(Hashtable*);
   runtime::BufferIterator* createIterator();
};

} // end namespace runtime
#endif // RUNTIME_HASHTABLE_H
