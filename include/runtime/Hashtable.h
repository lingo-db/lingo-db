#ifndef RUNTIME_HASHTABLE_H
#define RUNTIME_HASHTABLE_H
#include "runtime/Vector.h"
#include "runtime/helpers.h"
namespace runtime {
class Hashtable {
   struct Entry {
      Entry* next;
      size_t hashValue;
      //kv follows
   };
   runtime::FixedSizedBuffer<Entry*> ht;
   runtime::Vector values;
   //initial value follows...
   Hashtable(size_t initialCapacity, size_t typeSize) : ht(initialCapacity * 2), values(initialCapacity, typeSize) {}

   public:
   void resize();
   static Hashtable* create(size_t typeSize, size_t initialCapacity);
   static void destroy(Hashtable*);
};

} // end namespace runtime
#endif // RUNTIME_HASHTABLE_H
