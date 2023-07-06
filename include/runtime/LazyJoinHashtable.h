#ifndef RUNTIME_LAZYJOINHASHTABLE_H
#define RUNTIME_LAZYJOINHASHTABLE_H
#include "runtime/Buffer.h"
#include "runtime/helpers.h"
namespace runtime {
class GrowingBuffer;
class HashIndexedView {
   struct Entry {
      Entry* next;
      uint64_t hashValue;
      //kv follows
   };
   Entry** ht;
   size_t htMask; //NOLINT(clang-diagnostic-unused-private-field)
   HashIndexedView(size_t htSize,size_t htMask);
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
   static HashIndexedView* build(runtime::ExecutionContext* executionContext,GrowingBuffer* buffer);
   static void destroy(HashIndexedView*);
   ~HashIndexedView();
};
} // end namespace runtime
#endif // RUNTIME_LAZYJOINHASHTABLE_H
