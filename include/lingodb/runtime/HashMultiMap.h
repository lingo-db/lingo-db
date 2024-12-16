#ifndef LINGODB_RUNTIME_HASHMULTIMAP_H
#define LINGODB_RUNTIME_HASHMULTIMAP_H
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/helpers.h"
namespace lingodb::runtime {
class HashMultiMap {
   struct Value{
      Value* nextValue;
      uint8_t valueContent[];
   };
   struct Entry {
      Entry* next; //for resolving collisions
      size_t hashValue;
      Value* valueList;
      uint8_t keyContent[];
   };
   runtime::LegacyFixedSizedBuffer<Entry*> ht;
   size_t hashMask;
   runtime::FlexibleBuffer entries;
   runtime::FlexibleBuffer values;

   //initial value follows...
   HashMultiMap(size_t initialCapacity, size_t entryTypeSize, size_t valueTypeSize) : ht(initialCapacity * 2), hashMask(initialCapacity * 2 - 1), entries(initialCapacity,entryTypeSize),values(initialCapacity,valueTypeSize) {}

   public:
   void resize();
   Entry* insertEntry(size_t hash);
   Value* insertValue(Entry* entry);
   static HashMultiMap* create(runtime::ExecutionContext* executionContext,size_t entryTypeSize, size_t valueTypeSize, size_t initialCapacity);
   static void destroy(HashMultiMap*);
   runtime::BufferIterator* createIterator();
};

} // end namespace lingodb::runtime
#endif // LINGODB_RUNTIME_HASHMULTIMAP_H
