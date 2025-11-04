#ifndef LINGODB_RUNTIME_HASHTABLE_H
#define LINGODB_RUNTIME_HASHTABLE_H
#include "ThreadLocal.h"
#include "lingodb/runtime/Buffer.h"
#include "lingodb/runtime/helpers.h"

namespace lingodb::runtime {
struct HashtableIterator;
class Hashtable {
   struct Entry {
      Entry* next;
      size_t hashValue;
      uint8_t content[];
      //kv follows
   };
   runtime::LegacyFixedSizedBuffer<Entry*> ht;
   size_t hashMask;
   runtime::FlexibleBuffer values;
   size_t typeSize;
   bool (*isEq)(uint8_t*, uint8_t*);

   Hashtable(size_t initialCapacity, size_t typeSize) : ht(initialCapacity * 2), hashMask(initialCapacity * 2 - 1), values(initialCapacity, typeSize), typeSize(typeSize) {}

   public:
   void resize();
   void setEqFn(bool (*isEq)(uint8_t*, uint8_t*));
   Entry* insert(size_t hash);
   bool contains(size_t hash, uint8_t* keyVal);
   uint8_t* lookup(size_t hash, uint8_t* keyVal);
   uint8_t* lookUpOrInsert(size_t hash, uint8_t* keyVal);
   static Hashtable* create(size_t typeSize, size_t initialCapacity);
   static void destroy(Hashtable*);
   size_t size();
   void mergeEntries(bool (*isEq)(uint8_t*, uint8_t*), void (*merge)(uint8_t*, uint8_t*), Hashtable* other);
   static runtime::Hashtable* merge(ThreadLocal*, bool (*eq)(uint8_t*, uint8_t*), void (*combine)(uint8_t*, uint8_t*));

   runtime::BufferIterator* createIterator();

   static HashtableIterator* createHtIterator(Hashtable* ht);

   friend struct HashtableIterator;
};
//allows iterating over the single elements of a hashtable using the BufferIterator interface
struct HashtableIterator {
   runtime::BufferIterator* bufferIt;
   size_t typeSize;
   size_t currPosInBuffer;
   runtime::Buffer currBuffer;
   bool valid;
   HashtableIterator(runtime::BufferIterator* bufferIt, size_t typeSize)
      : bufferIt(bufferIt), typeSize(typeSize), currPosInBuffer(0), currBuffer({0, 0}), valid(false) {
      if (bufferIt->isValid()) {
         currBuffer = bufferIt->getCurrentBuffer();
         valid = currPosInBuffer * typeSize < currBuffer.numElements;
      } else {
         valid = false;
      }
   }
   bool isValid();
   void next();
   uint8_t* getCurrent();
};

} // end namespace lingodb::runtime
#endif // LINGODB_RUNTIME_HASHTABLE_H
