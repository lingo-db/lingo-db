#include "runtime/Hashtable.h"
runtime::Hashtable* runtime::Hashtable::create(size_t typeSize, size_t initialCapacity) {
   return new (malloc(sizeof(Hashtable) + typeSize)) Hashtable(initialCapacity, typeSize);
}
void runtime::Hashtable::resize() {
   size_t oldHtSize = hashMask+1;
   size_t newHtSize = oldHtSize * 2;
   ht.setNewSize(newHtSize);
   hashMask = newHtSize - 1;
   for(auto buffer:values.buffers){
      for(size_t i=0;i<buffer.numElements;i++){
         auto* entry = (Entry*) &buffer.ptr[i*values.typeSize];
         auto pos = entry->hashValue & hashMask;
         auto* previousPtr = ht.at(pos);
         ht.at(pos) = entry;
         entry->next = previousPtr;
      }
   }
}
void runtime::Hashtable::destroy(runtime::Hashtable* ht) {
   ht->~Hashtable();
   free(ht);
}
runtime::Hashtable::Entry* runtime::Hashtable::insert(size_t hash) {
   if (values.totalLen > hashMask / 2) {
      resize();
   }
   Entry* res = (Entry*) values.insert();
   auto pos = hash & hashMask;
   auto* previousPtr = ht.at(pos);
   res->next=previousPtr;
   res->hashValue=hash;
   ht.at(pos) = res;
   return res;
}
using Iterator = runtime::Hashtable::Iterator;

Iterator* runtime::Hashtable::startIteration() {
   return new Iterator;
}
void runtime::Hashtable::endIteration(Iterator* it) {
   delete it;
}
bool runtime::Hashtable::isIteratorValid(Iterator* it) {
   return it->currBuffer < values.buffers.size();
}
void runtime::Hashtable::nextIterator(Iterator* it) {
   it->currBuffer++;
}
runtime::Buffer runtime::Hashtable::getCurrentBuffer(Iterator* it) {
   Buffer orig = values.buffers[it->currBuffer];
   return Buffer{orig.numElements * values.typeSize, orig.ptr};
}