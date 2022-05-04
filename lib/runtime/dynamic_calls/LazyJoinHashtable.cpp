#include "runtime/LazyJoinHashtable.h"

void runtime::LazyJoinHashtable::resize() {
   values.resize();
}
void runtime::LazyJoinHashtable::finalize() {
   size_t htSize = std::max(nextPow2(values.getLen()), 1ul);
   htMask = htSize - 1;
   ht.setNewSize(htSize);
   for (size_t i = 0; i < values.getLen(); i++) {
      auto* entry = values.ptrAt<Entry>(i);
      size_t hash = (size_t) entry->next;
      auto pos = hash & htMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = runtime::tag(entry, previousPtr, hash);
      entry->next = previousPtr;
   }
}
runtime::LazyJoinHashtable* runtime::LazyJoinHashtable::create(size_t typeSize) {
   return new LazyJoinHashtable(1024, typeSize);
}
void runtime::LazyJoinHashtable::destroy(LazyJoinHashtable* ht) {
   delete ht;
}