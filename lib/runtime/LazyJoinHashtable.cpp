#include "runtime/LazyJoinHashtable.h"
#include "runtime/GrowingBuffer.h"

runtime::HashIndexedView* runtime::HashIndexedView::build(runtime::GrowingBuffer* buffer) {
   auto& values=buffer->getValues();
   size_t htSize = std::max(nextPow2(values.getLen() * 1.25), 1ul);
   size_t htMask = htSize - 1;
   auto *htView= new HashIndexedView(htSize,htMask);
   values.iterate([&](uint8_t* ptr) {
      auto *entry = (Entry*) ptr;
      size_t hash = (size_t) entry->hashValue;
      auto pos = hash & htMask;
      auto* previousPtr = htView->ht.at(pos);
      htView->ht.at(pos) = runtime::tag(entry, previousPtr, hash);
      entry->next = previousPtr;
   });
   return htView;
}
void runtime::HashIndexedView::destroy(runtime::HashIndexedView* ht) {
   delete ht;
}