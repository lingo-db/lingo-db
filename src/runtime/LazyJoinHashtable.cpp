#include "lingodb/runtime/LazyJoinHashtable.h"
#include "lingodb/runtime/GrowingBuffer.h"
#include "lingodb/utility/Tracer.h"

#include <atomic>
#include <iostream>

namespace {
static utility::Tracer::Event buildEvent("HashIndexedView", "build");
} // end namespace
lingodb::runtime::HashIndexedView* lingodb::runtime::HashIndexedView::build(lingodb::runtime::ExecutionContext* executionContext, lingodb::runtime::GrowingBuffer* buffer) {
   utility::Tracer::Trace trace(buildEvent);
   auto& values = buffer->getValues();
   size_t htSize = std::max(nextPow2(values.getLen() * 1.25), 1ul);
   size_t htMask = htSize - 1;
   auto* htView = new HashIndexedView(htSize, htMask);
   executionContext->registerState({htView, [](void* ptr) { delete reinterpret_cast<lingodb::runtime::HashIndexedView*>(ptr); }});
   values.iterateParallel([&](uint8_t* ptr) {
      auto* entry = (Entry*) ptr;
      size_t hash = (size_t) entry->hashValue;
      auto pos = hash & htMask;
      std::atomic_ref<Entry*> slot(htView->ht[pos]);
      Entry* current = slot.load();
      Entry* newEntry;
      do {
         entry->next = current;
         newEntry = lingodb::runtime::tag(entry, current, hash);
      } while (!slot.compare_exchange_weak(current, newEntry));
   });
   trace.stop();
   return htView;
}
void lingodb::runtime::HashIndexedView::destroy(lingodb::runtime::HashIndexedView* ht) {
   delete ht;
}
lingodb::runtime::HashIndexedView::HashIndexedView(size_t htSize, size_t htMask) : ht(lingodb::runtime::FixedSizedBuffer<Entry*>::createZeroed(htSize)), htMask(htMask) {}
lingodb::runtime::HashIndexedView::~HashIndexedView() {
   lingodb::runtime::FixedSizedBuffer<Entry*>::deallocate(ht, htMask + 1);
}
