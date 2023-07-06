#include "runtime/LazyJoinHashtable.h"
#include "runtime/GrowingBuffer.h"
#include "utility/Tracer.h"

#include <atomic>
#include <iostream>

namespace {
static utility::Tracer::Event buildEvent("HashIndexedView", "build");
} // end namespace
runtime::HashIndexedView* runtime::HashIndexedView::build(runtime::ExecutionContext* executionContext, runtime::GrowingBuffer* buffer) {
   utility::Tracer::Trace trace(buildEvent);
   auto& values = buffer->getValues();
   size_t htSize = std::max(nextPow2(values.getLen() * 1.25), 1ul);
   size_t htMask = htSize - 1;
   auto* htView = new HashIndexedView(htSize, htMask);
   executionContext->registerState({htView, [](void* ptr) { delete reinterpret_cast<runtime::HashIndexedView*>(ptr); }});
   values.iterateParallel([&](uint8_t* ptr) {
      auto* entry = (Entry*) ptr;
      size_t hash = (size_t) entry->hashValue;
      auto pos = hash & htMask;
      std::atomic_ref<Entry*> slot(htView->ht[pos]);
      Entry* current = slot.load();
      Entry* newEntry;
      do {
         entry->next = current;
         newEntry = runtime::tag(entry, current, hash);
      } while (!slot.compare_exchange_weak(current, newEntry));
   });
   trace.stop();
   return htView;
}
void runtime::HashIndexedView::destroy(runtime::HashIndexedView* ht) {
   delete ht;
}
runtime::HashIndexedView::HashIndexedView(size_t htSize, size_t htMask) : ht(runtime::FixedSizedBuffer<Entry*>::createZeroed(htSize)), htMask(htMask) {}
runtime::HashIndexedView::~HashIndexedView() {
   runtime::FixedSizedBuffer<Entry*>::deallocate(ht, htMask + 1);
}
