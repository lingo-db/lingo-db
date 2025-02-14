#include "lingodb/runtime/PreAggregationHashtable.h"
#include "lingodb/runtime/helpers.h"
#include "lingodb/utility/Tracer.h"
#include <iostream>

namespace {
static utility::Tracer::Event createEvent("OHtFragment", "create");
static utility::Tracer::Event mergeEvent("Oht", "merge");
static utility::Tracer::Event mergePartitionEvent("Oht", "mergePartition");
static utility::Tracer::Event mergeAllocate("Oht", "mergeAlloc");
static utility::Tracer::Event mergeDeallocate("Oht", "mergeDealloc");


class FragmentOutputsTask : public lingodb::scheduler::Task {
   std::vector<lingodb::runtime::FlexibleBuffer*> *outputs;
   std::function<void(std::vector<lingodb::runtime::FlexibleBuffer*>&)> cb;
   std::atomic<size_t> startIndex{0};

   public:
   FragmentOutputsTask(std::vector<lingodb::runtime::FlexibleBuffer*> *outputs, std::function<void(std::vector<lingodb::runtime::FlexibleBuffer*>&)> cb) : outputs(outputs), cb(cb) {}
   void run() override {
      constexpr size_t numPartitions = lingodb::runtime::PreAggregationHashtableFragment::numOutputs;
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex >= numPartitions) {
         workExhausted.store(true);
         return;
      }
      auto& batch = outputs[localStartIndex];
      cb(batch);
   }
};
} // end namespace
lingodb::runtime::PreAggregationHashtableFragment::Entry* lingodb::runtime::PreAggregationHashtableFragment::insert(size_t hash) {
   constexpr size_t outputMask = numOutputs - 1;
   constexpr size_t htMask = hashtableSize - 1;
   constexpr size_t htShift = 6; //2^6=64
   len++;
   auto outputIdx = hash & outputMask;
   if (!outputs[outputIdx]) {
      outputs[outputIdx] = new FlexibleBuffer(256, typeSize);
   }
   auto* newEntry = reinterpret_cast<lingodb::runtime::PreAggregationHashtableFragment::Entry*>(outputs[outputIdx]->insert());
   newEntry->hashValue = hash;
   newEntry->next = nullptr;
   ht[hash >> htShift & htMask] = newEntry;
   return newEntry;
}

lingodb::runtime::PreAggregationHashtableFragment* lingodb::runtime::PreAggregationHashtableFragment::create(lingodb::runtime::ExecutionContext* context, size_t typeSize) {
   utility::Tracer::Trace trace(createEvent);
   auto* fragment = new PreAggregationHashtableFragment(typeSize);
   context->registerState({fragment, [](void* ptr) { delete reinterpret_cast<PreAggregationHashtableFragment*>(ptr); }});
   return fragment;
}
lingodb::runtime::PreAggregationHashtableFragment::~PreAggregationHashtableFragment() {
   for (size_t i = 0; i < numOutputs; i++) {
      if (outputs[i]) {
         delete outputs[i];
      }
   }
}
lingodb::runtime::PreAggregationHashtable* lingodb::runtime::PreAggregationHashtable::merge(lingodb::runtime::ExecutionContext* context, lingodb::runtime::ThreadLocal* threadLocal, bool (*eq)(uint8_t*, uint8_t*), void (*combine)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace trace(mergeEvent);
   constexpr size_t htShift = 6; //2^6=64

   std::mutex mutex;
   using Entry = lingodb::runtime::PreAggregationHashtableFragment::Entry;
   constexpr size_t numPartitions = lingodb::runtime::PreAggregationHashtableFragment::numOutputs;
   std::vector<FlexibleBuffer*> outputs[numPartitions];
   for (auto* fragment : threadLocal->getThreadLocalValues<PreAggregationHashtableFragment>()) {
      if (!fragment) {
         continue;
      }
      for (size_t i = 0; i < numPartitions; i++) {
         auto* current = fragment->outputs[i];
         if (current) {
            outputs[i].push_back(current);
         }
      }
   }
   auto* res = new PreAggregationHashtable();
   context->registerState({res, [](void* ptr) { delete reinterpret_cast<PreAggregationHashtable*>(ptr); }});
   auto handleMerge = [&](std::vector<FlexibleBuffer*>& input) {
      size_t id = &input - &outputs[0];
      utility::Tracer::Trace trace(mergePartitionEvent);
      size_t totalValues = 0;
      size_t minValues = 0;

      for (auto* o : input) {
         totalValues += o->getLen();
         minValues = std::max((size_t) 0, o->getLen());
      }
      lingodb::runtime::FlexibleBuffer localBuffer(minValues, sizeof(Entry*));
      auto nextPow2 = [](uint64_t v) {
         v--;
         v |= v >> 1;
         v |= v >> 2;
         v |= v >> 4;
         v |= v >> 8;
         v |= v >> 16;
         v |= v >> 32;
         v++;
         return v;
      };

      size_t htSize = std::max(nextPow2(totalValues * 1.25), 1ul);
      size_t htMask = htSize - 1;
      utility::Tracer::Trace allocTrace(mergeAllocate);
      Entry** ht = lingodb::runtime::FixedSizedBuffer<Entry*>::createZeroed(htSize);
      allocTrace.stop();
      for (auto* o : input) {
         o->iterate([&](uint8_t* entryRawPtr) {
            Entry* curr = reinterpret_cast<Entry*>(entryRawPtr);
            auto pos = curr->hashValue >> htShift & htMask;
            auto* currCandidate = lingodb::runtime::untag(ht[pos]);
            bool merged = false;
            while (currCandidate) {
               if (currCandidate->hashValue == curr->hashValue && eq(currCandidate->content, curr->content)) {
                  combine(currCandidate->content, curr->content);
                  merged = true;
                  break;
               }
               currCandidate = currCandidate->next;
            }
            if (!merged) {
               auto* loc = reinterpret_cast<Entry**>(localBuffer.insert());
               *loc = curr;
               auto* previousPtr = ht[pos];
               ht[pos] = lingodb::runtime::tag(curr, previousPtr, curr->hashValue);
               curr->next = lingodb::runtime::untag(previousPtr);
            }
         });
      }
      utility::Tracer::Trace deallocTrace(mergeDeallocate);
      res->ht[id] = {ht, htMask};
      deallocTrace.stop();
      std::unique_lock<std::mutex> lock(mutex);
      res->buffer.merge(localBuffer);
      trace.stop();
   };
   lingodb::scheduler::awaitChildTask(std::make_unique<FragmentOutputsTask>(outputs, handleMerge));
   return res;
}
lingodb::runtime::BufferIterator* lingodb::runtime::PreAggregationHashtable::createIterator() {
   return buffer.createIterator();
}
lingodb::runtime::PreAggregationHashtable::Entry* lingodb::runtime::PreAggregationHashtable::lookup(size_t hash) {
   constexpr size_t partitionMask = lingodb::runtime::PreAggregationHashtableFragment::numOutputs - 1;
   auto partition = hash & partitionMask;
   if (!ht[partition].ht) {
      return nullptr;
   } else {
      return lingodb::runtime::filterTagged(ht[partition].ht[ht[partition].hashMask & hash >> 6], hash);
   }
}

void lingodb::runtime::PreAggregationHashtable::lock(Entry* entry, size_t subtract) {
   //utility::Tracer::Trace trace(lockEvent);
   entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
   uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
   std::atomic_ref<uintptr_t> l(nextPtr);
   uintptr_t mask = 0xffff000000000000;
   while (l.exchange(nextPtr | mask) & mask) {
   }
}
void lingodb::runtime::PreAggregationHashtable::unlock(Entry* entry, size_t subtract) {
   entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
   uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
   std::atomic_ref<uintptr_t> l(nextPtr);
   l.store(nextPtr & ~0xffff000000000000);
}

lingodb::runtime::PreAggregationHashtable::~PreAggregationHashtable() {
   for (auto p : ht) {
      lingodb::runtime::FixedSizedBuffer<Entry*>::deallocate(p.ht, p.hashMask + 1);
   }
}