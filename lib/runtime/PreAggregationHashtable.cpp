#include "runtime/PreAggregationHashtable.h"
#include "runtime/helpers.h"
#include "utility/Tracer.h"
#include <iostream>
#include <oneapi/tbb.h>
namespace {
static utility::Tracer::Event createEvent("OHtFragment", "create");
static utility::Tracer::Event mergeEvent("Oht", "merge");
static utility::Tracer::Event mergePartitionEvent("Oht", "mergePartition");
static utility::Tracer::Event mergeAllocate("Oht", "mergeAlloc");
static utility::Tracer::Event mergeDeallocate("Oht", "mergeDealloc");

} // end namespace
runtime::PreAggregationHashtableFragment::Entry* runtime::PreAggregationHashtableFragment::insert(size_t hash) {
   constexpr size_t outputMask = numOutputs - 1;
   constexpr size_t htMask = hashtableSize - 1;
   len++;
   auto outputIdx = hash & outputMask;
   if (!outputs[outputIdx]) {
      outputs[outputIdx] = new FlexibleBuffer(256, typeSize);
   }
   auto* newEntry = reinterpret_cast<runtime::PreAggregationHashtableFragment::Entry*>(outputs[outputIdx]->insert());
   newEntry->hashValue = hash;
   newEntry->next = nullptr;
   ht[hash & htMask] = newEntry;
   return newEntry;
}

runtime::PreAggregationHashtableFragment* runtime::PreAggregationHashtableFragment::create(runtime::ExecutionContext* context, size_t typeSize) {
   utility::Tracer::Trace trace(createEvent);
   auto* fragment = new PreAggregationHashtableFragment(typeSize);
   context->registerState({fragment, [](void* ptr) { delete reinterpret_cast<PreAggregationHashtableFragment*>(ptr); }});
   return fragment;
}
runtime::PreAggregationHashtable* runtime::PreAggregationHashtable::merge(runtime::ThreadLocal* threadLocal, bool (*eq)(uint8_t*, uint8_t*), void (*combine)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace trace(mergeEvent);

   std::mutex mutex;
   using Entry = runtime::PreAggregationHashtableFragment::Entry;
   constexpr size_t numPartitions = runtime::PreAggregationHashtableFragment::numOutputs;
   std::vector<FlexibleBuffer*> outputs[numPartitions];
   for (auto* ptr : threadLocal->getTls()) {
      auto* fragment = reinterpret_cast<PreAggregationHashtableFragment*>(ptr);
      for (size_t i = 0; i < numPartitions; i++) {
         auto* current = fragment->outputs[i];
         if (current) {
            outputs[i].push_back(current);
         }
      }
   }
   auto* res = new PreAggregationHashtable();
   tbb::parallel_for_each(&outputs[0], &outputs[numPartitions], [&](const std::vector<FlexibleBuffer*>& input) {
      utility::Tracer::Trace trace(mergePartitionEvent);
      size_t totalValues = 0;
      size_t minValues = 0;

      for (auto* o : input) {
         totalValues += o->getLen();
         minValues = std::max((size_t) 0, o->getLen());
      }
      runtime::FlexibleBuffer localBuffer(minValues, sizeof(Entry*));
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
      Entry** ht = runtime::FixedSizedBuffer<Entry*>::createZeroed(htSize);
      allocTrace.stop();
      for (auto* o : input) {
         o->iterate([&](uint8_t* entryRawPtr) {
            Entry* curr = reinterpret_cast<Entry*>(entryRawPtr);
            auto pos = curr->hashValue & htMask;
            auto* currCandidate = ht[pos];
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
               curr->next = ht[pos];
               ht[pos] = curr;
            }
         });
      }
      utility::Tracer::Trace deallocTrace(mergeDeallocate);
      runtime::FixedSizedBuffer<Entry*>::deallocate(ht, htSize);
      deallocTrace.stop();
      std::unique_lock<std::mutex> lock(mutex);
      res->buffer.merge(localBuffer);
   });
   return res;
}
runtime::BufferIterator* runtime::PreAggregationHashtable::createIterator() {
   return buffer.createIterator();
}