#include "runtime/Hashtable.h"
#include "utility/Tracer.h"
#include <iostream>
namespace {
static utility::Tracer::Event mergeEvent("Hashtable", "merge");
} // end namespace
runtime::Hashtable* runtime::Hashtable::create(runtime::ExecutionContext* executionContext, size_t typeSize, size_t initialCapacity) {
   auto* ht = new Hashtable(initialCapacity, typeSize);
   executionContext->registerState({ht, [](void* ptr) { delete reinterpret_cast<runtime::Hashtable*>(ptr); }});
   return ht;
}
void runtime::Hashtable::resize() {
   size_t oldHtSize = hashMask + 1;
   size_t newHtSize = oldHtSize * 2;
   ht.setNewSize(newHtSize);
   hashMask = newHtSize - 1;
   values.iterate([&](uint8_t* entryRawPtr) {
      auto* entry = (Entry*) entryRawPtr;
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = runtime::tag(entry, previousPtr, entry->hashValue);
      entry->next = runtime::untag(previousPtr);
   });
}
void runtime::Hashtable::destroy(runtime::Hashtable* ht) {
   delete ht;
}
runtime::Hashtable::Entry* runtime::Hashtable::insert(size_t hash) {
   if (values.getLen() > hashMask / 2) {
      resize();
   }
   Entry* res = (Entry*) values.insert();
   auto pos = hash & hashMask;
   auto* previousPtr = ht.at(pos);
   ht.at(pos) = runtime::tag(res, previousPtr, hash);
   res->next = runtime::untag(previousPtr);
   res->hashValue = hash;
   return res;
}

runtime::BufferIterator* runtime::Hashtable::createIterator() {
   return values.createIterator();
}

runtime::Hashtable* runtime::Hashtable::merge(runtime::ThreadLocal* threadLocal, bool (*isEq)(uint8_t*, uint8_t*), void (*merge)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace mergeHt(mergeEvent);
   runtime::Hashtable* first = nullptr;
   for (auto* ptr : threadLocal->getTls()) {
      auto* current = reinterpret_cast<runtime::Hashtable*>(ptr);
      if (!first) {
         first = current;
      } else {
         first->mergeEntries(isEq, merge, current);
      }
   }
   return first;
}
void runtime::Hashtable::mergeEntries(bool (*isEq)(uint8_t*, uint8_t*), void (*merge)(uint8_t*, uint8_t*), runtime::Hashtable* other) {
   utility::Tracer::Trace trace(mergeEvent);
   other->values.iterate([&](uint8_t* entryRawPtr) {
      auto* otherEntry = (Entry*) entryRawPtr;
      auto otherHash = otherEntry->hashValue;
      auto pos = otherHash & hashMask;
      auto* candidate = ht.at(pos);
      candidate = runtime::filterTagged(candidate, otherHash);
      bool matchFound = false;
      while (candidate) {
         if (candidate->hashValue == otherHash) {
            auto* candidateContent = reinterpret_cast<uint8_t*>(&candidate->content);
            auto* otherContent = reinterpret_cast<uint8_t*>(&otherEntry->content);
            if (isEq(candidateContent, otherContent)) {
               merge(candidateContent, otherContent);
               matchFound = true;
               break;
            }
         }
         candidate = candidate->next;
      }
      if (!matchFound) {
         auto* newEntry = insert(otherHash);
         std::memcpy(newEntry->content, otherEntry->content, typeSize - sizeof(Entry));
      }
   });

   //todo migrate buffers?
}
void runtime::Hashtable::lock(runtime::Hashtable::Entry* entry, size_t subtract) {
   entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
   uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
   std::atomic_ref<uintptr_t> l(nextPtr);
   uintptr_t mask = 0xffff000000000000;
   while (l.exchange(nextPtr | mask) & mask) {
   }
}
void runtime::Hashtable::unlock(runtime::Hashtable::Entry* entry, size_t subtract) {
   entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
   uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
   std::atomic_ref<uintptr_t> l(nextPtr);
   l.store(nextPtr & ~0xffff000000000000);

}