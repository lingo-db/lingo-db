#include "lingodb/runtime/Hashtable.h"
#include "lingodb/utility/Tracer.h"
#include <iostream>
namespace {
static utility::Tracer::Event mergeEvent("Hashtable", "merge");
} // end namespace
lingodb::runtime::Hashtable* lingodb::runtime::Hashtable::create(lingodb::runtime::ExecutionContext* executionContext, size_t typeSize, size_t initialCapacity) {
   auto* ht = new Hashtable(initialCapacity, typeSize);
   executionContext->registerState({ht, [](void* ptr) { delete reinterpret_cast<lingodb::runtime::Hashtable*>(ptr); }});
   return ht;
}
void lingodb::runtime::Hashtable::resize() {
   size_t oldHtSize = hashMask + 1;
   size_t newHtSize = oldHtSize * 2;
   ht.setNewSize(newHtSize);
   hashMask = newHtSize - 1;
   values.iterate([&](uint8_t* entryRawPtr) {
      auto* entry = (Entry*) entryRawPtr;
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht.at(pos);
      ht.at(pos) = lingodb::runtime::tag(entry, previousPtr, entry->hashValue);
      entry->next = lingodb::runtime::untag(previousPtr);
   });
}
void lingodb::runtime::Hashtable::destroy(lingodb::runtime::Hashtable* ht) {
   delete ht;
}
lingodb::runtime::Hashtable::Entry* lingodb::runtime::Hashtable::insert(size_t hash) {
   if (values.getLen() > hashMask / 2) {
      resize();
   }
   Entry* res = (Entry*) values.insert();
   auto pos = hash & hashMask;
   auto* previousPtr = ht.at(pos);
   ht.at(pos) = lingodb::runtime::tag(res, previousPtr, hash);
   res->next = lingodb::runtime::untag(previousPtr);
   res->hashValue = hash;
   return res;
}

lingodb::runtime::BufferIterator* lingodb::runtime::Hashtable::createIterator() {
   return values.createIterator();
}

lingodb::runtime::Hashtable* lingodb::runtime::Hashtable::merge(lingodb::runtime::ThreadLocal* threadLocal, bool (*isEq)(uint8_t*, uint8_t*), void (*merge)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace mergeHt(mergeEvent);
   lingodb::runtime::Hashtable* first = nullptr;
   for (auto* current : threadLocal->getThreadLocalValues<lingodb::runtime::Hashtable>()) {
      if(!current) continue;
      if (!first) {
         first = current;
      } else {
         first->mergeEntries(isEq, merge, current);
      }
   }
   return first;
}
void lingodb::runtime::Hashtable::mergeEntries(bool (*isEq)(uint8_t*, uint8_t*), void (*merge)(uint8_t*, uint8_t*), lingodb::runtime::Hashtable* other) {
   utility::Tracer::Trace trace(mergeEvent);
   other->values.iterate([&](uint8_t* entryRawPtr) {
      auto* otherEntry = (Entry*) entryRawPtr;
      auto otherHash = otherEntry->hashValue;
      auto pos = otherHash & hashMask;
      auto* candidate = ht.at(pos);
      candidate = lingodb::runtime::filterTagged(candidate, otherHash);
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
void lingodb::runtime::Hashtable::lock(lingodb::runtime::Hashtable::Entry* entry, size_t subtract) {
   entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
   uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
   std::atomic_ref<uintptr_t> l(nextPtr);
   uintptr_t mask = 0xffff000000000000;
   while (l.exchange(nextPtr | mask) & mask) {
   }
}
void lingodb::runtime::Hashtable::unlock(lingodb::runtime::Hashtable::Entry* entry, size_t subtract) {
   entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
   uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
   std::atomic_ref<uintptr_t> l(nextPtr);
   l.store(nextPtr & ~0xffff000000000000);

}