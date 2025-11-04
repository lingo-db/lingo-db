#include "lingodb/runtime/Hashtable.h"
#include "lingodb/utility/Tracer.h"
#include <atomic>
#include <cstring>
#include <iostream>

namespace {
static lingodb::utility::Tracer::Event mergeEvent("Hashtable", "merge");
} // end namespace
lingodb::runtime::Hashtable* lingodb::runtime::Hashtable::create(size_t typeSize, size_t initialCapacity) {
   lingodb::runtime::ExecutionContext* executionContext = runtime::getCurrentExecutionContext();
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
      if (!current) continue;
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

uint8_t* lingodb::runtime::Hashtable::lookup(size_t hash, uint8_t* keyVal) {
   //try to find an existing entry
   auto pos = hash & hashMask;
   auto* candidate = ht.at(pos);
   candidate = lingodb::runtime::filterTagged(candidate, hash);
   while (candidate) {
      if (candidate->hashValue == hash) {
         auto* candidateContent = reinterpret_cast<uint8_t*>(&candidate->content);
         if (isEq(candidateContent, keyVal)) {
            return &candidate->content[0];
         }
      }
      candidate = candidate->next;
   }
   throw std::runtime_error("Hashtable: lookup failed, no entry found for hash " + std::to_string(hash));
}

uint8_t* lingodb::runtime::Hashtable::lookUpOrInsert(size_t hash, uint8_t* keyVal) {
   //try to find an existing entry
   auto pos = hash & hashMask;
   auto* candidate = ht.at(pos);
   candidate = lingodb::runtime::filterTagged(candidate, hash);
   while (candidate) {
      if (candidate->hashValue == hash) {
         auto* candidateContent = reinterpret_cast<uint8_t*>(&candidate->content);
         if (isEq(candidateContent, keyVal)) {
            return &candidate->content[0];
         }
      }
      candidate = candidate->next;
   }
   //not found, insert a new entry
   return &insert(hash)->content[0];
}

size_t lingodb::runtime::Hashtable::size() {
   return values.getLen();
}

void lingodb::runtime::Hashtable::setEqFn(bool (*isEq)(uint8_t*, uint8_t*)) {
   this->isEq = isEq;
}

bool lingodb::runtime::Hashtable::contains(size_t hash, uint8_t* keyVal) {
   //try to find an existing entry
   auto pos = hash & hashMask;
   auto* candidate = ht.at(pos);
   candidate = lingodb::runtime::filterTagged(candidate, hash);
   while (candidate) {
      if (candidate->hashValue == hash) {
         auto* candidateContent = reinterpret_cast<uint8_t*>(&candidate->content);
         if (isEq(candidateContent, keyVal)) {
            return true;
         }
      }
      candidate = candidate->next;
   }
   return false;
}

bool lingodb::runtime::HashtableIterator::isValid() {
   return valid;
}

void lingodb::runtime::HashtableIterator::next() {
   if (!valid) {
      throw std::runtime_error("HashtableIterator: next called on invalid iterator");
   }
   currPosInBuffer++;
   if (currPosInBuffer * typeSize >= currBuffer.numElements) {
      bufferIt->next();
      if (bufferIt->isValid()) {
         currBuffer = bufferIt->getCurrentBuffer();
         currPosInBuffer = 0;
         valid = currPosInBuffer * typeSize < currBuffer.numElements;
      } else {
         valid = false;
      }
   }
}

lingodb::runtime::HashtableIterator* lingodb::runtime::Hashtable::createHtIterator(Hashtable* ht) {
   auto* bufferIt = ht->values.createIterator();
   auto* it = new HashtableIterator(bufferIt, ht->typeSize);
   runtime::getCurrentExecutionContext()->registerState({it, [](void* ptr) { delete reinterpret_cast<HashtableIterator*>(ptr); }});
   return it;
}

uint8_t* lingodb::runtime::HashtableIterator::getCurrent() {
   if (!valid) {
      throw std::runtime_error("HashtableIterator: getCurrent called on invalid iterator");
   }
   return &reinterpret_cast<Hashtable::Entry*>(&currBuffer.ptr[currPosInBuffer * typeSize])->content[0];
}
