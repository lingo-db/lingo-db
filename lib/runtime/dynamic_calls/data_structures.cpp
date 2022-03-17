#include "runtime/helpers.h"
#include <iostream>

struct Vec {
   size_t len;
   size_t cap;
   runtime::Bytes bytes;
   Vec(size_t len, size_t cap, size_t numBytes) : len(len), cap(cap), bytes((uint8_t*) malloc(numBytes)) {}
};
struct AggrHt {
   struct Entry {
      Entry* next;
      size_t hashValue;
      //kv follows
   };
   size_t numValues;
   size_t capacity;
   runtime::Bytes values;
   runtime::Bytes ht;
   //initial value follows...
   AggrHt(size_t initialCapacity, size_t typeSize) : numValues(0), capacity(initialCapacity), values((uint8_t*) malloc(initialCapacity * typeSize)), ht((uint8_t*) malloc(initialCapacity * 2 * sizeof(uint64_t))) {
      ht.fill(0x00, initialCapacity * 2 * sizeof(uint64_t));
   }
};
EXPORT Vec* rt_create_vec(size_t sizeOfType, size_t initialCapacity) {
   return new Vec(0, initialCapacity, initialCapacity * sizeOfType);
}
EXPORT void rt_resize_vec(Vec* v, size_t typeSize) {
   v->bytes.resize(v->cap * typeSize, v->cap * 2 * typeSize);
   v->cap *= 2;
}
EXPORT AggrHt* rt_create_aggr_ht(size_t typeSize, size_t initialCapacity) {
   return new (malloc(sizeof(AggrHt) + typeSize)) AggrHt(initialCapacity, typeSize);
}
EXPORT void rt_resize_aggr_ht(AggrHt* aggrHt, size_t typeSize) {
   size_t oldHtSize = aggrHt->capacity * 2 * sizeof(uint64_t);
   size_t oldVecSize = aggrHt->capacity * typeSize;
   size_t newVecSize = oldVecSize * 2;
   size_t newHtSize = oldHtSize * 2;
   aggrHt->values.resize(oldVecSize, newVecSize);
   aggrHt->ht.resize(oldHtSize, newHtSize);
   aggrHt->ht.fill(0x00, newHtSize);
   aggrHt->capacity *= 2;
   auto* ht = (AggrHt::Entry**) aggrHt->ht.getPtr();
   size_t hashMask = (newHtSize / sizeof(size_t)) - 1;
   auto* valuesPtr = aggrHt->values.getPtr();
   for (size_t i = 0; i < aggrHt->numValues; i++) {
      auto* entry = (AggrHt::Entry*) &valuesPtr[i * typeSize];
      auto pos = entry->hashValue & hashMask;
      auto* previousPtr = ht[pos];
      ht[pos] = entry;
      entry->next = previousPtr;
   }
}
//!llvm.ptr<struct<(ptr<>, i64, struct<(ptr<i64>, i64)>, i64)>>
struct JoinHt {
   struct Entry {
      Entry* next;
      //kv follows
   };
   runtime::Bytes values;
   size_t numValues;
   runtime::Bytes ht;
   size_t htMask;
   JoinHt(const runtime::Bytes& values, size_t numValues, size_t htMask, size_t htSize) : values(values), numValues(numValues), ht((uint8_t*) malloc(htSize * sizeof(uint64_t))), htMask(htMask) {}
};
EXPORT uint64_t next_pow_2(uint64_t v) {
   v--;
   v |= v >> 1;
   v |= v >> 2;
   v |= v >> 4;
   v |= v >> 8;
   v |= v >> 16;
   v |= v >> 32;
   v++;
   return v;
}
template <typename T>
T* tag(T* ptr, T* previousPtr, size_t hash) {
   constexpr uint64_t ptrMask = 0x0000ffffffffffffull;
   constexpr uint64_t tagMask = 0xffff000000000000ull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   size_t previousAsInt = reinterpret_cast<size_t>(previousPtr);
   size_t previousTag = previousAsInt & tagMask;
   size_t currentTag = hash & tagMask;
   auto tagged = (asInt & ptrMask) | previousTag | currentTag;
   auto* res = reinterpret_cast<T*>(tagged);
   return res;
}
template <typename T>
T* untag(T* ptr) {
   constexpr size_t ptrMask = 0x0000ffffffffffffull;
   size_t asInt = reinterpret_cast<size_t>(ptr);
   return reinterpret_cast<T*>(asInt & ptrMask);
}
EXPORT JoinHt* rt_build_join_ht(Vec* v, size_t typeSize) {
   size_t htSize = next_pow_2(v->len);
   size_t htMask = htSize - 1;
   auto* joinHt = new JoinHt(v->bytes, v->len, htMask, htSize);
   joinHt->ht.fill(0x00, htSize * sizeof(uint64_t));
   auto* valuesPtr = joinHt->values.getPtr();
   auto* ht = (JoinHt::Entry**) joinHt->ht.getPtr();
   for (size_t i = 0; i < v->len; i++) {
      auto* entry = (JoinHt::Entry*) &valuesPtr[i * typeSize];
      size_t hash = (size_t) entry->next;
      auto pos = hash & htMask;
      auto* previousPtr = ht[pos];
      ht[pos] = tag(entry, ht[pos], hash);
      entry->next = previousPtr;
   }
   return joinHt;
}