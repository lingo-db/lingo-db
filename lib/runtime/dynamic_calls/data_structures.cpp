#include "runtime/helpers.h"
#include <iostream>

struct Vec {
   size_t len;
   size_t cap;
   uint8_t* ptr;
   size_t typeSize;
   Vec(size_t cap, size_t typeSize) : len(0), cap(cap), ptr((uint8_t*) malloc(typeSize * cap)), typeSize(typeSize) {}
   void resize() {
      size_t newCapacity = cap * 2;
      ptr = runtime::MemoryHelper::resize(ptr, len * typeSize, newCapacity * typeSize);
      cap=newCapacity;
   }
   template <class T>
   T* ptrAt(size_t i) {
      return (T*) &ptr[i * typeSize];
   }
};
template <class T>
struct FixedSizedBuffer {
   FixedSizedBuffer(size_t size) : ptr((T*) malloc(size * sizeof(T))) {
      runtime::MemoryHelper::zero((uint8_t*) ptr, size * sizeof(T));
   }
   T* ptr;
   void setNewSize(size_t newSize) {
      free(ptr);
      ptr = (T*) malloc(newSize * sizeof(T));
      runtime::MemoryHelper::zero((uint8_t*) ptr, newSize * sizeof(T));
   }
   T& at(size_t i) {
      return ptr[i];
   }
};
struct AggrHt {
   struct Entry {
      Entry* next;
      size_t hashValue;
      //kv follows
   };
   FixedSizedBuffer<Entry*> ht;
   Vec values;
   //initial value follows...
   AggrHt(size_t initialCapacity, size_t typeSize) : ht(initialCapacity * 2), values(initialCapacity, typeSize) {}
   void resize() {
      size_t oldHtSize = values.cap * 2;
      size_t newHtSize = oldHtSize * 2;
      values.resize();
      ht.setNewSize(newHtSize);
      size_t hashMask = newHtSize - 1;
      for (size_t i = 0; i < values.len; i++) {
         auto* entry = values.ptrAt<Entry>(i);
         auto pos = entry->hashValue & hashMask;
         auto* previousPtr = ht.at(pos);
         ht.at(pos) = entry;
         entry->next = previousPtr;
      }
      /*
       *
       */
   }
};
EXPORT Vec* rt_create_vec(size_t sizeOfType, size_t initialCapacity) {
   return new Vec(initialCapacity, sizeOfType);
}
EXPORT void rt_resize_vec(Vec* v) {
   v->resize();
}
EXPORT AggrHt* rt_create_aggr_ht(size_t typeSize, size_t initialCapacity) {
   return new (malloc(sizeof(AggrHt) + typeSize)) AggrHt(initialCapacity, typeSize);
}
EXPORT void rt_resize_aggr_ht(AggrHt* aggrHt) {
   aggrHt->resize();
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
//!llvm.ptr<struct<(ptr<>, i64, struct<(ptr<i64>, i64)>, i64)>>
struct JoinHt {
   struct Entry {
      Entry* next;
      //kv follows
   };
   FixedSizedBuffer<Entry*> ht;
   size_t htMask;
   Vec values;
   JoinHt(Vec* values, size_t htMask, size_t htSize) : ht(htSize), htMask(htMask),values(*values) {}
   static uint64_t next_pow_2(uint64_t v) {
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
   static JoinHt* build(Vec* values) {
      size_t htSize = next_pow_2(values->len);
      size_t htMask = htSize - 1;
      auto* joinHt = new JoinHt(values, htMask, htSize);
      for (size_t i = 0; i < values->len; i++) {
         auto* entry = values->ptrAt<Entry>(i);
         size_t hash = (size_t) entry->next;
         auto pos = hash & htMask;
         auto* previousPtr = joinHt->ht.at(pos);
         joinHt->ht.at(pos) = tag(entry, previousPtr, hash);
         entry->next = previousPtr;
      }
      return joinHt;
   }
};

EXPORT JoinHt* rt_build_join_ht(Vec* v) {
   return JoinHt::build(v);
}