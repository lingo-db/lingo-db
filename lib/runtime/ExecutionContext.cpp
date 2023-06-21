#include "runtime/ExecutionContext.h"
#include <iostream>

void runtime::ExecutionContext::setResult(uint32_t id, uint8_t* ptr) {
   states.erase(ptr);
   results[id] = ptr;
}

void runtime::ExecutionContext::setTupleCount(uint32_t id, int64_t tupleCount) {
   tupleCounts[id] = tupleCount;
}
void runtime::ExecutionContext::reset() {
   for (auto s : states) {
      s.second.freeFn(s.second.ptr);
   }
   for (auto local : allocators) {
      for (auto a : local) {
         a.second.freeFn(a.second.ptr);
      }
   }
   allocators.clear();
   states.clear();
}
runtime::ExecutionContext::~ExecutionContext() {
   reset();
}