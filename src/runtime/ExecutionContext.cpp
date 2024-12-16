#include "lingodb/runtime/ExecutionContext.h"
#include <iostream>

void lingodb::runtime::ExecutionContext::setResult(uint32_t id, uint8_t* ptr) {
   states.erase(ptr);
   results[id] = ptr;
}
void lingodb::runtime::ExecutionContext::clearResult(uint32_t id){
   results.erase(id);
}

void lingodb::runtime::ExecutionContext::setTupleCount(uint32_t id, int64_t tupleCount) {
   tupleCounts[id] = tupleCount;
}
void lingodb::runtime::ExecutionContext::reset() {
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
lingodb::runtime::ExecutionContext::~ExecutionContext() {
   reset();
}