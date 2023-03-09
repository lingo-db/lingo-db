#include "runtime/ExecutionContext.h"
runtime::Database* runtime::ExecutionContext::getDatabase() {
   return db.get();
}

void runtime::ExecutionContext::setResult(uint32_t id, uint8_t* ptr) {
   results[id] = ptr;
}

void runtime::ExecutionContext::setTupleCount(uint32_t id, int64_t tupleCount){
   tupleCounts[id] = tupleCount;
}