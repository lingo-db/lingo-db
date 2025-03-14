#include "lingodb/runtime/ExecutionContext.h"

void lingodb::runtime::ExecutionContext::setResult(uint32_t id, uint8_t* ptr) {
   auto* context = getCurrentExecutionContext();
   assert(context);
   context->states.erase(ptr);
   context->results[id] = ptr;
}
void lingodb::runtime::ExecutionContext::clearResult(uint32_t id) {
   auto* context = getCurrentExecutionContext();
   context->results.erase(id);
}

void lingodb::runtime::ExecutionContext::setTupleCount(uint32_t id, int64_t tupleCount) {
   auto* context = getCurrentExecutionContext();
   context->tupleCounts[id] = tupleCount;
}
void lingodb::runtime::ExecutionContext::reset() {
   states.forEach([&](void* key, State value) {
      value.freeFn(value.ptr);
   });

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
namespace {
thread_local lingodb::runtime::ExecutionContext* currentExecutionContext = nullptr;
} // end namespace
void lingodb::runtime::setCurrentExecutionContext(lingodb::runtime::ExecutionContext* context) {
   currentExecutionContext = context;
}

lingodb::runtime::ExecutionContext* lingodb::runtime::getCurrentExecutionContext() {
   assert(currentExecutionContext);
   return currentExecutionContext;
}