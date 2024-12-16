#ifndef LINGODB_RUNTIME_SIMPLESTATE_H
#define LINGODB_RUNTIME_SIMPLESTATE_H
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/ThreadLocal.h"

namespace lingodb::runtime {
class SimpleState {
   public:
   static uint8_t* create(runtime::ExecutionContext* executionContext, size_t sizeOfType);
   static uint8_t* merge(ThreadLocal* threadLocal, void (*merge)(uint8_t* dest, uint8_t* src));
};
} //end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_SIMPLESTATE_H
