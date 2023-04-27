#ifndef RUNTIME_SIMPLESTATE_H
#define RUNTIME_SIMPLESTATE_H
#include "runtime/ExecutionContext.h"
#include "runtime/ThreadLocal.h"

namespace runtime {
class SimpleState {
   public:
   static uint8_t* create(runtime::ExecutionContext* executionContext, size_t sizeOfType);
   static uint8_t* merge(ThreadLocal* threadLocal, void (*merge)(uint8_t* dest, uint8_t* src));
};
} //end namespace runtime
#endif //RUNTIME_SIMPLESTATE_H
