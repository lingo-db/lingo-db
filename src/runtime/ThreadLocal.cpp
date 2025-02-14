#include "lingodb/runtime/ThreadLocal.h"
#include "lingodb/utility/Tracer.h"
namespace {
static utility::Tracer::Event getLocalEvent("ThreadLocal", "getLocal");
static utility::Tracer::Event mergeEvent("ThreadLocal", ",merge");
} // end namespace
uint8_t* lingodb::runtime::ThreadLocal::getLocal() {
   utility::Tracer::Trace trace(getLocalEvent);
   assert(lingodb::scheduler::currentWorkerId() < lingodb::scheduler::getNumWorkers());
   if (!values[lingodb::scheduler::currentWorkerId()]) {
      values[lingodb::scheduler::currentWorkerId()] = initFn(arg);
   }
   return values[lingodb::scheduler::currentWorkerId()];
}
lingodb::runtime::ThreadLocal* lingodb::runtime::ThreadLocal::create(uint8_t* (*initFn)(uint8_t*), uint8_t* initArg) {
   return new ThreadLocal(initFn, initArg);
}

uint8_t* lingodb::runtime::ThreadLocal::merge(void (*mergeFn)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace trace(mergeEvent);
   uint8_t* first = nullptr;
   for (auto* ptr : getThreadLocalValues<uint8_t>()) {
      if(ptr) {
         auto* current = ptr;
         if (!first) {
            first = current;
         } else {
            mergeFn(first, current);
         }
      }
   }
   trace.stop();
   return first;
}
