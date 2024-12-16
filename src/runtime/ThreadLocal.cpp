#include "lingodb/runtime/ThreadLocal.h"
#include "lingodb/utility/Tracer.h"
namespace {
static utility::Tracer::Event getLocalEvent("ThreadLocal", "getLocal");
static utility::Tracer::Event mergeEvent("ThreadLocal", ",merge");
} // end namespace
uint8_t* lingodb::runtime::ThreadLocal::getLocal() {
   utility::Tracer::Trace trace(getLocalEvent);
   auto* local = tls.local();
   trace.stop();
   return local;
}
lingodb::runtime::ThreadLocal* lingodb::runtime::ThreadLocal::create(uint8_t* (*initFn)(uint8_t*), uint8_t* initArg) {
   return new ThreadLocal(initFn,initArg);
}

uint8_t* lingodb::runtime::ThreadLocal::merge(void (*mergeFn)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace trace(mergeEvent);
   uint8_t* first = nullptr;
   for (auto* ptr : getTls()) {
      auto* current = ptr;
      if (!first) {
         first = current;
      } else {
         mergeFn(first, current);
      }
   }
   trace.stop();
   return first;
}
