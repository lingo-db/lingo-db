#include "runtime/ThreadLocal.h"
#include "utility/Tracer.h"
namespace {
static utility::Tracer::Event getLocalEvent("ThreadLocal", "getLocal");
} // end namespace
uint8_t* runtime::ThreadLocal::getLocal() {
   utility::Tracer::Trace trace(getLocalEvent);
   auto* local = tls.local();
   trace.stop();
   return local;
}
runtime::ThreadLocal* runtime::ThreadLocal::create(uint8_t* (*initFn)()) {
   return new ThreadLocal(initFn);
}
