#include "runtime/SimpleState.h"
#include "utility/Tracer.h"
#include <iostream>
namespace {
static utility::Tracer::Event createEvent("SimpleState", "create");
static utility::Tracer::Event mergeEvent("SimpleState", "merge");
} // end namespace
uint8_t* runtime::SimpleState::create(runtime::ExecutionContext* executionContext, size_t sizeOfType) {
   utility::Tracer::Trace trace(createEvent);
   uint8_t* res = (uint8_t*) malloc(sizeOfType);
   executionContext->registerState({res, [](void* ptr) { free(ptr); }});
   trace.stop();
   return res;
}

uint8_t* runtime::SimpleState::merge(runtime::ThreadLocal* threadLocal, void (*merge)(uint8_t*, uint8_t*)) {
   utility::Tracer::Trace trace(mergeEvent);
   uint8_t* first = nullptr;
   for (auto* ptr : threadLocal->getTls()) {
      auto* current = ptr;
      if (!first) {
         first = current;
      } else {
         merge(first, current);
      }
   }
   trace.stop();
   return first;
}