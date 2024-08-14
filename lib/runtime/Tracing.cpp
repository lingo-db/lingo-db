#include "runtime/Tracing.h"
#include "utility/Tracer.h"

#ifdef TRACER
static utility::Tracer::StringMetaDataEvent executionStepEvent("Execution", "Step", "location");
uint8_t * runtime::ExecutionStepTracing::start(runtime::VarLen32 step) {
        return reinterpret_cast<uint8_t*>(new utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>( executionStepEvent, step.str()));
}
void runtime::ExecutionStepTracing::end(uint8_t* tracing) {
        delete reinterpret_cast<utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>*>(tracing);
}
#else
uint8_t * runtime::ExecutionStepTracing::start(runtime::VarLen32 step){
   return nullptr;
}
void runtime::ExecutionStepTracing::end(uint8_t* tracing){
}

#endif