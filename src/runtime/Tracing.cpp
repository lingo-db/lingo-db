#include "lingodb/runtime/Tracing.h"
#include "lingodb/utility/Tracer.h"

#ifdef TRACER
static utility::Tracer::StringMetaDataEvent executionStepEvent("Execution", "Step", "location");
uint8_t * lingodb::runtime::ExecutionStepTracing::start(lingodb::runtime::VarLen32 step) {
        return reinterpret_cast<uint8_t*>(new utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>( executionStepEvent, step.str()));
}
void lingodb::runtime::ExecutionStepTracing::end(uint8_t* tracing) {
        delete reinterpret_cast<utility::Tracer::MetaDataTrace<utility::Tracer::StringMetaDataEvent, std::string>*>(tracing);
}
#else
uint8_t * lingodb::runtime::ExecutionStepTracing::start(lingodb::runtime::VarLen32 step){
   return nullptr;
}
void lingodb::runtime::ExecutionStepTracing::end(uint8_t* tracing){
}

#endif