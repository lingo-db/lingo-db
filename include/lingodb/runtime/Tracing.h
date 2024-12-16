#ifndef LINGODB_RUNTIME_TRACING_H
#define LINGODB_RUNTIME_TRACING_H

#include "helpers.h"

namespace lingodb::runtime{
   class ExecutionStepTracing{
      public:
      static uint8_t * start(runtime::VarLen32 step);
        static void end(uint8_t* tracing);

   };
}; // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_TRACING_H
