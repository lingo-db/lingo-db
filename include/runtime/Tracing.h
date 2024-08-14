#ifndef RUNTIME_TRACING_H
#define RUNTIME_TRACING_H

#include "helpers.h"

namespace runtime{
   class ExecutionStepTracing{
      public:
      static uint8_t * start(runtime::VarLen32 step);
        static void end(uint8_t* tracing);

   };
}; // namespace runtime

#endif //RUNTIME_TRACING_H
