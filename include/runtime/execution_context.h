#ifndef RUNTIME_EXECUTION_CONTEXT_H
#define RUNTIME_EXECUTION_CONTEXT_H
#include "database.h"
namespace runtime {
class ExecutionContext {
   public:
   int id;
   std::unique_ptr<Database> db;
};
} // end namespace runtime

#endif // RUNTIME_EXECUTION_CONTEXT_H
