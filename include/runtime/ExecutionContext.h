#ifndef RUNTIME_EXECUTIONCONTEXT_H
#define RUNTIME_EXECUTIONCONTEXT_H
#include "Database.h"
namespace runtime {
class ExecutionContext {
   public:
   int id;
   std::unique_ptr<Database> db;
   Database* getDatabase();
};
} // end namespace runtime

#endif // RUNTIME_EXECUTIONCONTEXT_H
