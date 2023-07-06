#ifndef RUNTIME_SESSION_H
#define RUNTIME_SESSION_H
#include "Catalog.h"
#include <memory>
namespace runtime {
class ExecutionContext;
class Session {
   std::shared_ptr<Catalog> catalog;
   Session(std::shared_ptr<Catalog> catalog) : catalog(catalog) {}

   public:
   static std::shared_ptr<Session> createSession();
   static std::shared_ptr<Session> createSession(std::string dbDir,bool eagerLoading=true);
   std::shared_ptr<Catalog> getCatalog();
   std::unique_ptr<ExecutionContext> createExecutionContext();
};
} //end namespace runtime
#endif //RUNTIME_SESSION_H
