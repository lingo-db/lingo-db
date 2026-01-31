#ifndef LINGODB_RUNTIME_SESSION_H
#define LINGODB_RUNTIME_SESSION_H
#include "lingodb/catalog/Catalog.h"

#include <memory>
#include <mutex>
#include <stack>
namespace lingodb::runtime {
class ExecutionContext;
class Session {
   std::shared_ptr<catalog::Catalog> catalog;

   public:
   Session(std::shared_ptr<catalog::Catalog> catalog) : catalog(catalog) {}
   static std::shared_ptr<Session> createSession();
   static std::shared_ptr<Session> createSession(std::string dbDir, bool eagerLoading = true);
   std::shared_ptr<catalog::Catalog> getCatalog();
   std::unique_ptr<ExecutionContext> createExecutionContext();
};
} //end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_SESSION_H
