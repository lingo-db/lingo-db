#ifndef LINGODB_RUNTIME_SESSION_H
#define LINGODB_RUNTIME_SESSION_H
#include "lingodb/catalog/Catalog.h"

#include <memory>
#include <mutex>
#include <stack>
namespace mlir {
class MLIRContext;
} //end namespace mlir
namespace lingodb::runtime {
class ExecutionContext;
class Session {
   std::shared_ptr<catalog::Catalog> catalog;

   public:
   std::mutex contextStackMutex;
   std::stack<mlir::MLIRContext*> llvmContextStack;
   std::stack<mlir::MLIRContext*> noLlvmContextStack;
   Session(std::shared_ptr<catalog::Catalog> catalog) : catalog(catalog) {}
   static std::shared_ptr<Session> createSession();
   static std::shared_ptr<Session> createSession(std::string dbDir, bool eagerLoading = true);
   std::shared_ptr<catalog::Catalog> getCatalog();
   std::unique_ptr<ExecutionContext> createExecutionContext();
   ~Session();
};
} //end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_SESSION_H
