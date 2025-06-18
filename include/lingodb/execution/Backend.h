#ifndef LINGODB_EXECUTION_BACKEND_H
#define LINGODB_EXECUTION_BACKEND_H
#include "Error.h"
#include "Instrumentation.h"
#include "lingodb/runtime/ExecutionContext.h"
namespace mlir {
class ModuleOp;
} // namespace mlir
namespace lingodb::execution {
using mainFnType = std::add_pointer<void()>::type;
class ExecutionBackend {
   protected:
   std::unordered_map<std::string, double> timing;
   Error error;
   bool verify = true;
   std::shared_ptr<SnapshotState> serializationState;

   public:
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   void disableVerification() {
      verify = false;
   }
   virtual void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) = 0;
   void setSerializationState(std::shared_ptr<SnapshotState> serializationState) {
      ExecutionBackend::serializationState = serializationState;
   }
   auto getSerializationState() {
      return serializationState;
   }

   virtual ~ExecutionBackend() {}
};
void visitBareFunctions(const std::function<void(std::string, void*)>& fn);
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_BACKEND_H
