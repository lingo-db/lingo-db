#ifndef EXECUTION_BACKEND_H
#define EXECUTION_BACKEND_H
#include "Error.h"
#include "runtime/ExecutionContext.h"
namespace mlir {
class ModuleOp;
} // namespace mlir
namespace execution {
using mainFnType = std::add_pointer<void()>::type;
using setExecutionContextFnType = std::add_pointer<void(runtime::ExecutionContext*)>::type;
class ExecutionBackend {
   protected:
   size_t numRepetitions = 1;
   std::unordered_map<std::string, double> timing;
   Error error;
   bool verify = true;
   size_t snapShotCounter;

   public:
   size_t getNumRepetitions() const {
      return numRepetitions;
   }
   void setNumRepetitions(size_t numRepetitions) {
      ExecutionBackend::numRepetitions = numRepetitions;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   void disableVerification() {
      verify = false;
   }
   virtual void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) = 0;
   virtual bool requiresSnapshotting() = 0;
   void setSnapShotCounter(size_t snapShotCounter) {
      ExecutionBackend::snapShotCounter = snapShotCounter;
   }
   virtual ~ExecutionBackend() {}
};
void visitBareFunctions(const std::function<void(std::string, void*)>& fn);
} // namespace execution
#endif //EXECUTION_BACKEND_H
