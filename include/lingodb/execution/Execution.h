#ifndef LINGODB_EXECUTION_EXECUTION_H
#define LINGODB_EXECUTION_EXECUTION_H
#include "Backend.h"
#include "Error.h"
#include "Frontend.h"
#include "Instrumentation.h"
#include "ResultProcessing.h"
#include "Timing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/scheduler/Tasks.h"
#include <functional>
namespace mlir {
class ModuleOp;
} // namespace mlir
namespace lingodb::execution {
class QueryOptimizer {
   protected:
   runtime::Catalog* catalog;
   std::unordered_map<std::string, double> timing;
   Error error;
   bool verify = true;
   std::shared_ptr<SnapshotState> serializationState;

   public:
   runtime::Catalog* getDatabase() const {
      return catalog;
   }
   void setCatalog(runtime::Catalog* catalog) {
      QueryOptimizer::catalog = catalog;
   }
   void setSerializationState(std::shared_ptr<SnapshotState> serializationState) {
      QueryOptimizer::serializationState = serializationState;
   }
   auto getSerializationState() {
      return serializationState;
   }

   void disableVerification() {
      verify = false;
   }

   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   virtual void optimize(mlir::ModuleOp& moduleOp) = 0;
   virtual ~QueryOptimizer() {}
};
class LoweringStep {
   protected:
   runtime::Catalog* catalog;
   std::unordered_map<std::string, double> timing;
   Error error;
   bool verify = true;
   std::shared_ptr<SnapshotState> serializationState;

   public:
   virtual std::string getShortName() const = 0;
   runtime::Catalog* getCatalog() const {
      return catalog;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   void setCatalog(runtime::Catalog* catalog) {
      LoweringStep::catalog = catalog;
   }
   void setSerializationState(std::shared_ptr<SnapshotState> serializationState) {
      LoweringStep::serializationState = serializationState;
   }
   auto getSerializationState() {
      return serializationState;
   }

   void disableVerification() {
      verify = false;
   }
   Error& getError() { return error; }
   virtual void implement(mlir::ModuleOp& moduleOp) = 0;
   virtual ~LoweringStep() {}
};

struct QueryExecutionConfig {
   std::unique_ptr<Frontend> frontend;
   std::unique_ptr<QueryOptimizer> queryOptimizer;
   std::vector<std::unique_ptr<LoweringStep>> loweringSteps;
   std::unique_ptr<ExecutionBackend> executionBackend;
   std::unique_ptr<ResultProcessor> resultProcessor;
   std::unique_ptr<TimingProcessor> timingProcessor;
   bool trackTupleCount = false;
   bool parallel = true;
};

enum class ExecutionMode {
   SPEED = 0, //Aim for maximum speed (no verification of generated MLIR
   DEFAULT = 1, //Execute without introducing extra steps for debugging/profiling, but verify generated MLIR
   PERF = 2, //Profiling
   DEBUGGING = 3, //Make generated code debuggable
   CHEAP = 4, // compile as cheap (compile time) as possible
   EXTREME_CHEAP = 5, // compile as cheap (compile time) as possible, don't verify MLIR module
   C = 6,
   GPU = 7, // compile with support for GPUs
   NONE = 8
};
std::unique_ptr<QueryExecutionConfig> createQueryExecutionConfig(ExecutionMode runMode, bool sqlInput);
ExecutionMode getExecutionMode();

class QueryExecuter {
   protected:
   std::unique_ptr<QueryExecutionConfig> queryExecutionConfig;
   std::unique_ptr<runtime::ExecutionContext> executionContext;
   std::optional<std::string> data;
   std::optional<std::string> file;

   public:
   QueryExecuter(std::unique_ptr<QueryExecutionConfig> queryExecutionConfig, std::unique_ptr<runtime::ExecutionContext> executionContext) : queryExecutionConfig(std::move(queryExecutionConfig)), executionContext(std::move(executionContext)), data(), file() {}
   void fromData(std::string data) {
      this->data = data;
   }
   void fromFile(std::string file) {
      this->file = file;
   }
   virtual void execute() = 0;
   QueryExecutionConfig& getConfig() { return *queryExecutionConfig; }
   static std::unique_ptr<QueryExecuter> createDefaultExecuter(std::unique_ptr<QueryExecutionConfig> queryExecutionConfig, runtime::Session& session);
   virtual ~QueryExecuter() {}
   runtime::ExecutionContext* getExecutionContext() {
      return executionContext.get();
   }
};
class QueryExecutionTask : public lingodb::scheduler::TaskWithContext {
   std::unique_ptr<QueryExecuter> queryExecutor;
   std::function<void()> beforeDestroyFn;

   public:
   QueryExecutionTask(std::unique_ptr<QueryExecuter> queryExecutor, std::function<void()> beforeDestroyFn = nullptr) : TaskWithContext(queryExecutor->getExecutionContext()), queryExecutor(std::move(queryExecutor)), beforeDestroyFn(beforeDestroyFn) {}
   bool allocateWork() override {
      if (workExhausted.exchange(true)) {
         return false;
      }
      return true;
   }
   void performWork() override {
      queryExecutor->execute();
      if (beforeDestroyFn != nullptr) {
         beforeDestroyFn();
      }
   }
};

} // namespace lingodb::execution

#endif //LINGODB_EXECUTION_EXECUTION_H