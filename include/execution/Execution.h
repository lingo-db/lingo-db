#ifndef EXECUTION_EXECUTION_H
#define EXECUTION_EXECUTION_H
#include "Backend.h"
#include "Error.h"
#include "Frontend.h"
#include "ResultProcessing.h"
#include "Timing.h"
namespace mlir {
class ModuleOp;
} // namespace mlir
namespace execution {
class QueryOptimizer {
   protected:
   runtime::Database* database;
   std::unordered_map<std::string, double> timing;
   Error error;
   bool verify = true;

   public:
   runtime::Database* getDatabase() const {
      return database;
   }
   void setDatabase(runtime::Database* db) {
      QueryOptimizer::database = db;
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
   runtime::Database* database;
   std::unordered_map<std::string, double> timing;
   Error error;
   bool verify = true;

   public:
   runtime::Database* getDatabase() const {
      return database;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   void setDatabase(runtime::Database* db) {
      LoweringStep::database = db;
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
};

enum class ExecutionMode {
   SPEED = 0, //Aim for maximum speed (no verification of generated MLIR
   DEFAULT = 1, //Execute without introducing extra steps for debugging/profiling, but verify generated MLIR
   PERF = 2, //Profiling
   DEBUGGING = 3, //Make generated code debuggable
   CHEAP = 4, // compile as cheap (compile time) as possible
   EXTREME_CHEAP = 5, // compile as cheap (compile time) as possible, don't verify MLIR module
   C = 6,
};
std::unique_ptr<QueryExecutionConfig> createQueryExecutionConfig(ExecutionMode runMode, bool sqlInput);
ExecutionMode getExecutionMode();

class QueryExecuter {
   protected:
   std::unique_ptr<QueryExecutionConfig> queryExecutionConfig;
   runtime::ExecutionContext* executionContext;
   std::optional<std::string> data;
   std::optional<std::string> file;

   public:
   QueryExecuter(std::unique_ptr<QueryExecutionConfig> queryExecutionConfig) : queryExecutionConfig(std::move(queryExecutionConfig)), executionContext(nullptr), data(), file() {}
   void fromData(std::string data) {
      this->data = data;
   }
   void fromFile(std::string file) {
      this->file = file;
   }
   void setExecutionContext(runtime::ExecutionContext* executionContext) {
      this->executionContext = executionContext;
   }
   virtual void execute() = 0;
   QueryExecutionConfig& getConfig() { return *queryExecutionConfig; }
   static std::unique_ptr<QueryExecuter> createDefaultExecuter(std::unique_ptr<QueryExecutionConfig> queryExecutionConfig);
   virtual ~QueryExecuter() {}
};

} // namespace execution
#endif //EXECUTION_EXECUTION_H