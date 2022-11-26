#ifndef EXECUTION_RUNNER_H
#define EXECUTION_RUNNER_H

#include "runtime/ExecutionContext.h"
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mlir {
class ModuleOp;
} // namespace mlir
namespace runner {

using mainFnType = std::add_pointer<void()>::type;
using setExecutionContextFnType = std::add_pointer<void(runtime::ExecutionContext*)>::type;

class Error {
   bool present = false;
   std::stringstream message;

   public:
   std::string getMessage() { return message.str(); }
   operator bool() const {
      return present;
   }
   std::stringstream& emit() {
      present = true;
      return message;
   }
};
class Frontend {
   protected:
   runtime::Database* database;
   Error error;

   std::unordered_map<std::string, double> timing;

   public:
   runtime::Database* getDatabase() const {
      return database;
   }
   void setDatabase(runtime::Database* db) {
      Frontend::database = db;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   virtual void loadFromFile(std::string fileName) = 0;
   virtual void loadFromString(std::string data) = 0;
   virtual mlir::ModuleOp* getModule() = 0;
   virtual ~Frontend() {}
};
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
class QueryImplementor {
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
      QueryImplementor::database = db;
   }
   void disableVerification() {
      verify = false;
   }
   Error& getError() { return error; }
   virtual void implement(mlir::ModuleOp& moduleOp) = 0;
   virtual ~QueryImplementor() {}
};
class ExecutionBackend {
   protected:
   size_t numRepetitions = 1;
   std::unordered_map<std::string, double> timing;
   Error error;
   bool verify = true;

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
   virtual ~ExecutionBackend() {}
};
class ResultProcessor {
   public:
   virtual void process(runtime::ExecutionContext* executionContext) = 0;
   virtual ~ResultProcessor() {}
};
std::unique_ptr<ResultProcessor> createTablePrinter();
std::unique_ptr<ResultProcessor> createTableRetriever(std::shared_ptr<arrow::Table>& result);
struct QueryExecutionConfig {
   std::unique_ptr<Frontend> frontend;
   std::unique_ptr<QueryOptimizer> queryOptimizer;
   std::unique_ptr<QueryImplementor> queryImplementor;
   std::unique_ptr<ExecutionBackend> executionBackend;
   std::unique_ptr<ResultProcessor> resultProcessor;
};
enum class RunMode {
   SPEED = 0, //Aim for maximum speed (no verification of generated MLIR
   DEFAULT = 1, //Execute without introducing extra steps for debugging/profiling, but verify generated MLIR
   PERF = 2, //Profiling
   DEBUGGING = 3 //Make generated code debuggable
};
std::unique_ptr<QueryExecutionConfig> createQueryExecutionConfig(RunMode runMode, bool sqlInput);
RunMode getRunMode();

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
} //namespace runner
#endif // EXECUTION_RUNNER_H
