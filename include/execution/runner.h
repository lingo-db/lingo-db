#ifndef EXECUTION_RUNNER_H
#define EXECUTION_RUNNER_H

#include "runtime/ExecutionContext.h"
#include <functional>
#include <iomanip>
#include <iostream>
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
class ResultProcessor {
   public:
   virtual void process(runtime::ExecutionContext* executionContext) = 0;
   virtual ~ResultProcessor() {}
};
class TimingProcessor {
   public:
   virtual void addTiming(const std::unordered_map<std::string, double>& timing) = 0;
   virtual void process() = 0;
   virtual ~TimingProcessor() {}
};
class TimingPrinter : public TimingProcessor {
   std::unordered_map<std::string, double> timing;
   std::string queryName;

   public:
   TimingPrinter(std::string queryFile) {
      if (queryFile.find('/') != std::string::npos) {
         queryName = queryFile.substr(queryFile.find_last_of("/\\") + 1);
      } else {
         queryName = queryFile;
      }
   }
   void addTiming(const std::unordered_map<std::string, double>& timing) override {
      this->timing.insert(timing.begin(), timing.end());
   }
   void process() override {
      double total = 0.0;
      for (auto [name, t] : timing) {
         total += t;
      }
      timing["total"] = total;
      std::vector<std::string> printOrder = {"QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime", "total"};
      std::cout << std::endl
                << std::endl;
      std::cout << std::setw(10) << "name";
      for (auto n : printOrder) {
         std::cout << std::setw(15) << n;
      }
      std::cout << std::endl;
      std::cout << std::setw(10) << queryName;
      for (auto n : printOrder) {
         if (timing.contains(n)) {
            std::cout << std::setw(15) << timing[n];
         } else {
            std::cout << std::setw(15) << "";
         }
      }
   }
};
std::unique_ptr<ResultProcessor> createTablePrinter();
std::unique_ptr<ResultProcessor> createTableRetriever(std::shared_ptr<arrow::Table>& result);
struct QueryExecutionConfig {
   std::unique_ptr<Frontend> frontend;
   std::unique_ptr<QueryOptimizer> queryOptimizer;
   std::vector<std::unique_ptr<LoweringStep>> loweringSteps;
   std::unique_ptr<ExecutionBackend> executionBackend;
   std::unique_ptr<ResultProcessor> resultProcessor;
   std::unique_ptr<TimingProcessor> timingProcessor;
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
