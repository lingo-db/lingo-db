#include "bridge.h"

#include "lingodb/execution/Execution.h"

#include "mlir/CAPI/IR.h"

#include <arrow/table.h>

namespace {
using namespace lingodb;
class TimingCollector : public execution::TimingProcessor {
   std::unordered_map<std::string, double> collected;
   std::unordered_map<std::string, double>& target;

   public:
   explicit TimingCollector(std::unordered_map<std::string, double>& target) : target(target) {}
   virtual void addTiming(const std::unordered_map<std::string, double>& timing) override {
      this->collected.insert(timing.begin(), timing.end());
   }
   virtual void process() override {
      target = collected;
   }
   virtual ~TimingCollector() {}
};
} // namespace

namespace bridge {
class Connection {
   std::shared_ptr<runtime::Session> session;
   std::unordered_map<std::string, double> times;
   std::unique_ptr<scheduler::SchedulerHandle> scheduler;

   public:
   Connection(std::shared_ptr<runtime::Session> session) : session(session) {
      scheduler = lingodb::scheduler::startScheduler();
   }
   runtime::Session& getSession() {
      return *session;
   }
   std::unordered_map<std::string, double>& getTimes() {
      return times;
   }
};
} //namespace bridge

bridge::Connection* bridge::createInMemory() {
   return new Connection(runtime::Session::createSession());
}
bridge::Connection* bridge::loadFromDisk(const char* directory) {
   return new Connection(runtime::Session::createSession(directory, true));
}
bool bridge::run(Connection* connection, const char* module, ArrowArrayStream* res) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, false);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);
   queryExecutionConfig->timingProcessor = std::make_unique<TimingCollector>(connection->getTimes());
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), connection->getSession());
   executer->fromData(module);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*result);
      if (!arrow::ExportRecordBatchReader(batchReader, res).ok()) {
         std::cerr << "export failed" << std::endl;
      } else {
         return true;
      }
   }
   return false;
}
bool bridge::runSQL(Connection* connection, const char* query, ArrowArrayStream* res) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, true);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);
   queryExecutionConfig->timingProcessor = std::make_unique<TimingCollector>(connection->getTimes());
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), connection->getSession());
   executer->fromData(query);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*result);
      if (!arrow::ExportRecordBatchReader(batchReader, res).ok()) {
         std::cerr << "export failed" << std::endl;
      } else {
         return true;
      }
   }
   return false;
}
void bridge::createTable(Connection* connection, const char* name, const char* metaData) {
   connection->getSession().getCatalog()->addTable(name, runtime::TableMetaData::create(metaData, name, {}));
}
void bridge::appendTable(Connection* connection, const char* name, ArrowArrayStream* recordBatchStream) {
   auto recordBatchReader = arrow::ImportRecordBatchReader(recordBatchStream).ValueOrDie();
   connection->getSession().getCatalog()->findRelation(name)->append(recordBatchReader->ToTable().ValueOrDie());
}
double bridge::getTiming(bridge::Connection* con, const char* type) {
   if (con->getTimes().contains(std::string(type))) {
      return con->getTimes()[std::string(type)];
   } else {
      return NAN;
   }
}
void bridge::closeConnection(bridge::Connection* con) {
   delete con;
}

void bridge::initContext(MlirContext context) {
   execution::initializeContext(*unwrap(context));
}
namespace bridge {
struct MLIRValueRangeImpl {
   std::vector<MlirValue> values;
};

} // end namespace bridge

void bridge::addUDF(bridge::Connection* con, const char* name, bridge::UDF* udf) {
   std::cout << "add udf " << name << std::endl;
}
void bridge::addValueToRange(bridge::MLIRValueRange range, MlirValue value) {
   range.impl->values.push_back(value);
}
MlirValue bridge::valueRangeGet(bridge::MLIRValueRange range, size_t offset) {
   return range.impl->values.at(offset);
}
size_t bridge::valueRangeGetLen(bridge::MLIRValueRange range) {
   return range.impl->values.size();
}
bridge::MLIRValueRange bridge::createValueRange() {
   return {.impl = new MLIRValueRangeImpl};
}