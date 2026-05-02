#include "bridge.h"

#include "lingodb/execution/Execution.h"

#include "mlir/CAPI/IR.h"

#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/runtime/RelationHelper.h"
#include "lingodb/runtime/storage/TableStorage.h"
#include <arrow/table.h>
#ifdef USE_CPYTHON_RUNTIME
#include "Python.h"
#endif

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
   // Field declaration order is chosen so that the SchedulerHandle is
   // destroyed FIRST (it stops the worker threads) and only after that is
   // the session torn down — otherwise workers might still touch session
   // state while it is being freed. The scheduler-before-session
   // construction order is forced in the ctor body instead, since session
   // ctor needs the scheduler running to size per-worker python state.
   std::shared_ptr<runtime::Session> session;
   std::unordered_map<std::string, double> times;
   std::unique_ptr<scheduler::SchedulerHandle> scheduler;

   public:
   Connection() {
      lingodb::compiler::support::eval::init();
      scheduler = lingodb::scheduler::startScheduler();
      session = runtime::Session::createSession();
   }
   Connection(const char* directory) {
      lingodb::compiler::support::eval::init();
      scheduler = lingodb::scheduler::startScheduler();
      session = runtime::Session::createSession(directory, true);
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
   return new Connection();
}
bridge::Connection* bridge::loadFromDisk(const char* directory) {
   return new Connection(directory);
}
bool bridge::run(Connection* connection, const char* module, ArrowArrayStream* res) {
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::SPEED, false);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor = execution::createTableRetriever(result);
   queryExecutionConfig->timingProcessor = std::make_unique<TimingCollector>(connection->getTimes());
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), connection->getSession());
   executer->fromData(module);
#ifdef USE_CPYTHON_RUNTIME
   PyThreadState* _save = PyEval_SaveThread();
#endif
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
#ifdef USE_CPYTHON_RUNTIME
   PyEval_RestoreThread(_save);
#endif
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(result);
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
#ifdef USE_CPYTHON_RUNTIME
   PyThreadState* _save = PyEval_SaveThread();
#endif
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
#ifdef USE_CPYTHON_RUNTIME
   PyEval_RestoreThread(_save);
#endif
   if (result) {
      auto batchReader = std::make_shared<arrow::TableBatchReader>(result);
      if (!arrow::ExportRecordBatchReader(batchReader, res).ok()) {
         std::cerr << "export failed" << std::endl;
      } else {
         return true;
      }
   }
   return false;
}
void bridge::appendTable(Connection* connection, const char* name, ArrowArrayStream* recordBatchStream) {
   auto recordBatchReader = arrow::ImportRecordBatchReader(recordBatchStream).ValueOrDie();
   runtime::RelationHelper::appendToTable(connection->getSession(), name, recordBatchReader->ToTable().ValueOrDie());
}
double bridge::getTiming(bridge::Connection* con, const char* type) {
   if (con->getTimes().contains(std::string(type))) {
      return con->getTimes()[std::string(type)];
   } else {
      return NAN;
   }
}
void bridge::closeConnection(bridge::Connection* con) {
   // Release the host's GIL across destruction. Tearing down the Connection
   // joins the scheduler workers, and the workers' last actions on shutdown
   // (~MLIRContext drops cached shared_ptrs to catalog entries, whose
   // RecordBatches transitively own pyarrow-imported NumPy buffers; their
   // destructors call PyGILState_Ensure to drop the NumPy refs). If the
   // calling thread (nanobind tp_dealloc, which is the host's main thread
   // holding the main interpreter's GIL) keeps the GIL, those workers
   // block waiting for it and either deadlock or — worse — race with us
   // and corrupt the heap. ~Session then runs after the scheduler has
   // joined; with the GIL released here it sees the same "no tstate / no
   // GIL" entry state it sees in standalone, where its existing
   // PyThreadState_Swap + Py_EndInterpreter loop already works.
#ifdef USE_CPYTHON_RUNTIME
   PyThreadState* _save = PyEval_SaveThread();
#endif
   delete con;
#ifdef USE_CPYTHON_RUNTIME
   PyEval_RestoreThread(_save);
#endif
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