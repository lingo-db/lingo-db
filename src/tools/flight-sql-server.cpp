#include "features.h"

#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include <arrow/flight/server.h>
#include <arrow/flight/sql/server.h>
#include <arrow/flight/types.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace flight = arrow::flight;
namespace flightsql = arrow::flight::sql;

namespace {

class LingoDBFlightSqlServer : public flightsql::FlightSqlServerBase {
   public:
   LingoDBFlightSqlServer(std::shared_ptr<lingodb::runtime::Session> session, std::string clientLocationUri)
      : session_(std::move(session)), clientLocationUri_(std::move(clientLocationUri)) {}

   arrow::Result<std::unique_ptr<flight::FlightInfo>> GetFlightInfoStatement(
      const flight::ServerCallContext&, const flightsql::StatementQuery& command,
      const flight::FlightDescriptor& descriptor) override {
      std::shared_ptr<arrow::Table> table;
      ARROW_RETURN_NOT_OK(ExecuteQuery(command.query, &table));

      std::shared_ptr<arrow::Schema> schema = table ? table->schema() : arrow::schema({});

      std::string handle = "lingodb-" + std::to_string(counter_.fetch_add(1));
      {
         std::lock_guard<std::mutex> lock(resultsMutex_);
         results_[handle] = table;
      }

      ARROW_ASSIGN_OR_RAISE(auto ticketStr, flightsql::CreateStatementQueryTicket(handle));
      flight::Ticket ticket{std::move(ticketStr)};

      std::vector<flight::Location> locations;
      ARROW_ASSIGN_OR_RAISE(auto loc, flight::Location::Parse(clientLocationUri_));
      locations.push_back(std::move(loc));
      flight::FlightEndpoint endpoint{std::move(ticket), std::move(locations), std::nullopt, ""};

      int64_t totalRecords = table ? table->num_rows() : 0;
      ARROW_ASSIGN_OR_RAISE(auto info,
                            flight::FlightInfo::Make(*schema, descriptor, {endpoint}, totalRecords, -1));
      return std::make_unique<flight::FlightInfo>(std::move(info));
   }

   arrow::Result<std::unique_ptr<flight::SchemaResult>> GetSchemaStatement(
      const flight::ServerCallContext&, const flightsql::StatementQuery&,
      const flight::FlightDescriptor&) override {
      // LingoDB compiles and executes together, so resolving just a schema
      // without executing the query is not cheap. Report UNIMPLEMENTED and
      // let clients discover the schema via GetFlightInfo instead.
      return arrow::Status::NotImplemented(
         "GetSchemaStatement is not supported by the LingoDB Flight SQL server; "
         "use GetFlightInfo to obtain both schema and data.");
   }

   arrow::Result<std::unique_ptr<flight::FlightDataStream>> DoGetStatement(
      const flight::ServerCallContext&, const flightsql::StatementQueryTicket& command) override {
      std::shared_ptr<arrow::Table> table;
      {
         std::lock_guard<std::mutex> lock(resultsMutex_);
         auto it = results_.find(command.statement_handle);
         if (it == results_.end()) {
            return arrow::Status::KeyError("Unknown statement handle: ", command.statement_handle);
         }
         table = std::move(it->second);
         results_.erase(it);
      }

      std::shared_ptr<arrow::Schema> schema = table ? table->schema() : arrow::schema({});
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      if (table && table->num_rows() > 0) {
         arrow::TableBatchReader batchReader(*table);
         ARROW_ASSIGN_OR_RAISE(batches, batchReader.ToRecordBatches());
      }
      ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make(std::move(batches), schema));
      return std::make_unique<flight::RecordBatchStream>(reader);
   }

   arrow::Result<int64_t> DoPutCommandStatementUpdate(
      const flight::ServerCallContext&, const flightsql::StatementUpdate& command) override {
      std::shared_ptr<arrow::Table> ignored;
      ARROW_RETURN_NOT_OK(ExecuteQuery(command.query, &ignored));
      // LingoDB does not report the number of affected rows; -1 ("unknown")
      // is a valid Flight SQL response here.
      return -1;
   }

   private:
   arrow::Status ExecuteQuery(const std::string& sql, std::shared_ptr<arrow::Table>* out) {
      // LingoDB's compilation pipeline and its context pre-warming are not
      // designed for concurrent queries from the same process; serialize.
      std::lock_guard<std::mutex> lock(executionMutex_);
      try {
         auto config = lingodb::execution::createQueryExecutionConfig(
            lingodb::execution::getExecutionMode(), /*sqlInput=*/true);
         config->resultProcessor = lingodb::execution::createTableRetriever(*out);
         auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(
            std::move(config), *session_);
         executer->setExitOnError(false);
         executer->fromData(sql);
         auto errorPtr = executer->getError(); // copy shared_ptr, retains after move
         lingodb::scheduler::awaitEntryTask(
            std::make_unique<lingodb::execution::QueryExecutionTask>(std::move(executer)));
         if (errorPtr && *errorPtr) {
            return arrow::Status::ExecutionError("LingoDB query failed: ", errorPtr->getMessage());
         }
         return arrow::Status::OK();
      } catch (const std::exception& e) {
         return arrow::Status::ExecutionError("LingoDB query threw: ", e.what());
      }
   }

   std::shared_ptr<lingodb::runtime::Session> session_;
   std::string clientLocationUri_;
   std::mutex executionMutex_;
   std::mutex resultsMutex_;
   std::unordered_map<std::string, std::shared_ptr<arrow::Table>> results_;
   std::atomic<uint64_t> counter_{0};
};

void printUsage(const char* prog) {
   std::cerr << "USAGE: " << prog << " <database-dir> [--host HOST] [--port PORT]\n"
             << "  --host HOST   bind address (default: 0.0.0.0)\n"
             << "  --port PORT   TCP port       (default: 31337)\n";
}

} // namespace

int main(int argc, char** argv) {
   using namespace lingodb;

   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }
   if (argc < 2) {
      printUsage(argv[0]);
      return 1;
   }

   std::string dbDir = argv[1];
   std::string host = "0.0.0.0";
   int port = 31337;
   for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--host" && i + 1 < argc) {
         host = argv[++i];
      } else if (arg == "--port" && i + 1 < argc) {
         port = std::atoi(argv[++i]);
      } else {
         printUsage(argv[0]);
         return 1;
      }
   }

   std::cerr << "Loading database from: " << dbDir << std::endl;
   auto session = runtime::Session::createSession(dbDir, /*eagerLoading=*/true);
   compiler::support::eval::init();
   auto scheduler = scheduler::startScheduler();

   std::string bindUri = "grpc://" + host + ":" + std::to_string(port);
   // Clients advertised through FlightEndpoint should connect to 127.0.0.1 when
   // the server binds to the wildcard address; otherwise reuse the bind host.
   std::string clientUri = "grpc://" + (host == "0.0.0.0" ? std::string("127.0.0.1") : host) +
                           ":" + std::to_string(port);

   auto bindLocationResult = flight::Location::Parse(bindUri);
   if (!bindLocationResult.ok()) {
      std::cerr << "Failed to parse bind URI " << bindUri << ": "
                << bindLocationResult.status().ToString() << std::endl;
      return 1;
   }
   flight::FlightServerOptions options(bindLocationResult.ValueOrDie());

   auto server = std::make_shared<LingoDBFlightSqlServer>(session, clientUri);
   if (auto st = server->Init(options); !st.ok()) {
      std::cerr << "Flight server Init failed: " << st.ToString() << std::endl;
      return 1;
   }
   if (auto st = server->SetShutdownOnSignals({SIGINT, SIGTERM}); !st.ok()) {
      std::cerr << "SetShutdownOnSignals failed: " << st.ToString() << std::endl;
      return 1;
   }

   std::cerr << "LingoDB Flight SQL server listening on " << bindUri << std::endl;
   if (auto st = server->Serve(); !st.ok()) {
      std::cerr << "Flight Serve failed: " << st.ToString() << std::endl;
      return 1;
   }
   return 0;
}
