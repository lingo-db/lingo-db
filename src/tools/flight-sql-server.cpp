#include "features.h"

#include "lingodb/catalog/Types.h"
#include "lingodb/compiler/frontend/ast/constant_value.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include <arrow/array.h>
#include <arrow/flight/server.h>
#include <arrow/flight/sql/server.h>
#include <arrow/flight/types.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace flight = arrow::flight;
namespace flightsql = arrow::flight::sql;

namespace {

/// Fallback parameter type when the frontend couldn't infer one.
std::shared_ptr<arrow::DataType> UnknownParameterType() {
   return arrow::utf8();
}

/// Map a LingoDB catalog::Type to an Arrow data type for use in a Flight SQL
/// parameter schema. Strict clients (Avatica / JDBC) build their binders from
/// this schema and reject cross-type `setXXX` calls, so the closer this is
/// to the real inferred type the more natural the client experience.
std::shared_ptr<arrow::DataType> LingoTypeToArrow(const lingodb::catalog::Type& t) {
   using lingodb::catalog::LogicalTypeId;
   switch (t.getTypeId()) {
      case LogicalTypeId::BOOLEAN: return arrow::boolean();
      case LogicalTypeId::INT: {
         auto info = const_cast<lingodb::catalog::Type&>(t).getInfo<lingodb::catalog::IntTypeInfo>();
         size_t width = info ? info->getBitWidth() : 32;
         bool isSigned = info ? info->getIsSigned() : true;
         if (isSigned) {
            if (width <= 8) return arrow::int8();
            if (width <= 16) return arrow::int16();
            if (width <= 32) return arrow::int32();
            return arrow::int64();
         }
         if (width <= 8) return arrow::uint8();
         if (width <= 16) return arrow::uint16();
         if (width <= 32) return arrow::uint32();
         return arrow::uint64();
      }
      case LogicalTypeId::FLOAT: return arrow::float32();
      case LogicalTypeId::DOUBLE: return arrow::float64();
      case LogicalTypeId::DATE: return arrow::date32();
      case LogicalTypeId::TIMESTAMP: return arrow::timestamp(arrow::TimeUnit::MICRO);
      case LogicalTypeId::STRING:
      case LogicalTypeId::CHAR: return arrow::utf8();
      case LogicalTypeId::DECIMAL: {
         auto info = const_cast<lingodb::catalog::Type&>(t).getInfo<lingodb::catalog::DecimalTypeInfo>();
         if (info) return arrow::decimal128(info->getPrecision(), info->getScale());
         return arrow::float64();
      }
      default: return UnknownParameterType();
   }
}

/// Convert one value from an Arrow Array (at `row`) into a LingoDB AST Value.
arrow::Result<std::shared_ptr<lingodb::ast::Value>> ArrayCellToAstValue(
   const arrow::Array& array, int64_t row) {
   if (array.IsNull(row)) {
      return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::NullValue>());
   }
   switch (array.type_id()) {
      case arrow::Type::BOOL:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::BoolValue>(
            static_cast<const arrow::BooleanArray&>(array).Value(row)));
      case arrow::Type::INT8:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::IntValue>(
            static_cast<const arrow::Int8Array&>(array).Value(row)));
      case arrow::Type::INT16:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::IntValue>(
            static_cast<const arrow::Int16Array&>(array).Value(row)));
      case arrow::Type::INT32:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::IntValue>(
            static_cast<const arrow::Int32Array&>(array).Value(row)));
      case arrow::Type::INT64: {
         int64_t v = static_cast<const arrow::Int64Array&>(array).Value(row);
         // LingoDB's IntValue is 32-bit signed; widen via FloatValue text when
         // it doesn't fit. Most Flight SQL clients send small integers as
         // int64 (ADBC + JDBC both do this).
         if (v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max()) {
            return std::shared_ptr<lingodb::ast::Value>(
               std::make_shared<lingodb::ast::IntValue>(static_cast<int>(v)));
         }
         return std::shared_ptr<lingodb::ast::Value>(
            std::make_shared<lingodb::ast::FloatValue>(std::to_string(v)));
      }
      case arrow::Type::UINT8:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::IntValue>(
            static_cast<const arrow::UInt8Array&>(array).Value(row)));
      case arrow::Type::UINT16:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::IntValue>(
            static_cast<const arrow::UInt16Array&>(array).Value(row)));
      case arrow::Type::UINT32:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::IntValue>(
            static_cast<int>(static_cast<const arrow::UInt32Array&>(array).Value(row))));
      case arrow::Type::UINT64:
         return std::shared_ptr<lingodb::ast::Value>(std::make_shared<lingodb::ast::UnsignedIntValue>(
            static_cast<const arrow::UInt64Array&>(array).Value(row)));
      case arrow::Type::FLOAT: {
         float v = static_cast<const arrow::FloatArray&>(array).Value(row);
         if (std::isnan(v) || std::isinf(v))
            return arrow::Status::Invalid("Cannot bind NaN/Inf as a SQL parameter.");
         std::ostringstream oss;
         oss << std::setprecision(9) << v;
         return std::shared_ptr<lingodb::ast::Value>(
            std::make_shared<lingodb::ast::FloatValue>(oss.str()));
      }
      case arrow::Type::DOUBLE: {
         double v = static_cast<const arrow::DoubleArray&>(array).Value(row);
         if (std::isnan(v) || std::isinf(v))
            return arrow::Status::Invalid("Cannot bind NaN/Inf as a SQL parameter.");
         std::ostringstream oss;
         oss << std::setprecision(17) << v;
         return std::shared_ptr<lingodb::ast::Value>(
            std::make_shared<lingodb::ast::FloatValue>(oss.str()));
      }
      case arrow::Type::STRING: {
         auto view = static_cast<const arrow::StringArray&>(array).GetView(row);
         return std::shared_ptr<lingodb::ast::Value>(
            std::make_shared<lingodb::ast::StringValue>(std::string(view)));
      }
      case arrow::Type::LARGE_STRING: {
         auto view = static_cast<const arrow::LargeStringArray&>(array).GetView(row);
         return std::shared_ptr<lingodb::ast::Value>(
            std::make_shared<lingodb::ast::StringValue>(std::string(view)));
      }
      case arrow::Type::DATE32: {
         int32_t days = static_cast<const arrow::Date32Array&>(array).Value(row);
         std::time_t secs = static_cast<std::time_t>(days) * 86400;
         std::tm tm{};
         gmtime_r(&secs, &tm);
         char buf[16];
         std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
         return std::shared_ptr<lingodb::ast::Value>(
            std::make_shared<lingodb::ast::DateValue>(buf));
      }
      case arrow::Type::DENSE_UNION: {
         // Clients that consume our `UnknownParameterType()` schema send
         // values inside a dense-union; unpack the active child.
         const auto& du = static_cast<const arrow::DenseUnionArray&>(array);
         int8_t typeCode = du.type_code(row);
         int childIdx = du.child_id(row);
         int32_t offset = du.value_offset(row);
         if (childIdx < 0) {
            return arrow::Status::Invalid(
               "Dense-union parameter had unknown type code ", static_cast<int>(typeCode));
         }
         return ArrayCellToAstValue(*du.field(childIdx), offset);
      }
      default:
         return arrow::Status::NotImplemented(
            "Flight SQL server cannot bind parameter of Arrow type ",
            array.type()->ToString());
   }
}

/// Convert one row of a RecordBatch into a list of LingoDB AST Values.
arrow::Result<std::vector<std::shared_ptr<lingodb::ast::Value>>> RowToAstValues(
   const arrow::RecordBatch& batch, int64_t row) {
   std::vector<std::shared_ptr<lingodb::ast::Value>> out;
   out.reserve(batch.num_columns());
   for (int c = 0; c < batch.num_columns(); ++c) {
      ARROW_ASSIGN_OR_RAISE(auto v, ArrayCellToAstValue(*batch.column(c), row));
      out.push_back(std::move(v));
   }
   return out;
}

class LingoDBFlightSqlServer : public flightsql::FlightSqlServerBase {
   public:
   LingoDBFlightSqlServer(std::shared_ptr<lingodb::runtime::Session> session, std::string clientLocationUri)
      : session_(std::move(session)), clientLocationUri_(std::move(clientLocationUri)) {}

   arrow::Result<std::unique_ptr<flight::FlightInfo>> GetFlightInfoStatement(
      const flight::ServerCallContext&, const flightsql::StatementQuery& command,
      const flight::FlightDescriptor& descriptor) override {
      std::shared_ptr<arrow::Table> table;
      ARROW_RETURN_NOT_OK(ExecuteQuery(command.query, {}, &table));

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
      return StreamFromTable(std::move(table));
   }

   arrow::Result<int64_t> DoPutCommandStatementUpdate(
      const flight::ServerCallContext&, const flightsql::StatementUpdate& command) override {
      std::shared_ptr<arrow::Table> ignored;
      ARROW_RETURN_NOT_OK(ExecuteQuery(command.query, {}, &ignored));
      return -1;
   }

   // ---- Prepared statements -------------------------------------------------
   //
   // `?` placeholders are parsed into `ParameterExpression` AST nodes by the
   // LingoDB frontend. On execution, we hand over the bound values as
   // `ast::Value`s via `Frontend::setParameters`; the analyzer resolves each
   // placeholder to a `ConstantExpression` inline. No SQL-text editing happens
   // in this server.

   arrow::Result<flightsql::ActionCreatePreparedStatementResult> CreatePreparedStatement(
      const flight::ServerCallContext&,
      const flightsql::ActionCreatePreparedStatementRequest& request) override {
      auto entry = std::make_shared<PreparedEntry>();
      entry->sql = request.query;
      entry->paramCount = CountPlaceholders(request.query);

      // The JDBC / Avatica client builds its Signature from `dataset_schema`
      // at prepare time and NPEs if it's missing. LingoDB has no separate
      // "plan" step, so we execute the query here (binding any placeholders
      // to NULL) purely to learn the output schema. For zero-parameter
      // statements we also keep the result around so the later
      // GetFlightInfo/DoGet cycle doesn't re-execute.
      std::vector<std::shared_ptr<lingodb::ast::Value>> probeParams(
         entry->paramCount, std::make_shared<lingodb::ast::NullValue>());

      std::shared_ptr<arrow::Table> probeTable;
      std::vector<std::optional<lingodb::catalog::Type>> inferredTypes;
      auto probeStatus = ExecuteQuery(entry->sql, std::move(probeParams), &probeTable, &inferredTypes);

      std::shared_ptr<arrow::Schema> datasetSchema;
      if (probeStatus.ok() && probeTable) {
         datasetSchema = probeTable->schema();
         if (entry->paramCount == 0) {
            entry->result = std::move(probeTable); // serve without re-executing
         }
      } else {
         datasetSchema = arrow::schema({});
      }

      std::string handle = "lingodb-prep-" + std::to_string(counter_.fetch_add(1));
      {
         std::lock_guard<std::mutex> lock(preparedMutex_);
         prepared_[handle] = entry;
      }

      arrow::FieldVector paramFields;
      paramFields.reserve(entry->paramCount);
      for (size_t i = 0; i < entry->paramCount; ++i) {
         std::shared_ptr<arrow::DataType> fieldType;
         if (i < inferredTypes.size() && inferredTypes[i].has_value()) {
            fieldType = LingoTypeToArrow(*inferredTypes[i]);
         } else {
            fieldType = UnknownParameterType();
         }
         paramFields.push_back(arrow::field("parameter_" + std::to_string(i + 1), fieldType));
      }
      return flightsql::ActionCreatePreparedStatementResult{
         /*dataset_schema=*/std::move(datasetSchema),
         /*parameter_schema=*/arrow::schema(paramFields),
         /*prepared_statement_handle=*/std::move(handle)};
   }

   arrow::Status ClosePreparedStatement(
      const flight::ServerCallContext&,
      const flightsql::ActionClosePreparedStatementRequest& request) override {
      std::lock_guard<std::mutex> lock(preparedMutex_);
      auto it = prepared_.find(request.prepared_statement_handle);
      if (it == prepared_.end()) {
         return arrow::Status::KeyError("Unknown prepared statement handle: ",
                                        request.prepared_statement_handle);
      }
      prepared_.erase(it);
      return arrow::Status::OK();
   }

   arrow::Status DoPutPreparedStatementQuery(
      const flight::ServerCallContext&, const flightsql::PreparedStatementQuery& command,
      flight::FlightMessageReader* reader, flight::FlightMetadataWriter*) override {
      ARROW_ASSIGN_OR_RAISE(auto entry, LookupPreparedStatement(command.prepared_statement_handle));
      return BindParametersFromReader(*entry, reader);
   }

   arrow::Result<std::unique_ptr<flight::FlightInfo>> GetFlightInfoPreparedStatement(
      const flight::ServerCallContext&, const flightsql::PreparedStatementQuery& command,
      const flight::FlightDescriptor& descriptor) override {
      ARROW_ASSIGN_OR_RAISE(auto entry, LookupPreparedStatement(command.prepared_statement_handle));
      std::vector<std::shared_ptr<lingodb::ast::Value>> params;
      std::string sql;
      std::shared_ptr<arrow::Table> cached;
      {
         std::lock_guard<std::mutex> lock(preparedMutex_);
         sql = entry->sql;
         params = entry->boundValues;
         cached = entry->result;
      }
      std::shared_ptr<arrow::Table> table;
      if (entry->paramCount == 0 && cached) {
         // CreatePreparedStatement already ran the zero-param query; reuse.
         table = cached;
      } else {
         ARROW_RETURN_NOT_OK(ExecuteQuery(sql, params, &table));
      }
      {
         std::lock_guard<std::mutex> lock(preparedMutex_);
         entry->result = table;
      }

      std::shared_ptr<arrow::Schema> schema = table ? table->schema() : arrow::schema({});
      // Round-trip the descriptor's command bytes as the ticket so the base
      // class's DoGet dispatches back to DoGetPreparedStatement.
      flight::Ticket ticket{descriptor.cmd};
      std::vector<flight::Location> locations;
      ARROW_ASSIGN_OR_RAISE(auto loc, flight::Location::Parse(clientLocationUri_));
      locations.push_back(std::move(loc));
      flight::FlightEndpoint endpoint{std::move(ticket), std::move(locations), std::nullopt, ""};

      int64_t totalRecords = table ? table->num_rows() : 0;
      ARROW_ASSIGN_OR_RAISE(auto info,
                            flight::FlightInfo::Make(*schema, descriptor, {endpoint}, totalRecords, -1));
      return std::make_unique<flight::FlightInfo>(std::move(info));
   }

   arrow::Result<std::unique_ptr<flight::FlightDataStream>> DoGetPreparedStatement(
      const flight::ServerCallContext&, const flightsql::PreparedStatementQuery& command) override {
      ARROW_ASSIGN_OR_RAISE(auto entry, LookupPreparedStatement(command.prepared_statement_handle));
      std::shared_ptr<arrow::Table> table;
      {
         std::lock_guard<std::mutex> lock(preparedMutex_);
         table = entry->result;
         entry->result.reset(); // force re-execution on next GetFlightInfo round
      }
      if (!table) {
         return arrow::Status::Invalid(
            "No result cached for prepared statement ", command.prepared_statement_handle,
            " — call GetFlightInfo before DoGet.");
      }
      return StreamFromTable(std::move(table));
   }

   arrow::Result<int64_t> DoPutPreparedStatementUpdate(
      const flight::ServerCallContext&, const flightsql::PreparedStatementUpdate& command,
      flight::FlightMessageReader* reader) override {
      ARROW_ASSIGN_OR_RAISE(auto entry, LookupPreparedStatement(command.prepared_statement_handle));
      ARROW_RETURN_NOT_OK(BindParametersFromReader(*entry, reader));
      std::shared_ptr<arrow::Table> ignored;
      ARROW_RETURN_NOT_OK(ExecuteQuery(entry->sql, entry->boundValues, &ignored));
      return -1;
   }

   private:
   struct PreparedEntry {
      std::string sql;
      size_t paramCount = 0;
      std::vector<std::shared_ptr<lingodb::ast::Value>> boundValues;
      std::shared_ptr<arrow::Table> result;
   };

   arrow::Result<std::shared_ptr<PreparedEntry>> LookupPreparedStatement(const std::string& handle) {
      std::lock_guard<std::mutex> lock(preparedMutex_);
      auto it = prepared_.find(handle);
      if (it == prepared_.end()) {
         return arrow::Status::KeyError("Unknown prepared statement handle: ", handle);
      }
      return it->second;
   }

   arrow::Status BindParametersFromReader(PreparedEntry& entry, flight::FlightMessageReader* reader) {
      ARROW_ASSIGN_OR_RAISE(auto batches, reader->ToRecordBatches());
      int64_t totalRows = 0;
      for (const auto& b : batches) totalRows += b->num_rows();

      // JDBC and some other clients drive even zero-parameter statements
      // through the prepared-statement path and send a schema-only batch
      // (one batch, zero rows). Accept that as "no values".
      if (entry.paramCount == 0) {
         std::lock_guard<std::mutex> lock(preparedMutex_);
         entry.boundValues.clear();
         return arrow::Status::OK();
      }

      if (totalRows == 0) {
         return arrow::Status::Invalid(
            "Prepared statement expected ", entry.paramCount,
            " parameter(s), but no parameter values were bound.");
      }
      if (totalRows != 1) {
         return arrow::Status::NotImplemented(
            "LingoDB Flight SQL server only supports single-row parameter binding; got ",
            totalRows, " rows.");
      }
      std::shared_ptr<arrow::RecordBatch> batch;
      for (const auto& b : batches) {
         if (b->num_rows() > 0) { batch = b; break; }
      }
      if (static_cast<size_t>(batch->num_columns()) != entry.paramCount) {
         return arrow::Status::Invalid(
            "Prepared statement expected ", entry.paramCount, " parameter(s), received ",
            batch->num_columns(), ".");
      }
      ARROW_ASSIGN_OR_RAISE(auto values, RowToAstValues(*batch, 0));
      std::lock_guard<std::mutex> lock(preparedMutex_);
      entry.boundValues = std::move(values);
      return arrow::Status::OK();
   }

   // Count `?` placeholders in SQL, skipping comments and quoted literals.
   static size_t CountPlaceholders(const std::string& sql) {
      size_t count = 0;
      const size_t n = sql.size();
      for (size_t i = 0; i < n; ++i) {
         char c = sql[i];
         if (c == '\'' || c == '"') {
            char quote = c;
            ++i;
            while (i < n) {
               if (sql[i] == quote) {
                  if (i + 1 < n && sql[i + 1] == quote) { i += 2; continue; }
                  break;
               }
               ++i;
            }
         } else if (c == '-' && i + 1 < n && sql[i + 1] == '-') {
            while (i < n && sql[i] != '\n') ++i;
         } else if (c == '/' && i + 1 < n && sql[i + 1] == '*') {
            i += 2;
            while (i + 1 < n && !(sql[i] == '*' && sql[i + 1] == '/')) ++i;
            ++i;
         } else if (c == '?') {
            ++count;
         }
      }
      return count;
   }

   arrow::Result<std::unique_ptr<flight::FlightDataStream>> StreamFromTable(
      std::shared_ptr<arrow::Table> table) {
      std::shared_ptr<arrow::Schema> schema = table ? table->schema() : arrow::schema({});
      std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
      if (table && table->num_rows() > 0) {
         arrow::TableBatchReader batchReader(*table);
         ARROW_ASSIGN_OR_RAISE(batches, batchReader.ToRecordBatches());
      }
      ARROW_ASSIGN_OR_RAISE(auto rbReader,
                            arrow::RecordBatchReader::Make(std::move(batches), schema));
      return std::make_unique<flight::RecordBatchStream>(rbReader);
   }

   arrow::Status ExecuteQuery(const std::string& sql,
                              std::vector<std::shared_ptr<lingodb::ast::Value>> params,
                              std::shared_ptr<arrow::Table>* out,
                              std::vector<std::optional<lingodb::catalog::Type>>* inferredParamTypes = nullptr) {
      std::lock_guard<std::mutex> lock(executionMutex_);
      try {
         auto config = lingodb::execution::createQueryExecutionConfig(
            lingodb::execution::getExecutionMode(), /*sqlInput=*/true);
         if (!params.empty()) {
            config->frontend->setParameters(std::move(params));
         }
         config->resultProcessor = lingodb::execution::createTableRetriever(*out);
         // Keep a raw handle to the frontend so a beforeDestroy hook can
         // snapshot the inferred parameter types before the task destroys
         // it. The executer takes ownership of the config (and hence the
         // frontend) via move below.
         auto* frontendPtr = config->frontend.get();
         auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(
            std::move(config), *session_);
         executer->setExitOnError(false);
         executer->fromData(sql);
         auto errorPtr = executer->getError();
         auto snapshotFn = [inferredParamTypes, frontendPtr]() {
            if (inferredParamTypes && frontendPtr) {
               *inferredParamTypes = frontendPtr->getInferredParameterTypes();
            }
         };
         lingodb::scheduler::awaitEntryTask(
            std::make_unique<lingodb::execution::QueryExecutionTask>(std::move(executer), snapshotFn));
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
   std::mutex preparedMutex_;
   std::unordered_map<std::string, std::shared_ptr<PreparedEntry>> prepared_;
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
