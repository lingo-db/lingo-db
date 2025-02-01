#include "bridge.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <iostream>
#include <arrow/c/bridge.h>
#include <arrow/python/pyarrow.h>
#include <arrow/table.h>

namespace nb = nanobind;
class ConnectionHandle {
   bridge::Connection* connection;

   public:
   ConnectionHandle(bridge::Connection* connection) : connection(connection) {}
   void appendTable(std::string name, nb::object table) {
      std::shared_ptr<arrow::Table> arrowTable = arrow::py::unwrap_table(table.ptr()).ValueOrDie();
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*arrowTable);
      ArrowArrayStream arrayStream;
      if (!arrow::ExportRecordBatchReader(batchReader, &arrayStream).ok()) {
         std::cerr << "export failed" << std::endl;
      }
      std::string n = name;
      bridge::appendTable(connection, n.c_str(), &arrayStream);
   }
   nb::handle sql_query(std::string query) {
      std::string q = query;
      ArrowArrayStream stream;
      if (bridge::runSQL(connection, q.c_str(), &stream)) {
         std::shared_ptr<arrow::Table> result = arrow::ImportRecordBatchReader(&stream).ValueOrDie()->ToTable().ValueOrDie();
         if (result) return arrow::py::wrap_table(result);
      }
      return nb::handle();
   }
   void sql_stmt(std::string query) {
      std::string q = query;
      ArrowArrayStream stream;
      bridge::runSQL(connection, q.c_str(), &stream);
   }
   nb::handle mlir(std::string module) {
      std::string m = module;
      ArrowArrayStream stream;
      if (bridge::run(connection, m.c_str(), &stream)) {
         std::shared_ptr<arrow::Table> result = arrow::ImportRecordBatchReader(&stream).ValueOrDie()->ToTable().ValueOrDie();
         if (result) return arrow::py::wrap_table(result);
      }
      return nb::handle();
   }
   void mlir_no_result(std::string module) {
      std::string m = module;
      ArrowArrayStream stream;
      bridge::run(connection, m.c_str(), &stream);
   }
   void createTable(std::string name, std::string metaData) {
      std::string m = metaData;
      std::string n = name;
      bridge::createTable(connection, n.c_str(), m.c_str());
   }
   double getTime(std::string type) {
      std::string t = type;
      return bridge::getTiming(connection, t.c_str());
   }
   ~ConnectionHandle() {
      bridge::closeConnection(connection);
   }
};

std::unique_ptr<ConnectionHandle> connectToDB(std::string directory) {
   std::string d = directory;
   return std::make_unique<ConnectionHandle>(bridge::loadFromDisk(d.c_str()));
}
std::unique_ptr<ConnectionHandle> inMemory() {
   return std::make_unique<ConnectionHandle>(bridge::createInMemory());
}

NB_MODULE(ext, m) {
   if (arrow::py::import_pyarrow())
      throw std::runtime_error("Failed to initialize PyArrow");
   nb::class_<ConnectionHandle>(m, "ConnectionHandle")
      .def("sql_query", &ConnectionHandle::sql_query)
      .def("sql_stmt", &ConnectionHandle::sql_stmt)
      .def("mlir", &ConnectionHandle::mlir)
      .def("mlir_no_result", &ConnectionHandle::mlir_no_result)
      .def("append", &ConnectionHandle::appendTable)
      .def("create_table", &ConnectionHandle::createTable)
      .def("get_time", &ConnectionHandle::getTime);
   m.def("connect_to_db", &connectToDB);
   m.def("in_memory", &inMemory);
}