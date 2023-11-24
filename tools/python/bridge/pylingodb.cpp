#include "bridge.h"
#include <pybind11/pybind11.h>

#include <iostream>
#include <arrow/c/bridge.h>
#include <arrow/python/pyarrow.h>
#include <arrow/table.h>
class ConnectionHandle {
   bridge::Connection* connection;

   public:
   ConnectionHandle(bridge::Connection* connection) : connection(connection) {}
   void appendTable(pybind11::str name, pybind11::object table) {
      std::shared_ptr<arrow::Table> arrowTable = arrow::py::unwrap_table(table.ptr()).ValueOrDie();
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*arrowTable);
      ArrowArrayStream arrayStream;
      if (!arrow::ExportRecordBatchReader(batchReader, &arrayStream).ok()) {
         std::cerr << "export failed" << std::endl;
      }
      std::string n = name;
      bridge::appendTable(connection, n.c_str(), &arrayStream);
   }
   pybind11::handle sql_query(pybind11::str query) {
      std::string q = query;
      ArrowArrayStream stream;
      if (bridge::runSQL(connection, q.c_str(), &stream)) {
         std::shared_ptr<arrow::Table> result = arrow::ImportRecordBatchReader(&stream).ValueOrDie()->ToTable().ValueOrDie();
         if (result) return arrow::py::wrap_table(result);
      }
      return pybind11::handle();
   }
   void sql_stmt(pybind11::str query) {
      std::string q = query;
      ArrowArrayStream stream;
      bridge::runSQL(connection, q.c_str(), &stream);
   }
   pybind11::handle mlir(pybind11::str module) {
      std::string m = module;
      ArrowArrayStream stream;
      if (bridge::run(connection, m.c_str(), &stream)) {
         std::shared_ptr<arrow::Table> result = arrow::ImportRecordBatchReader(&stream).ValueOrDie()->ToTable().ValueOrDie();
         if (result) return arrow::py::wrap_table(result);
      }
      return pybind11::handle();
   }
   void mlir_no_result(pybind11::str module) {
      std::string m = module;
      ArrowArrayStream stream;
      bridge::run(connection, m.c_str(), &stream);
   }
   void createTable(pybind11::str name, pybind11::str metaData) {
      std::string m = metaData;
      std::string n = name;
      bridge::createTable(connection, n.c_str(), m.c_str());
   }
   double getTime(pybind11::str type) {
      std::string t = type;
      return bridge::getTiming(connection, t.c_str());
   }
   ~ConnectionHandle() {
      bridge::closeConnection(connection);
   }
};

std::unique_ptr<ConnectionHandle> connectToDB(pybind11::str directory) {
   std::string d = directory;
   return std::make_unique<ConnectionHandle>(bridge::loadFromDisk(d.c_str()));
}
std::unique_ptr<ConnectionHandle> inMemory() {
   return std::make_unique<ConnectionHandle>(bridge::createInMemory());
}

PYBIND11_MODULE(ext, m) {
   if (arrow::py::import_pyarrow())
      throw std::runtime_error("Failed to initialize PyArrow");
   pybind11::class_<ConnectionHandle>(m, "ConnectionHandle")
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