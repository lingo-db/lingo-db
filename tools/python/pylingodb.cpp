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
   void addTable(pybind11::str name, pybind11::str metaData, pybind11::object table) {
      std::shared_ptr<arrow::Table> arrowTable = arrow::py::unwrap_table(table.ptr()).ValueOrDie();
      auto batchReader = std::make_shared<arrow::TableBatchReader>(*arrowTable);
      ArrowArrayStream arrayStream;
      if (!arrow::ExportRecordBatchReader(batchReader, &arrayStream).ok()) {
         std::cerr << "export failed" << std::endl;
      }
      std::string n = name;
      std::string mD = metaData;
      bridge::addTable(connection, n.c_str(), mD.c_str(), &arrayStream);
   }
   pybind11::handle sql(pybind11::str query) {
      std::string q = query;
      ArrowArrayStream stream;
      if (bridge::runSQL(connection, q.c_str(), &stream)) {
         std::shared_ptr<arrow::Table> result = arrow::ImportRecordBatchReader(&stream).ValueOrDie()->ToTable().ValueOrDie();
         if (result) return arrow::py::wrap_table(result);
      }
      return pybind11::handle();
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
      .def("sql", &ConnectionHandle::sql)
      .def("mlir", &ConnectionHandle::mlir)
      .def("add_table", &ConnectionHandle::addTable);
   m.def("connect_to_db", &connectToDB);
   m.def("in_memory", &inMemory);
}