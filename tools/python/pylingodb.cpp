#include "bridge.h"
#include <pybind11/pybind11.h>

#include <arrow/python/pyarrow.h>
#include <arrow/c/bridge.h>
#include <arrow/table.h>
#include <iostream>

void load(pybind11::str name, pybind11::object table) {
   std::shared_ptr<arrow::Table> arrowTable = arrow::py::unwrap_table(table.ptr()).ValueOrDie();
   auto batchReader=std::make_shared<arrow::TableBatchReader>(*arrowTable);
   ArrowArrayStream arrayStream;
   if(!arrow::ExportRecordBatchReader(batchReader,&arrayStream).ok()){
      std::cerr<<"export failed"<<std::endl;
   }
   std::string n=name;
   bridge::addTable(n.c_str(),&arrayStream);
}
void runSQL(pybind11::str query){
   std::string q=query;
   bridge::runSQL(q.c_str());

}
void run(pybind11::str module){
   std::string m=module;
   bridge::run(m.c_str());
}
void loadFromDisk(pybind11::str directory){
   std::string d=directory;
   bridge::loadFromDisk(d.c_str());
}


PYBIND11_MODULE(ext, m) {
   if (arrow::py::import_pyarrow())
      throw std::runtime_error("Failed to initialize PyArrow");
   m.def("createInMemory", &bridge::createInMemory);
   m.def("run", &run);
   m.def("load", &load);
   m.def("loadFromDisk", &loadFromDisk);
   m.def("runSQL", &runSQL);
}