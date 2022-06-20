#include <arrow/pretty_print.h>
#include <arrow/python/pyarrow.h>
#include <pybind11/pybind11.h>

#include "runner/runner.h"
#include "runtime/ExternalArrowDatabase.h"

runtime::ExecutionContext* executionContext;
void load(pybind11::dict dictionary) {
   executionContext = new runtime::ExecutionContext;
   auto database = std::make_unique<runtime::ExternalArrowDatabase>();
   for (auto item : dictionary) {
      std::string name = item.first.cast<pybind11::str>();
      auto arrowTable = arrow::py::unwrap_table(item.second.ptr()).ValueOrDie();
      database->addTable(name, arrowTable);
   };
   executionContext->db = std::move(database);
}
pybind11::handle run(pybind11::str module) {
   runner::Runner runner(runner::RunMode::SPEED);
   runner.loadString(module);
   //runner.dump();
   runner.optimize(*executionContext->db);
   //runner.dump();
   runner.lower();
   //runner.dump();
   runner.lowerToLLVM();
   //runner.dumpLLVM();
   pybind11::handle result;
   runner.runJit(executionContext, 1, [&](uint8_t* ptr) {
      auto table = *(std::shared_ptr<arrow::Table>*) ptr;
      //arrow::PrettyPrint(*table,arrow::PrettyPrintOptions(),&std::cout);
      result = arrow::py::wrap_table(table);
   });
   return result;
}


PYBIND11_MODULE(pymlirdbext,m) {
   m.def("load", &load);
   m.def("run", run);
}