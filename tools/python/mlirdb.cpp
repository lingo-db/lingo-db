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
   runner.runJit(executionContext, 1);
   auto table = runner::Runner::getTable(executionContext);
   if (table) return arrow::py::wrap_table(table);
   return pybind11::handle();
}

PYBIND11_MODULE(pymlirdbext, m) {
   m.def("load", &load);
   m.def("run", run);
}