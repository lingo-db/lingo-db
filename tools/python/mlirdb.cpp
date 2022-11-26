#include <arrow/pretty_print.h>
#include <arrow/python/pyarrow.h>
#include <pybind11/pybind11.h>

#include "execution/runner.h"
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
   auto queryExecutionConfig=runner::createQueryExecutionConfig(runner::RunMode::SPEED,false);
   std::shared_ptr<arrow::Table> result;
   queryExecutionConfig->resultProcessor=runner::createTableRetriever(result);
   auto executer=runner::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(module);
   executer->setExecutionContext(executionContext);
   executer->execute();
   if (result) return arrow::py::wrap_table(result);
   return pybind11::handle();
}

PYBIND11_MODULE(pymlirdbext, m) {
   m.def("load", &load);
   m.def("run", run);
}