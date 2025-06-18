#include "lingodb/runtime/ArrowTable.h"

#include "lingodb/utility/Tracer.h"

#include <arrow/table.h>
#include <lingodb/runtime/ExecutionContext.h>
using namespace lingodb::runtime;
namespace {
lingodb::utility::Tracer::Event tableMerge("ArrowTable", "merge");
} // end namespace
ArrowTable* ArrowTable::createEmpty() {
   auto table = arrow::Table::MakeEmpty(std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{})).ValueOrDie();
   auto* t = new ArrowTable(table);
   getCurrentExecutionContext()->registerState({t, [](void* ptr) { delete reinterpret_cast<ArrowTable*>(ptr); }});
   return t;
}
ArrowTable* ArrowTable::addColumn(VarLen32 name, ArrowColumn* column) {
   auto fields = this->table->schema()->fields();
   fields.push_back(arrow::field(name, column->getColumn()->type()));
   auto schema = std::make_shared<arrow::Schema>(fields);
   auto arrays = this->table->columns();
   arrays.push_back(column->getColumn());
   auto* t = new ArrowTable(arrow::Table::Make(schema, arrays));
   getCurrentExecutionContext()->registerState({t, [](void* ptr) { delete reinterpret_cast<ArrowTable*>(ptr); }});
   return t;
}
ArrowTable* ArrowTable::merge(ThreadLocal* threadLocal) {
   utility::Tracer::Trace trace(tableMerge);
   std::vector<std::shared_ptr<arrow::Table>> tables;
   for (auto* ptr : threadLocal->getThreadLocalValues<ArrowTable>()) {
      if (!ptr) continue;
      auto* current = ptr;
      tables.push_back(current->get());
   }
   auto concatenated = arrow::ConcatenateTables(tables).ValueOrDie();
   trace.stop();
   auto* t = new ArrowTable(concatenated);
   getCurrentExecutionContext()->registerState({t, [](void* ptr) { delete reinterpret_cast<ArrowTable*>(ptr); }});
   return t;
}
ArrowColumn* ArrowTable::getColumn(VarLen32 name) {
   auto* c = new ArrowColumn(get()->GetColumnByName(name.str()));
   getCurrentExecutionContext()->registerState({c, [](void* ptr) { delete reinterpret_cast<ArrowColumn*>(ptr); }});
   return c;
}
