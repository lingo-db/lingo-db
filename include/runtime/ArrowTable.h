#ifndef RUNTIME_ARROWTABLE_H
#define RUNTIME_ARROWTABLE_H
#include "ThreadLocal.h"
#include "helpers.h"
#include "runtime/ArrowColumn.h"
#include <arrow/type_fwd.h>
namespace runtime {
class ArrowTable {
   std::shared_ptr<arrow::Table> table;

   public:
   ArrowTable(std::shared_ptr<arrow::Table> table) : table(table){};
   static ArrowTable* createEmpty();
   ArrowTable* addColumn(VarLen32 name, ArrowColumn* column);
   ArrowColumn* getColumn(VarLen32 name);
   ArrowTable* merge(ThreadLocal* threadLocal);
   std::shared_ptr<arrow::Table> get() const { return table; }
};
} // end namespace runtime
#endif //RUNTIME_ARROWTABLE_H
