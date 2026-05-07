#ifndef LINGODB_RUNTIME_ARROWTABLE_H
#define LINGODB_RUNTIME_ARROWTABLE_H
#include "ThreadLocal.h"
#include "helpers.h"
#include "lingodb/runtime/ArrowColumn.h"
#include <arrow/type_fwd.h>
namespace lingodb::catalog {
class Type;
} // namespace lingodb::catalog
namespace lingodb::runtime {
// Map a LingoDB logical Type to the Arrow physical type LingoDB stores it as
// (e.g. STRING → utf8, INT(32) → int32, DECIMAL(p,s) → decimal128(p,s)).
// Single source of truth for both the on-disk LingoDB table format and any
// runtime checks that need to reason about expected Arrow types.
std::shared_ptr<arrow::DataType> arrowPhysicalTypeFor(lingodb::catalog::Type t);

class ArrowTable {
   std::shared_ptr<arrow::Table> table;

   public:
   ArrowTable(std::shared_ptr<arrow::Table> table) : table(table) {};
   static ArrowTable* createEmpty();
   ArrowTable* addColumn(VarLen32 name, ArrowColumn* column);
   ArrowColumn* getColumn(VarLen32 name);
   ArrowTable* merge(ThreadLocal* threadLocal);
   std::shared_ptr<arrow::Table> get() const { return table; }
   // Validates that the wrapped arrow::Table's schema matches the expected
   // (column-name, lingodb-logical-type) list encoded in `descriptor` (a
   // hex-serialized `vector<pair<string, catalog::Type>>`). Throws a clear
   // std::runtime_error on mismatch; returns the input pointer unchanged on
   // success so the caller can chain it inline.
   static ArrowTable* verifySchema(ArrowTable* table, VarLen32 descriptor);
};
} // end namespace lingodb::runtime
#endif //LINGODB_RUNTIME_ARROWTABLE_H
