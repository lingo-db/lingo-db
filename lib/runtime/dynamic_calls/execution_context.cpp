#include "runtime/execution_context.h"
#include "runtime/helpers.h"
EXPORT std::shared_ptr<arrow::Table>* rt_get_table(runtime::ExecutionContext* executionContext, runtime::Str tableName) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto table=(executionContext)->db->getTable(tableName);
   if(!table){
      throw std::runtime_error("could not find table");
   }
   return new std::shared_ptr<arrow::Table>(table);
}
EXPORT uint64_t rt_next_pow2(uint64_t v){
   v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}