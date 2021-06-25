#include "runtime/execution_context.h"
#include "runtime/helpers.h"
EXPORT runtime::Pointer<std::shared_ptr<arrow::Table>> _mlir_ciface_get_table(runtime::Pointer<runtime::ExecutionContext>* executionContext, runtime::String* tableName) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   auto table=(*executionContext)->db->getTable(*tableName);
   if(!table){
      throw std::runtime_error("could not find table");
   }
   return new std::shared_ptr<arrow::Table>(table);
}