#include "runtime/execution_context.h"
#include "runtime/helpers.h"
EXPORT runtime::Pointer<std::shared_ptr<arrow::Table>> _mlir_ciface_get_table(runtime::Pointer<runtime::ExecutionContext>* executionContext, runtime::String* tableName) { // NOLINT (clang-diagnostic-return-type-c-linkage)

   return new std::shared_ptr<arrow::Table>((*executionContext)->db->getTable(*tableName));
}