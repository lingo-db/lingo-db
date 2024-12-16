#ifndef LINGODB_EXECUTION_RESULTPROCESSING_H
#define LINGODB_EXECUTION_RESULTPROCESSING_H
#include "lingodb/runtime/ExecutionContext.h"
namespace lingodb::execution {
class ResultProcessor {
   public:
   virtual void process(lingodb::runtime::ExecutionContext* executionContext) = 0;
   virtual ~ResultProcessor() {}
};
std::unique_ptr<ResultProcessor> createTablePrinter();
std::unique_ptr<ResultProcessor> createBatchedTablePrinter();
std::unique_ptr<ResultProcessor> createTableRetriever(std::shared_ptr<arrow::Table>& result);
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_RESULTPROCESSING_H
