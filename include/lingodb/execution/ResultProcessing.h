#ifndef LINGODB_EXECUTION_RESULTPROCESSING_H
#define LINGODB_EXECUTION_RESULTPROCESSING_H
#include "lingodb/runtime/ExecutionContext.h"

#include <filesystem>

#include <arrow/csv/options.h>
#include <arrow/csv/writer.h>
namespace lingodb::execution {
class ResultProcessor {
   public:
   virtual void process(lingodb::runtime::ExecutionContext* executionContext) = 0;
   virtual ~ResultProcessor() {}
};
std::unique_ptr<ResultProcessor> createTablePrinter();
std::unique_ptr<ResultProcessor> createBatchedTablePrinter();

struct CSVConfig {
   char delimiter = ',';
   char quote = '"';
   char escape = '"';
   bool includeHeader = true;
   arrow::csv::QuotingStyle quotingStyle = arrow::csv::QuotingStyle::Needed;
   std::string eol = "\n";
   std::string filePath = (std::filesystem::temp_directory_path() / "lingodb_output.csv").string();
};

std::unique_ptr<ResultProcessor> createTableCSVPrinter(CSVConfig config = {});
std::unique_ptr<ResultProcessor> createBatchedTableCSVPrinter(CSVConfig config = {});

std::unique_ptr<ResultProcessor> createTableRetriever(std::shared_ptr<arrow::Table>& result);
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_RESULTPROCESSING_H