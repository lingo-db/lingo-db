#include <iomanip>
#include <iostream>

#include <arrow/pretty_print.h>
#include <arrow/table.h>

#include "execution/ResultProcessing.h"
#include "runtime/TableBuilder.h"
#include <functional>

static unsigned char hexval(unsigned char c) {
   if ('0' <= c && c <= '9')
      return c - '0';
   else if ('a' <= c && c <= 'f')
      return c - 'a' + 10;
   else if ('A' <= c && c <= 'F')
      return c - 'A' + 10;
   else
      abort();
}

class TableRetriever : public execution::ResultProcessor {
   std::shared_ptr<arrow::Table>& result;

   public:
   TableRetriever(std::shared_ptr<arrow::Table>& result) : result(result) {}
   void process(runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<runtime::ResultTable>(0);
      if (!resultTable) return;
      result = resultTable.value()->get();
   }
};
std::unique_ptr<execution::ResultProcessor> execution::createTableRetriever(std::shared_ptr<arrow::Table>& result) {
   return std::make_unique<TableRetriever>(result);
}

class TablePrinter : public execution::ResultProcessor {
   void process(runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<runtime::ResultTable>(0);
      if (!resultTable) return;
      auto table = resultTable.value()->get();
      std::vector<std::string> columnReps;
      std::vector<size_t> positions;
      arrow::PrettyPrintOptions options;
      options.indent_size = 0;
      options.window = 100;
      std::cout << "|";
      std::string rowSep = "-";
      std::vector<bool> convertHex;
      for (auto c : table->columns()) {
         std::cout << std::setw(30) << table->schema()->field(positions.size())->name() << "  |";
         convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
         rowSep += std::string(33, '-');
         std::stringstream sstr;
         arrow::PrettyPrint(*c.get(), options, &sstr); //NOLINT (clang-diagnostic-unused-result)
         columnReps.push_back(sstr.str());
         positions.push_back(0);
      }
      std::cout << std::endl
                << rowSep << std::endl;
      bool cont = true;
      while (cont) {
         cont = false;
         bool skipNL = false;
         for (size_t column = 0; column < columnReps.size(); column++) {
            char lastHex = 0;
            bool first = true;
            std::stringstream out;
            while (positions[column] < columnReps[column].size()) {
               cont = true;
               char curr = columnReps[column][positions[column]];
               char next = columnReps[column][positions[column] + 1];
               positions[column]++;
               if (first && (curr == '[' || curr == ']' || curr == ',')) {
                  continue;
               }
               if (curr == ',' && next == '\n') {
                  continue;
               }
               if (curr == '\n') {
                  break;
               } else {
                  if (convertHex[column]) {
                     if (std::isxdigit(curr)) {
                        if (lastHex == 0) {
                           first = false;
                           lastHex = curr;
                        } else {
                           char converted = (hexval(lastHex) << 4 | hexval(curr));
                           out << converted;
                           lastHex = 0;
                        }
                     } else {
                        first = false;
                        out << curr;
                     }
                  } else {
                     first = false;
                     out << curr;
                  }
               }
            }
            if (first) {
               skipNL = true;
            } else {
               if (column == 0) {
                  std::cout << "|";
               }
               std::cout << std::setw(30) << out.str() << "  |";
            }
         }
         if (!skipNL) {
            std::cout << "\n";
         }
      }
   }
};
std::unique_ptr<execution::ResultProcessor> execution::createTablePrinter() {
   return std::make_unique<TablePrinter>();
}
