#include "arrow/pretty_print.h"
#include <iomanip>
#include <iostream>
#include <runner/runner.h>
void runner::Runner::printTable(uint8_t* ptr) {
   auto table = *(std::shared_ptr<arrow::Table>*) ptr;
   std::vector<std::string> columnReps;
   std::vector<size_t> positions;
   arrow::PrettyPrintOptions options;
   options.indent_size = 0;
   std::cout << "|";
   std::string rowSep = "-";
   for (auto c : table->columns()) {
      std::cout << std::setw(30) << table->schema()->field(positions.size())->name() << "  |";
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
               first = false;
               out << curr;
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