#include <iomanip>
#include <iostream>

#include <arrow/pretty_print.h>
#include <arrow/table.h>

#include "md5.h"
#include "runner/runner.h"
#include <functional>

unsigned char hexval(unsigned char c) {
   if ('0' <= c && c <= '9')
      return c - '0';
   else if ('a' <= c && c <= 'f')
      return c - 'a' + 10;
   else if ('A' <= c && c <= 'F')
      return c - 'A' + 10;
   else
      abort();
}
std::function<void(uint8_t*)> runner::Runner::hashResult(runner::Runner::SortMode sortMode, size_t& numValues, std::string& hash, std::string& lines, bool tsv) {
   return [sortMode = sortMode, tsv, &hash, &numValues, &lines](uint8_t* ptr) {
      auto table = *(std::shared_ptr<arrow::Table>*) ptr;
      std::vector<std::string> toHash;
      std::vector<std::string> columnReps;
      std::vector<size_t> positions;
      arrow::PrettyPrintOptions options;
      options.indent_size = 0;
      options.window = 1000000;
      std::vector<bool> convertHex;
      for (auto c : table->columns()) {
         convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
         std::stringstream sstr;
         arrow::PrettyPrint(*c.get(), options, &sstr); //NOLINT (clang-diagnostic-unused-result)
         columnReps.push_back(sstr.str());
         positions.push_back(0);
      }

      bool cont = true;
      while (cont) {
         cont = false;
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
            if (!first) {
               toHash.push_back(out.str());
            }
         }
      }
      for (size_t i = 0; i < toHash.size(); i++) {
         if (toHash[i] == "null") {
            toHash[i] = "NULL";
         }
         if(toHash[i]=="true"){
            toHash[i]="t";
         }
         if(toHash[i]=="false"){
            toHash[i]="f";
         }
         if (toHash[i].starts_with("\"") && toHash[i].ends_with("\"")) {
            toHash[i] = toHash[i].substr(1, toHash[i].size() - 2);
         }
      }
      size_t numColumns = table->num_columns();
      if (sortMode == SORTROWS) {
         std::vector<std::vector<std::string>> rows;
         std::vector<std::string> row;
         for (auto& s : toHash) {
            row.push_back(s);
            if (row.size() == numColumns) {
               rows.push_back(row);
               row.clear();
            }
         }
         std::sort(rows.begin(), rows.end());
         toHash.clear();
         for (auto& r : rows)
            for (auto& v : r)
               toHash.push_back(v);
      } else if (sortMode == SORT) {
         std::sort(toHash.begin(), toHash.end());
      }
      hash = md5Strings(toHash);
      numValues = toHash.size();
      std::string linesRes;
      if (toHash.size() < 1000) {
         if (tsv) {
            size_t i = 0;
            for (auto x : toHash) {
               linesRes += x + ((((i + 1) % numColumns) == 0) ? "\n" : "\t");
               i++;
            }
         } else {
            for (auto x : toHash) {
               linesRes += x + "\n";
            }
         }
      }
      lines = linesRes;
      //std::cout << toHash.size() << " values hashing to " <<  << std::endl;
   };
}
void runner::Runner::printTable(uint8_t* ptr) {
   auto table = *(std::shared_ptr<arrow::Table>*) ptr;
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