#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "arrow/array.h"
#include "execution/Execution.h"
#include "md5.h"
#include "mlir-support/eval.h"
#include "runtime/ArrowDirDatabase.h"
#include "runtime/TableBuilder.h"
#include <arrow/pretty_print.h>
#include <arrow/table.h>

enum SortMode {
   NONE,
   SORT,
   SORTROWS
};
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
struct ResultHasher : public execution::ResultProcessor {
   //input
   SortMode sortMode;
   bool tsv;
   //output
   std::string hash = "";
   size_t numValues = 0;
   std::string lines = "";
   void process(runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<runtime::ResultTable>(0);
      if (!resultTable) {
         return; //todo: proper error handling
      }
      auto table = resultTable.value()->get();
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
         if (toHash[i] == "true") {
            toHash[i] = "t";
         }
         if (toHash[i] == "false") {
            toHash[i] = "f";
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
   }
};

std::vector<std::string> readTestFile(std::string path) {
   std::vector<std::string> res;
   std::string line;
   std::ifstream infile(path);
   while (std::getline(infile, line)) {
      res.push_back(line);
   }
   return res;
}
std::vector<std::string> filterLines(const std::vector<std::string>& lines) {
   std::vector<std::string> res;
   for (auto str : lines) {
      if (auto split = str.find('#'); split != std::string::npos)
         str.resize(split);

      while ((!str.empty()) && (str.back() == ' '))
         str.pop_back();
      res.push_back(str);
   }
   return res;
}
static std::vector<std::string> split(std::string_view input)
// Split the input into parts
{
   std::vector<std::string> result;
   std::string current;
   for (char c : input) {
      if (c == ' ') {
         if (!current.empty()) {
            result.push_back(current);
            current.clear();
         }
      } else {
         current.push_back(c);
      }
   }
   if (!current.empty()) {
      result.push_back(current);
   }
   return result;
}
void runStatement(runtime::ExecutionContext& context, const std::vector<std::string>& lines, size_t& line) {
   auto parts = split(lines[line]);
   line++;
   std::string statement;
   while (line < lines.size()) {
      if (lines[line].empty()) {
         line++;
         break;
      }
      statement += lines[line] + "\n";
      line++;
   }
   if (statement.starts_with("CREATE INDEX")) {
      return;
   }
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::DEFAULT, true);
   queryExecutionConfig->resultProcessor = std::unique_ptr<execution::ResultProcessor>();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(statement);
   executer->setExecutionContext(&context);
   executer->execute();
}
void runQuery(runtime::ExecutionContext& context, const std::vector<std::string>& lines, size_t& line) {
   auto parts = split(lines[line]);
   line++;
   std::string query;
   while (line < lines.size()) {
      if (lines[line] == "----") {
         line++;
         break;
      }
      query += lines[line] + "\n";
      line++;
   }
   SortMode sort = NONE;
   auto parseSort = [&](const std::string& s) {
      if (s == "nosort") {
         sort = NONE;
         return true;
      } else if (s == "valuesort") {
         sort = SORT;
         return true;
      } else if (s == "rowsort") {
         sort = SORTROWS;
         return true;
      } else {
         return false;
      }
   };
   bool tsv = parts.size() > 1 && parts[1] == "tsv";
   std::vector<std::function<bool(const std::string&)>> parsers = {parseSort};
   for (unsigned i = 2; i < parts.size(); i++) {
      for (const auto& parser : parsers) {
         if (parser(parts[i])) {
            break;
         }
      }
   }

   std::string expectedResult;
   while (line < lines.size()) {
      if (lines[line].empty()) {
         line++;
         break;
      }
      expectedResult += lines[line] + "\n";
      line++;
   }
   auto queryExecutionConfig = execution::createQueryExecutionConfig(execution::ExecutionMode::DEFAULT, true);
   auto resultHasher = std::make_unique<ResultHasher>();
   auto& resultHasherRef = *resultHasher;
   resultHasher->sortMode = sort;
   resultHasher->tsv = tsv;
   queryExecutionConfig->resultProcessor = std::move(resultHasher);

   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig));
   executer->fromData(query);
   executer->setExecutionContext(&context);
   executer->execute();
   std::string result = std::to_string(resultHasherRef.numValues) + " values hashing to " + resultHasherRef.hash + "\n";
   if (result != expectedResult && expectedResult != resultHasherRef.lines) {
      std::cout << "executing query:" << query << std::endl;
      std::cout << "expecting:" << expectedResult << std::endl;
      std::cerr << "ERROR: result did not match" << std::endl;
      std::cerr << "EXPECTED: \"" << expectedResult << "\"" << std::endl;
      std::cerr << "RESULT: \"" << result << "\"" << std::endl;
      std::cerr << "LINES: \"" << resultHasherRef.lines << "\"" << std::endl;
      exit(1);
   }
}
int main(int argc, char** argv) {
   runtime::ExecutionContext context;
   context.db = runtime::ArrowDirDatabase::empty();
   context.db->setPersist(false);
   support::eval::init();
   if (argc != 2) {
      std::cerr << "usage: sqllite-tester file" << std::endl;
      exit(1);
   }
   auto lines = filterLines(readTestFile(argv[1]));
   size_t line = 0;
   while (line < lines.size()) {
      auto parts = split(lines[line]);
      if (parts.empty()) {
         line++;
         continue;
      }
      if (parts[0] == "statement") {
         runStatement(context, lines, line);
      }
      if (parts[0] == "query") {
         runQuery(context, lines, line);
      }
      if (parts[0] == "hash-threshold") {
         line += 2;
      }
   }

   return 0;
}
