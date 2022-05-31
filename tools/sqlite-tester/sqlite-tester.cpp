#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "arrow/array.h"
#include "mlir-support/eval.h"
#include "runner/runner.h"
#include "runtime/ArrowDirDatabase.h"

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
   if(statement.starts_with("CREATE INDEX")){
      return;
   }
   std::cout << "executing statement:" << statement << std::endl;
   runner::Runner runner(runner::RunMode::DEFAULT);
   runner.loadSQL(statement, *context.db);
   runner.optimize(*context.db);
   runner.lower();
   runner.lowerToLLVM();
   runner.runJit(&context, 1, [](uint8_t*) {});
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
   runner::Runner::SortMode sort = runner::Runner::NONE;
   auto parseSort = [&](const std::string& s) {
      if (s == "nosort") {
         sort = runner::Runner::NONE;
         return true;
      } else if (s == "valuesort") {
         sort = runner::Runner::SORT;
         return true;
      } else if (s == "rowsort") {
         sort = runner::Runner::SORTROWS;
         return true;
      } else {
         return false;
      }
   };
   bool tsv = parts.size()>1&&parts[1]=="tsv";
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

   runner::Runner runner(runner::RunMode::DEFAULT);
   runner.loadSQL(query, *context.db);
   runner.optimize(*context.db);
   runner.lower();
   runner.lowerToLLVM();
   size_t runs = 1;
   size_t numValues = 0;
   std::string hash;
   std::string resultLines;
   runner.runJit(&context, runs, runner::Runner::hashResult(sort, numValues, hash, resultLines, tsv));
   std::string result = std::to_string(numValues) + " values hashing to " + hash + "\n";
   if (result != expectedResult && expectedResult != resultLines) {
      std::cout << "executing query:" << query << std::endl;
      std::cout << "expecting:" << expectedResult << std::endl;
      std::cerr << "ERROR: result did not match" << std::endl;
      std::cerr << "EXPECTED: \"" << expectedResult << "\"" << std::endl;
      std::cerr << "RESULT: \"" << result << "\"" << std::endl;
      std::cerr << "LINES: \"" << resultLines << "\"" << std::endl;
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
      if (parts[0] =="hash-threshold"){
         line+=2;
      }
   }

   return 0;
}

