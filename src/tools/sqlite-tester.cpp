#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/runtime/ArrowTable.h"

#include <arrow/array.h>
#include <arrow/pretty_print.h>
#include <arrow/table.h>

#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#include "md5.h"

namespace {
using namespace lingodb;
enum SortMode {
   NONE,
   SORT,
   SORTROWS
};
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
struct ResultHasher : public execution::ResultProcessor {
   //input
   SortMode sortMode;
   bool tsv;
   //output
   std::string hash;
   size_t numValues;
   std::string lines;
   ResultHasher() : hash(), numValues(0), lines() {}
   void process(runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<runtime::ArrowTable>(0);
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
      options.container_window = 1000000;
      options.element_size_limit = 10000;
      std::vector<bool> convertHex;
      std::vector<bool> isFloat;
      for (auto c : table->columns()) {
         convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
         isFloat.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::DOUBLE);
         std::stringstream sstr;
         arrow::PrettyPrint(*c.get(), options, &sstr); //NOLINT (clang-diagnostic-unused-result)
         columnReps.push_back(sstr.str());
         positions.push_back(0);
      }

      bool cont = true;
      while (cont) {
         cont = false;
         for (size_t column = 0; column < columnReps.size(); column++) {
            char32_t currChar = U'\0';
            uint8_t currCharSize = 0;

            bool first = true;
            bool afterComma = false;
            size_t digits = 0;
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
                  if (isFloat[column]) {
                     if (std::isdigit(curr)) {
                        if (afterComma && digits < 3) {
                           digits++;
                           out << curr;
                        } else if (!afterComma) {
                           out << curr;
                           first = false;
                        }
                     } else if (curr == '.') {
                        afterComma = true;
                        out << curr;
                        digits = 0;
                     } else {
                        afterComma = false;
                        digits = 0;
                        first = false;
                        out << curr;
                     }
                  } else if (convertHex[column]) {
                     first = false;
                     if (std::isxdigit(curr)) {
                        if (currCharSize % 2 == 0)
                           currChar |= hexval(curr) << (currCharSize++ * 4 + 4);
                        else
                           currChar |= hexval(curr) << (currCharSize++ * 4 - 4);
                     } else {
                        out << curr;
                     }
                  } else {
                     first = false;
                     out << curr;
                  }
               }
            }
            if (currChar != U'\0') {
               assert(currChar <= 0xFF && "Only ASCII characters supported for sqlite testing");
               out << static_cast<char>(currChar);
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
      std::string linesRes = "";
      if (toHash.size() < 1000 || tsv) {
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
   }
};

std::vector<std::string> readTestFile(std::string path) {
   std::vector<std::string> res;
   std::string line;
   std::ifstream infile(path);

   if (!infile) {
      std::cerr << "File could not be opened. Please verify the file path";
      exit(1);
   }

   while (std::getline(infile, line)) {
      res.push_back(line);
   }
   return res;
}
std::vector<std::string> filterLines(const std::vector<std::string>& lines) {
   std::vector<std::string> res;
   for (auto str : lines) {
      if (auto split = str.find('#'); split != std::string::npos) {
         if (split <= 1) {
            str.resize(split);
         }
      }

      while ((!str.empty()) && (str.back() == ' '))
         str.pop_back();
      res.push_back(str);
   }
   return res;
}
static std::vector<std::string> split(std::string_view input, char del = ' ')
// Split the input into parts
{
   std::vector<std::string> result;
   std::string current;
   for (char c : input) {
      if (c == del) {
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

void runStatement(runtime::Session& session, const std::vector<std::string>& lines, size_t& line) {
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
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), session);
   executer->fromData(statement);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
}
inline std::string& rtrim(std::string& s, const char* t) {
   s.erase(s.find_last_not_of(t) + 1);
   return s;
}
bool compareFuzzy(std::string expected, std::string result) {
   auto linesExpected = split(expected, '\n');
   auto linesResult = split(result, '\n');
   if (linesExpected.size() != linesResult.size()) {
      std::cerr << "different number of rows: " << linesExpected.size() << " vs " << linesResult.size() << std::endl;
      return false;
   }
   std::regex decRegex("(\\d+)\\.(\\d+)");
   std::regex zeroERegex("0\\.E-\\d+");
   std::regex zeroDecRegex("0\\.0+");

   for (auto i = 0ul; i < linesExpected.size(); i++) {
      auto expectedLine = linesExpected[i];
      auto resultLine = linesResult[i];
      auto expectedVals = split(expectedLine, '\t');
      auto resultVals = split(resultLine, '\t');
      if (expectedVals.size() != resultVals.size()) {
         std::cerr << "different number of cols: " << expectedVals.size() << " vs " << resultVals.size() << std::endl;
         return false;
      }
      for (auto j = 0ul; j < expectedVals.size(); j++) {
         auto expectedVal = rtrim(expectedVals[j], " ");
         auto resultVal = rtrim(resultVals[j], " ");
         if (expectedVal == resultVal) {
         } else {
            std::smatch expectedDecMatches;
            std::smatch resultDecMatches;
            if (std::regex_search(expectedVal, expectedDecMatches, decRegex) && std::regex_search(resultVal, resultDecMatches, decRegex)) {
               auto resultAfterComma = resultDecMatches[2].str();
               auto expectedAfterComma = expectedDecMatches[2].str();
               if (resultDecMatches[1] == expectedDecMatches[1] && (resultAfterComma.starts_with(expectedAfterComma) || expectedAfterComma.starts_with(resultAfterComma))) {
                  continue;
               }
               if (resultDecMatches[1] == expectedDecMatches[1] && resultAfterComma.length() > 4 && expectedAfterComma.length() > 4 && resultAfterComma.substr(0, 4) == expectedAfterComma.substr(0, 4)) {
                  continue;
               }
            }
            if (std::regex_match(expectedVal, zeroDecRegex) && std::regex_match(resultVal, zeroERegex)) {
               continue;
            }
            std::cerr << "did not match: '" << expectedVal << "' vs '" << resultVal << "'" << std::endl;
            return false;
         }
      }
   }
   return true;
}
void runQuery(runtime::Session& session, const std::vector<std::string>& lines, size_t& line) {
   auto description = lines[line];
   auto parts = split(description);
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
   auto queryExecutionConfig = execution::createQueryExecutionConfigWithNewFrontend(execution::getExecutionMode(), true);
   auto resultHasher = std::make_unique<ResultHasher>();
   auto& resultHasherRef = *resultHasher;
   resultHasher->sortMode = sort;
   resultHasher->tsv = tsv;
   queryExecutionConfig->resultProcessor = std::move(resultHasher);
   std::cerr << "executing:" << description << std::endl;

   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), session);
   executer->fromData(query);
   auto task = std::make_unique<execution::QueryExecutionTask>(std::move(executer), [&]() {
      std::string result = std::to_string(resultHasherRef.numValues) + " values hashing to " + resultHasherRef.hash + "\n";
      auto resultLines = std::regex_replace(resultHasherRef.lines, std::regex("\\s+\n"), "\n");
      if (result != expectedResult && !compareFuzzy(expectedResult, resultLines)) {
         std::cerr << "executing:" << description << std::endl;
         std::cerr << "ERROR: result did not match" << std::endl;
         std::cerr << "RESULT: \"" << result << "\"" << std::endl;
         std::cerr << "LINES: \"" << resultHasherRef.lines << "\"" << std::endl;
         exit(1);
      }
   });
   scheduler::awaitEntryTask(std::move(task));
}
} // namespace
int main(int argc, char** argv) {
   lingodb::compiler::support::eval::init();
   if (argc < 2 || argc > 3) {
      std::cerr << "usage: sqllite-tester file [dataset]" << std::endl;
      exit(1);
   }
   std::shared_ptr<runtime::Session> session;
   if (argc == 3) {
      session = runtime::Session::createSession(std::string(argv[2]), true);
   } else {
      session = runtime::Session::createSession();
   }
   auto scheduler = scheduler::startScheduler();
   auto lines = filterLines(readTestFile(argv[1]));
   size_t line = 0;
   bool first = true;
   while (line < lines.size()) {
      auto parts = split(lines[line]);
      if (parts.empty()) {
         line++;
         continue;
      }
      if (parts[0] == "statement") {
         try {
            runStatement(*session, lines, line);
         } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            std::cerr << "while executing statement: " << lines[line] << std::endl;
         }
      }
      if (parts[0] == "query") {
         if (first) {
            first = false;
            continue;
         }
         try {
            runQuery(*session, lines, line);
         } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            std::cerr << "while executing query: " << lines[line] << std::endl;
         }
      }
      if (parts[0] == "hash-threshold") {
         line += 2;
      }
   }

   return 0;
}
