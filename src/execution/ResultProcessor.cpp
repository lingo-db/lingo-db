#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <arrow/array.h>
#include <arrow/csv/options.h>
#include <arrow/csv/writer.h>
#include <arrow/ipc/writer.h>
#include <arrow/pretty_print.h>
#include <arrow/table.h>

#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/runtime/ArrowTable.h"

namespace {
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

class TableRetriever : public lingodb::execution::ResultProcessor {
   std::shared_ptr<arrow::Table>& result;

   public:
   TableRetriever(std::shared_ptr<arrow::Table>& result) : result(result) {}
   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(0);
      if (!resultTable) return;
      result = resultTable.value()->get();
   }
};

void printTable(const std::shared_ptr<arrow::Table>& table) {
   // Do not output anything for insert or copy statements
   if (table->columns().empty()) {
      std::cout << "Statement executed successfully." << std::endl;
      return;
   }

   std::vector<std::string> columnReps;
   std::vector<size_t> positions;
   arrow::PrettyPrintOptions options;
   options.indent_size = 0;
   options.window = 100;
   options.element_size_limit = 10000;
   std::cout << "|";
   std::string rowSep = "-";
   std::vector<bool> convertHex;
   for (auto c : table->columns()) {
      std::cout << std::setw(30) << table->schema()->field(positions.size())->name() << "  |";
      convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
      rowSep += std::string(33, '-');
      std::string str;
      (void) arrow::PrettyPrint(*c.get(), options, &str);
      columnReps.push_back(str);
      positions.push_back(0);
   }
   std::cout << std::endl
             << rowSep << std::endl;
   bool cont = true;
   while (cont) {
      cont = false;
      bool skipNL = false;
      for (size_t column = 0; column < columnReps.size(); column++) {
         char32_t currChar = U'\0';
         uint8_t currCharSize = 0;

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
               if (convertHex[column] && isxdigit(curr)) {
                  if (currCharSize % 2 == 0)
                     currChar |= hexval(curr) << (currCharSize++ * 4 + 4);
                  else
                     currChar |= hexval(curr) << (currCharSize++ * 4 - 4);
               } else if ((curr & (1 << 7)) == (1 << 7)) {
                  const auto extendedCurr = static_cast<char32_t>(curr) & 0xFF;
                  currChar |= static_cast<char32_t>(extendedCurr << (currCharSize * 4));
                  currCharSize += 2;
               } else {
                  if (currChar != U'\0') {
                     for (size_t i = 0; i < currCharSize / 2; i++) {
                        const char slice = reinterpret_cast<char*>(&currChar)[i];
                        if (slice != 0) {
                           out << slice;
                        }
                     }
                     currChar = U'\0';
                     currCharSize = 0;
                  }
                  out << curr;
               }
            }
         }
         if (currChar != U'\0') {
            for (size_t i = 0; i < currCharSize / 2; i++) {
               const char slice = reinterpret_cast<char*>(&currChar)[i];
               if (slice != 0) {
                  out << slice;
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

class TablePrinter : public lingodb::execution::ResultProcessor {
   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(0);
      if (!resultTable) return;
      auto table = resultTable.value()->get();
      printTable(table);
   }
};
class BatchedTablePrinter : public lingodb::execution::ResultProcessor {
   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      for (size_t i = 0;; i++) {
         auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(i);
         if (!resultTable) return;
         auto table = resultTable.value()->get();
         printTable(table);
      }
   }
};

// Safely escape CSV fields that contain delimiters, quotes, or newlines
std::string escapeCSV(const std::string& val, const lingodb::execution::CSVConfig& config) {
   bool needs_quote = val.find(config.delimiter) != std::string::npos ||
      val.find(config.quote) != std::string::npos ||
      val.find('\n') != std::string::npos ||
      val.find('\r') != std::string::npos;
   if (!needs_quote) return val;

   std::string escaped = std::string(1, config.quote);
   for (char c : val) {
      if (c == config.quote) {
         escaped += config.escape;
      }
      escaped += c;
   }
   escaped += config.quote;
   return escaped;
}

// Core parsing logic adapted from LingoDB to output CSV rows
void writeTableToCSV(const std::shared_ptr<arrow::Table>& table, std::ostream& os, const lingodb::execution::CSVConfig& config, bool is_first_batch) {
   if (table->columns().empty()) {
      if (is_first_batch) os << "Statement executed successfully." << std::endl;
      return;
   }

   std::vector<std::string> columnReps;
   std::vector<size_t> positions;
   std::vector<bool> convertHex;
   arrow::PrettyPrintOptions options;
   options.indent_size = 0;
   options.window = 100;
   options.element_size_limit = 10000;

   // Print Header
   if (config.includeHeader && is_first_batch) {
      for (int i = 0; i < table->num_columns(); i++) {
         if (i > 0) os << config.delimiter;
         os << escapeCSV(table->schema()->field(i)->name(), config);
      }
      os << config.eol;
   }

   for (auto c : table->columns()) {
      auto typeId = table->schema()->field(positions.size())->type()->id();
      convertHex.push_back(typeId == arrow::Type::FIXED_SIZE_BINARY);
      std::string str;
      if (typeId == arrow::Type::DOUBLE || typeId == arrow::Type::FLOAT) {
         // Reformat the numeric token on each value line: clean decimals stay clean
         // (0.0, 0.5, 32.5) while genuinely tiny/huge values use scientific notation
         // (std::to_chars shortest round-trippable form, %g-like). Non-finite values are
         // emitted as "infinity"/"-infinity"/"nan" so they match the Graphalytics reference.
         // We reuse PrettyPrint's exact line framing (brackets, commas, indentation and
         // any windowing/truncation) and only rewrite the numeric token on each value
         // line, so the per-column line parser below stays perfectly aligned with the
         // other columns of the same table.
         std::string raw;
         (void) arrow::PrettyPrint(*c.get(), options, &raw);
         std::ostringstream ss;
         std::istringstream in(raw);
         std::string line;
         bool firstLine = true;
         while (std::getline(in, line)) {
            if (!firstLine) ss << "\n";
            firstLine = false;
            // Candidate token start: a digit, sign, dot, or the lead char of inf/nan.
            size_t start = line.find_first_of("-+.0123456789iInN");
            if (start == std::string::npos) {
               ss << line; // bracket / ellipsis / "null" line: keep verbatim
               continue;
            }
            const char* cstart = line.c_str() + start;
            char* endptr = nullptr;
            double v = std::strtod(cstart, &endptr);
            if (endptr == cstart) {
               ss << line; // not a parseable number (e.g. "null"): keep verbatim
               continue;
            }
            size_t end = start + static_cast<size_t>(endptr - cstart);
            std::string tok;
            if (std::isinf(v)) {
               tok = std::signbit(v) ? "-infinity" : "infinity";
            } else if (std::isnan(v)) {
               tok = "nan";
            } else {
               // Shortest round-trippable form decides clean-vs-scientific by *cleanliness*,
               // not magnitude: short exact decimals (SSSP weights like 0.5, 32.5) print
               // fixed, while messy/computed values (PageRank like 0.6142957099999999) and
               // tiny/huge values print in scientific notation with 15 digits to match the
               // Graphalytics reference.
               char buf[64];
               auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
               std::string shortest(buf, ptr);
               bool sciAlready = shortest.find_first_of("eE") != std::string::npos;
               size_t sigDigits = 0;
               if (!sciAlready) {
                  std::string digits;
                  for (char ch : shortest)
                     if (ch >= '0' && ch <= '9') digits += ch;
                  size_t b = digits.find_first_not_of('0');
                  size_t e = digits.find_last_not_of('0');
                  if (b != std::string::npos) sigDigits = e - b + 1;
               }
               if (sciAlready || sigDigits > 3) {
                  std::ostringstream num;
                  num << std::scientific << std::setprecision(15) << v;
                  tok = num.str();
               } else {
                  tok = shortest;
                  if (tok.find('.') == std::string::npos) {
                     tok += ".0"; // integer-looking -> keep a decimal point (0 -> 0.0, 2 -> 2.0)
                  }
               }
            }
            ss << line.substr(0, start) << tok << line.substr(end);
         }
         str = ss.str();
      } else {
         (void) arrow::PrettyPrint(*c.get(), options, &str);
      }
      columnReps.push_back(str);
      positions.push_back(0);
   }

   bool cont = true;
   while (cont) {
      cont = false;
      bool skipNL = false;
      std::vector<std::string> rowValues;

      for (size_t column = 0; column < columnReps.size(); column++) {
         char32_t currChar = U'\0';
         uint8_t currCharSize = 0;
         bool first = true;
         std::stringstream out;

         while (positions[column] < columnReps[column].size()) {
            cont = true;
            char curr = columnReps[column][positions[column]];
            char next = (positions[column] + 1 < columnReps[column].size()) ? columnReps[column][positions[column] + 1] : '\0';
            positions[column]++;

            if (first && (curr == '[' || curr == ']' || curr == ',')) continue;
            if (curr == ',' && next == '\n') continue;
            if (curr == '\n') break;

            first = false;
            if (convertHex[column] && isxdigit(curr)) {
               if (currCharSize % 2 == 0)
                  currChar |= hexval(curr) << (currCharSize++ * 4 + 4);
               else
                  currChar |= hexval(curr) << (currCharSize++ * 4 - 4);
            } else if ((curr & (1 << 7)) == (1 << 7)) {
               const auto extendedCurr = static_cast<char32_t>(curr) & 0xFF;
               currChar |= static_cast<char32_t>(extendedCurr << (currCharSize * 4));
               currCharSize += 2;
            } else {
               if (currChar != U'\0') {
                  for (size_t i = 0; i < currCharSize / 2; i++) {
                     const char slice = reinterpret_cast<char*>(&currChar)[i];
                     if (slice != 0) out << slice;
                  }
                  currChar = U'\0';
                  currCharSize = 0;
               }
               out << curr;
            }
         }

         if (currChar != U'\0') {
            for (size_t i = 0; i < currCharSize / 2; i++) {
               const char slice = reinterpret_cast<char*>(&currChar)[i];
               if (slice != 0) out << slice;
            }
         }

         if (first) {
            skipNL = true; // Indicates no more elements in this column
         } else {
            rowValues.push_back(out.str());
         }
      }

      if (!skipNL) {
         for (size_t i = 0; i < rowValues.size(); i++) {
            if (i > 0) os << config.delimiter;
            os << escapeCSV(rowValues[i], config);
         }
         os << config.eol;
      }
   }
}

class CSVTablePrinter : public lingodb::execution::ResultProcessor {
   lingodb::execution::CSVConfig config_;

   public:
   explicit CSVTablePrinter(lingodb::execution::CSVConfig config) : config_(std::move(config)) {}

   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(0);
      if (!resultTable) return;

      std::ofstream fileStream;
      std::ostream* os = &std::cout;

      if (!config_.filePath.empty()) {
         fileStream.open(config_.filePath, std::ios::out | std::ios::trunc);
         if (fileStream.is_open()) {
            os = &fileStream;
         } else {
            std::cerr << "Failed to open file: " << config_.filePath << " - falling back to stdout.\n";
         }
      }

      writeTableToCSV(resultTable.value()->get(), *os, config_, true);

      if (fileStream.is_open()) {
         std::cout << "Result successfully written to: " << config_.filePath << std::endl;
      }
   }
};

class BatchedCSVTablePrinter : public lingodb::execution::ResultProcessor {
   lingodb::execution::CSVConfig config_;

   public:
   explicit BatchedCSVTablePrinter(lingodb::execution::CSVConfig config) : config_(std::move(config)) {}

   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      std::ofstream fileStream;
      std::ostream* os = &std::cout;

      if (!config_.filePath.empty()) {
         fileStream.open(config_.filePath, std::ios::out | std::ios::trunc);
         if (fileStream.is_open()) {
            os = &fileStream;
         } else {
            std::cerr << "Failed to open file: " << config_.filePath << " - falling back to stdout.\n";
         }
      }

      bool wrote_data = false;
      for (size_t i = 0;; i++) {
         auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(i);
         if (!resultTable) break;

         writeTableToCSV(resultTable.value()->get(), *os, config_, (i == 0));
         wrote_data = true;
      }

      if (fileStream.is_open() && wrote_data) {
         std::cout << "Result successfully written to: " << config_.filePath << std::endl;
      }
   }
};

} // namespace

std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createTableRetriever(std::shared_ptr<arrow::Table>& result) {
   return std::make_unique<TableRetriever>(result);
}

std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createTablePrinter() {
   return std::make_unique<TablePrinter>();
}

std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createBatchedTablePrinter() {
   return std::make_unique<BatchedTablePrinter>();
}
std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createTableCSVPrinter(CSVConfig config) {
   return std::make_unique<CSVTablePrinter>(std::move(config));
}

std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createBatchedTableCSVPrinter(CSVConfig config) {
   return std::make_unique<BatchedCSVTablePrinter>(std::move(config));
}
