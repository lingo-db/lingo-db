#ifndef LINGODB_EXECUTION_TIMING_H
#define LINGODB_EXECUTION_TIMING_H

#include "Error.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace lingodb::execution {
class TimingProcessor {
   public:
   virtual void addTiming(const std::unordered_map<std::string, double>& timing) = 0;

   virtual void process() = 0;

   virtual ~TimingProcessor() {
   }
};

class TimingPrinter : public TimingProcessor {
   std::unordered_map<std::string, double> timing;
   std::string queryName;

   public:
   TimingPrinter(std::string queryFile) {
      if (queryFile.find('/') != std::string::npos) {
         queryName = queryFile.substr(queryFile.find_last_of("/\\") + 1);
      } else {
         queryName = queryFile;
      }
   }

   void addTiming(const std::unordered_map<std::string, double>& timing) override {
      this->timing.insert(timing.begin(), timing.end());
   }

   void process() override {
      double total = 0.0;
      for (const auto t : timing | std::ranges::views::values) {
         total += t;
      }
      timing["total"] = total;
      std::vector<std::string> printOrder = {
          "QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerArrow", "lowerToLLVM", "baselineLowering",
         "toLLVMIR",
         "llvmOptimize", "llvmCodeGen", "baselineCodeGen", "baselineEmit", "executionTime", "total"};
      const unsigned headerLen = std::ranges::max_element(
                                    printOrder.begin(), printOrder.end(),
                                    [](const std::string& a, const std::string& b) {
                                       return a.length() < b.length();
                                    })
                                    ->length() +
         3;
      const unsigned queryNameLen = queryName.length();
      std::cout << std::endl
                << std::endl;
      std::cout << std::setw(queryNameLen) << "name";
      for (auto n : printOrder) {
         std::cout << std::setw(headerLen) << n;
      }
      std::cout << std::endl;
      std::cout << std::setw(queryNameLen) << queryName;
      for (auto n : printOrder) {
         if (timing.contains(n)) {
            std::cout << std::setw(headerLen) << timing[n];
         } else {
            std::cout << std::setw(headerLen) << "";
         }
      }
   }
};
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_TIMING_H
