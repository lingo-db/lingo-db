#ifndef EXECUTION_TIMING_H
#define EXECUTION_TIMING_H
#include "Error.h"

#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>
namespace execution {
class TimingProcessor {
   public:
   virtual void addTiming(const std::unordered_map<std::string, double>& timing) = 0;
   virtual void process() = 0;
   virtual ~TimingProcessor() {}
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
      for (auto [name, t] : timing) {
         total += t;
      }
      timing["total"] = total;
      std::vector<std::string> printOrder = {"QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime", "total"};
      std::cout << std::endl
                << std::endl;
      std::cout << std::setw(10) << "name";
      for (auto n : printOrder) {
         std::cout << std::setw(15) << n;
      }
      std::cout << std::endl;
      std::cout << std::setw(10) << queryName;
      for (auto n : printOrder) {
         if (timing.contains(n)) {
            std::cout << std::setw(15) << timing[n];
         } else {
            std::cout << std::setw(15) << "";
         }
      }
   }
};
} // namespace execution
#endif //EXECUTION_TIMING_H
