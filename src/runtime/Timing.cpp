#include "lingodb/runtime/Timing.h"
#include "lingodb/utility/PerfEvent.h"
#include <chrono>
#include <iostream>
namespace {
std::chrono::steady_clock::time_point initial = std::chrono::steady_clock::now();
PerfEvent* currentEvent = nullptr;

} // end namespace

uint64_t lingodb::runtime::Timing::start() {
   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
   return std::chrono::duration_cast<std::chrono::microseconds>(begin - initial).count();
}
void lingodb::runtime::Timing::stop(uint64_t start) {
   uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - initial).count();
   std::cout << "Timing: " << (end - start) / 1000.0 << " ms" << std::endl;
}
void lingodb::runtime::Timing::startPerf() {
   std::cout << "start perf" << std::endl;
   currentEvent = new PerfEvent();
   currentEvent->startCounters();
}
void lingodb::runtime::Timing::stopPerf() {
   std::cout << "stop perf" << std::endl;
   currentEvent->stopCounters();
   currentEvent->printReport(std::cout, 1);
   delete currentEvent;
   currentEvent = nullptr;
}