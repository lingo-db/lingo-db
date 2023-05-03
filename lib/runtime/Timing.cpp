#include "runtime/Timing.h"
#include <chrono>
#include <iostream>
namespace {
std::chrono::steady_clock::time_point initial = std::chrono::steady_clock::now();
} // end namespace

uint64_t runtime::Timing::start() {
   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
   return std::chrono::duration_cast<std::chrono::microseconds>(begin - initial).count();
}
void runtime::Timing::stop(uint64_t start) {
   uint64_t end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - initial).count();
   std::cout << "Timing: " << (end - start) / 1000.0 << " ms" << std::endl;
}