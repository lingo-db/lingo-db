#include "runtime/helpers.h"
#include <chrono>
#include <iostream>
extern "C" std::chrono::high_resolution_clock::time_point* _mlir_ciface_start_execution(){// NOLINT (clang-diagnostic-return-type-c-linkage)
   auto *timepoint=new std::chrono::high_resolution_clock::time_point(std::chrono::high_resolution_clock::now());
   return timepoint;
}
extern "C" void _mlir_ciface_finish_execution(std::chrono::high_resolution_clock::time_point* started){// NOLINT (clang-diagnostic-return-type-c-linkage)
   auto now=std::chrono::high_resolution_clock::now();
   std::cout<<"runtime: "<<std::chrono::duration_cast<std::chrono::milliseconds >(now-*started).count()<< " ms"<<std::endl;
   delete started;
}