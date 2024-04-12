#ifndef EXECUTION_LLVMBACKENDS_H
#define EXECUTION_LLVMBACKENDS_H
#include "Backend.h"
namespace execution{
   std::unique_ptr<ExecutionBackend> createDefaultLLVMBackend();
   std::unique_ptr<ExecutionBackend> createLLVMDebugBackend();
   std::unique_ptr<ExecutionBackend> createLLVMProfilingBackend();
   std::unique_ptr<ExecutionBackend> createGPULLVMBackend();
} // namespace execution
#endif //EXECUTION_LLVMBACKENDS_H
