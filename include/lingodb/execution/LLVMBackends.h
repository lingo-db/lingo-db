#ifndef LINGODB_EXECUTION_LLVMBACKENDS_H
#define LINGODB_EXECUTION_LLVMBACKENDS_H
#include "Backend.h"
namespace lingodb::execution{
   std::unique_ptr<ExecutionBackend> createDefaultLLVMBackend();
   std::unique_ptr<ExecutionBackend> createLLVMDebugBackend();
   std::unique_ptr<ExecutionBackend> createLLVMProfilingBackend();
   std::unique_ptr<ExecutionBackend> createGPULLVMBackend();
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_LLVMBACKENDS_H
