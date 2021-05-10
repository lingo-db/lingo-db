#include "runtime/execution_context.h"
extern "C" {
int bitcode_test(int a, int b) {
   return a + b;
}
int precompiled_execution_context_test(char* ptr){
  return ((runtime::ExecutionContext*)ptr)->id;
}
}