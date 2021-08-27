#include "runtime/helpers.h"
uint8_t* execution_context;
extern "C"  __attribute__((always_inline)) void _mlir_ciface_set_execution_context(uint8_t* ctxt) {
   execution_context=ctxt;
}
extern "C"  __attribute__((always_inline)) uint8_t* _mlir_ciface_get_execution_context() {
   return execution_context;
}