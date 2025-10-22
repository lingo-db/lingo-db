#include <fstream>
#include <iostream>
#include <string>
#include "bh_platform.h"
#include "bh_read_file.h"
#include "wasm_c_api_internal.h"
#include "wasm_export.h"

int main(int argc, char** argv) {
   char error_buf[128] = { 0 };
   uint32 size, stack_size = 8092, heap_size = 8092;

   /* initialize the wasm runtime by default configurations */
   wasm_runtime_init();

   /* read WASM file into a memory buffer */
   uint8_t * buffer = reinterpret_cast<uint8_t*>(bh_read_file_to_buffer("/home/doellerer/CLionProjects/lingo-db/build/cpython-wasm/Python-3.14.0/cross-build/wasm32-wasip1/python.aot", &size));
   if (!buffer) {
       std::cerr << "Failed to read WASM file" << std::endl;
       return -1;
   }

   /* parse the WASM file from buffer and create a WASM module */
   wasm_module_t module = wasm_runtime_load(buffer, size, error_buf, sizeof(error_buf));
   if (!module) {
       std::cerr << error_buf << std::endl;
       return -1;
   }

   /* create an instance of the WASM module (WASM linear memory is ready) */
   wasm_module_inst_t module_inst = wasm_runtime_instantiate(module, stack_size, heap_size,
                                          error_buf, sizeof(error_buf));
   if (!module_inst) {
       std::cerr << error_buf << std::endl;
       wasm_runtime_unload(module);
       return -1;
   }

   // test if interpreter is initialized
   wasm_function_inst_t Py_IsInitialized = wasm_runtime_lookup_function(module_inst, "Py_GetVersion");
   if (!Py_IsInitialized) {
       std::cerr << "Function Py_IsInitialized not found" << std::endl;
       return -1;
   }
   wasm_exec_env_t exec_env = wasm_runtime_create_exec_env(module_inst, stack_size);
   if (!exec_env) {
       std::cerr << "Create exec env failed" << std::endl;
       return -1;
   }

   uint32 num_args = 0, num_results = 1;
   wasm_val_t args[0], results[1];

   /* call the WASM function */
   if (wasm_runtime_call_wasm_a(exec_env, Py_IsInitialized, num_results, results, num_args, args)) {
      void* ret = wasm_runtime_addr_app_to_native(module_inst, results[0].of.i32);
      printf("function return: %s\n", ret);
   }
   else {
      /* exception is thrown if call fails */
      printf("ERROR! %s\n", wasm_runtime_get_exception(module_inst));
   }

   // free all
   wasm_runtime_destroy_exec_env(exec_env);
   wasm_runtime_deinstantiate(module_inst);
   wasm_runtime_unload(module);
   wasm_runtime_destroy();
}
