#include <fstream>
#include <iostream>
#include <string>
#include <format>
#include <vector>
#include "bh_platform.h"
#include "bh_read_file.h"
#include "wasm_c_api_internal.h"
#include "wasm_export.h"

wasm_function_inst_t get_py_func(wasm_module_inst_t module_inst, const char* func_name) {
   wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, func_name);
   if (!func)
      throw std::runtime_error{std::format("Function '{}' not found in module", func_name)};
   return func;
}

template<typename T>
void pack_val(void* args_void, uint32_t& idx, T v)
requires (std::integral<T> && sizeof(T) == 4) {
    static_cast<wasm_val_t*>(args_void)[idx++].of.i32 = static_cast<int32_t>(v);
}
template<typename T>
void pack_val(void* args_void, uint32_t& idx, T v)
requires (std::integral<T> && sizeof(T) == 8) {
    static_cast<wasm_val_t*>(args_void)[idx++].of.i64 = static_cast<int64_t>(v);
}
inline void pack_val(void* args_void, uint32_t& idx, float v) {
    static_cast<wasm_val_t*>(args_void)[idx++].of.f32 = v;
}
inline void pack_val(void* args_void, uint32_t& idx, double v) {
    static_cast<wasm_val_t*>(args_void)[idx++].of.f64 = v;
}

// Fallback to produce a clear compile-time error for unsupported types.
template<typename T>
void pack_val(void* /*args*/, uint32_t& /*idx*/, T /*v*/) {
   static_assert(sizeof(T) == 0, "pack_val: unsupported argument type");
}

template<typename T, typename... Rest>
int count_results() {
   int count = 0;
   if constexpr (!std::is_same_v<T, void>) count = 1;
   if constexpr (sizeof...(Rest) == 0) return count;
   else return count + count_results<Rest...>();
}

template<typename... Outs, typename... Ins>
void serialize_args(wasm_val_t* args, uint32_t& num_args, uint32_t& num_results, Ins&&... ins) {
    num_args = 0;
    num_results = count_results<Outs...>();
    // Use fold expression to pack each input in order
    (pack_val(static_cast<void*>(args), num_args, std::forward<Ins>(ins)), ...);
}

// - Outs... must be supplied explicitly
// - Ins... are deduced from the provided inputs.
template<typename... Outs, typename... Ins>
std::vector<wasm_val_t> call_py_func(wasm_exec_env_t exec_env,
                  wasm_module_inst_t module_inst,
                  std::string_view func_name, Ins&&... ins) {
   wasm_function_inst_t func = get_py_func(module_inst, func_name.data());
   std::vector<wasm_val_t> args(sizeof...(Ins));
   uint32_t num_args = 0, num_results = 0;
   serialize_args<Outs...>(args.data(), num_args, num_results, ins...);
   std::vector<wasm_val_t> results(num_results);
   assert(num_args == sizeof...(Ins));
   bool success = wasm_runtime_call_wasm_a(exec_env, func, num_results, results.data(), num_args, args.data());
   if (!success) {
      /* exception is thrown if call fails */
      throw std::runtime_error{wasm_runtime_get_exception(module_inst)};
   }
   return results;
}

int main(int argc, char** argv) {
   char error_buf[128] = { 0 };
   uint32 size, stack_size = 16777216, heap_size = 8092;

   /* initialize the wasm runtime by default configurations */
   wasm_runtime_init();

   /* read WASM file into a memory buffer */
   uint8_t * buffer = reinterpret_cast<uint8_t*>(bh_read_file_to_buffer("/home/bachmaier/projects/lingo-db/build/cpython-wasm/Python-3.14.0/cross-build/wasm32-wasip1/python.aot", &size));
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
   /* --- 1. Setup 'dir_list' (for --dir=.) --- */
   // This sets the CWD for python.aot (This is correct)
   const char* dir_list[] = {
      "/home/bachmaier/projects/lingo-db/build/cpython-wasm/Python-3.14.0/cross-build/wasm32-wasip1"
  };
   uint32_t dir_count = 1;


   /* --- 2. Setup 'map_dir_list' (for --map-dir=...) --- */
   // !! THIS IS THE CRITICAL CHANGE !!
   const char* map_dir_list[] = {
      // Entry 1: Map the whole Python-3.14.0 dir to /
      "/::/home/bachmaier/projects/lingo-db/build/cpython-wasm/Python-3.14.0/",

      // Entry 2: Map the path Python EXPECTS (/lib/python3.14)
      "/lib/python3.14::/home/bachmaier/projects/lingo-db/build/cpython-wasm/Python-3.14.0/Lib",

      // --- FIX 1: ADD THIS LINE ---
      // Map your 'ptest' dir (containing multiply.py) to '/app' in the sandbox
      "/app::/home/bachmaier/projects/lingo-db/ptest"
   };
   // Update the count to 2
   uint32_t map_dir_count = 3;


   /* --- 3. Setup 'env_list' (for --env=...) --- */
   // We must update PYTHONPATH to match the new mapping
   const char* env_list[] = {
      "PYTHONHOME=/app"

  };
   uint32_t env_count = 1;


   /* --- 4. Setup 'argv' (the command argument "./python.aot") --- */
   // This is correct: relative to the CWD set in dir_list
   char arg0[] = "./python.aot";
   char* argv2[] = { arg0 };
   int argc2 = 1;


   /* --- 5. The Final Function Call --- */
   wasm_runtime_set_wasi_args(module,
                              dir_list, dir_count,
                              map_dir_list, map_dir_count,
                              env_list, env_count,
                              argv2, argc2);

   /* create an instance of the WASM module (WASM linear memory is ready) */
   wasm_module_inst_t module_inst = wasm_runtime_instantiate(module, stack_size, heap_size,
                                          error_buf, sizeof(error_buf));
   if (!module_inst) {
      std::cerr << error_buf << std::endl;
      wasm_runtime_unload(module);
      return -1;
   }

   wasm_exec_env_t exec_env = wasm_runtime_create_exec_env(module_inst, stack_size);
   if (!exec_env) {
      std::cerr << "Create exec env failed" << std::endl;
      return -1;
   }
   using PyObjectPtr = uint32_t;
   call_py_func<void>(exec_env, module_inst, "Py_Initialize");
   auto init = call_py_func<bool>(exec_env, module_inst, "Py_IsInitialized").at(0).of.i32;
   if (!init) {
      throw std::runtime_error{std::format("Py_Initialize failed")};
   }

   {

      //Init Buffer

   }
   //Add module path
   {
      const char* script = "import sys; sys.path.append('/app')";

      // Malloc buffer in wasm for the script
      void* native_buf_addr = nullptr;
      uint64_t inst_buf_addr = wasm_runtime_module_malloc(module_inst, strlen(script) + 1, &native_buf_addr);
      if (!native_buf_addr) {
         throw std::runtime_error{"Failed to malloc wasm buffer for script"};
      }
      char* native_char_buf = std::bit_cast<char*>(native_buf_addr);

      // Copy script into wasm buffer
      memcpy(native_char_buf, script, strlen(script) + 1);

      // Call PyRun_SimpleString(script)
      auto result = call_py_func<int>(exec_env, module_inst, "PyRun_SimpleString", (uint32_t)inst_buf_addr).at(0).of.i32;

      // Free wasm buffer
      wasm_runtime_module_free(module_inst, inst_buf_addr);

      if (result != 0) {
         call_py_func<void>(exec_env, module_inst, "PyErr_Print");
         throw std::runtime_error{"Failed to run sys.path.append script"};
      }
      std::cerr << "Successfully added /app to sys.path" << std::endl;
   }






   /*
    *
    * https://docs.python.org/3/extending/embedding.html
    */

   {
      std::string funcName = "multiply";
      void* native_buf_addr = nullptr;
      uint64_t inst_buf_addr = wasm_runtime_module_malloc(module_inst, 200, &native_buf_addr);
      char* native_char_buf = std::bit_cast<char*>(native_buf_addr);


      memcpy(native_char_buf, funcName.c_str(), funcName.size() + 1);
      uint32_t fNameWasmStr = static_cast<uint32_t>(inst_buf_addr);
      uint32_t arg1 = 10;
      uint32_t arg2 = 20;


      // pName = PyUnicode_DecodeFSDefault(argv[1]);
      auto pName = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyUnicode_DecodeFSDefault", fNameWasmStr).at(0).of.i32;
      //pModule = PyImport_Import(pName);
      auto pModule = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyImport_Import", pName).at(0).of.i32;
      std::cerr << "pModule: " << pModule <<  " pName:" <<   pName << std::endl;
      if (!pModule) {
         call_py_func<void>(exec_env, module_inst, "PyErr_Print");
         throw std::runtime_error{"Module not found"};
      }
      memcpy(native_char_buf, funcName.c_str(), funcName.size() + 1);
       fNameWasmStr = static_cast<uint32_t>(inst_buf_addr);
      auto pFunc = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyObject_GetAttrString", pModule,fNameWasmStr).at(0).of.i32;
      //Check callable
      std::cerr << "pFunc: " << pFunc << std::endl;
      if (!pFunc || call_py_func<bool>(exec_env, module_inst, "PyCallable_Check", pFunc).at(0).of.i32 == 0) {
         throw std::runtime_error{"Function is not callable"};
      }
      auto pArgs = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyTuple_New", 2).at(0).of.i32;
      if (!pArgs) {
         throw std::runtime_error{"Failed to create args tuple"};
      }
      auto pArg1 = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyLong_FromLong", arg1).at(0).of.i32;
      auto pArg2 = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyLong_FromLong", arg2).at(0).of.i32;
      if (!pArg1 || !pArg2) {
         throw std::runtime_error{"Failed to create args"};
      }

      call_py_func<int>(exec_env, module_inst, "PyTuple_SetItem", pArgs, 0, pArg1);
      call_py_func<int>(exec_env, module_inst, "PyTuple_SetItem", pArgs, 1, pArg2);

      PyObjectPtr resultObj = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyObject_CallObject", pFunc, pArgs).at(0).of.i32;

      if (!resultObj) {
         throw std::runtime_error{"Failed to call function"};
      }
      auto result = call_py_func<int64_t>(exec_env, module_inst, "PyLong_AsLong", resultObj).at(0).of.i64;
      std::cerr << "Result: " << result << std::endl;




   }


   // free all
   wasm_runtime_destroy_exec_env(exec_env);
   wasm_runtime_deinstantiate(module_inst);
   wasm_runtime_unload(module);
   wasm_runtime_destroy();
}
