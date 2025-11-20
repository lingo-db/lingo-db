#include <fstream>
#include <iostream>
#include <string>
#include <format>
#include <vector>
#include "bh_platform.h"
#include "bh_read_file.h"
#include "wasm_c_api_internal.h"
#include "wasm_export.h"

#ifndef CPYTHON_WASM_SOURCE_DIR
#error CPYTHON_WASM_SOURCE_DIR not defined
#endif

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
   char errorBuf[128] = { 0 };
   uint32 size, stackSize = 16777216, heapSize = 8092;

   /* initialize the wasm runtime by default configurations */
   wasm_runtime_init();

   /* read WASM file into a memory buffer */
   uint8_t * buffer = reinterpret_cast<uint8_t*>(bh_read_file_to_buffer(CPYTHON_WASM_AOT_FILE, &size));
   if (!buffer) {
      std::cerr << "Failed to read WASM file" << std::endl;
      return -1;
   }

   /* parse the WASM file from buffer and create a WASM module */
   wasm_module_t module = wasm_runtime_load(buffer, size, errorBuf, sizeof(errorBuf));
   if (!module) {
      std::cerr << errorBuf << std::endl;
      return -1;
   }
   /* --- 1. Setup 'dir_list' (for --dir=.) --- */
   // This sets the CWD for python.aot (This is correct)
   std::array dirList{
      CPYTHON_WASM_BUILD_DIR
  };


   /* --- 2. Setup 'map_dir_list' (for --map-dir=...) --- */
   // !! THIS IS THE CRITICAL CHANGE !!
   std::array mapDirList{
      // Entry 1: Map the source root
      "/::" CPYTHON_WASM_SOURCE_DIR,

      // Entry 2: Map the python libs (/lib/python3.14)
      "/lib/python3.14::" CPYTHON_WASM_SOURCE_DIR "/Lib",

      // TODO: do we need this?
      // "/app::/home/bachmaier/projects/lingo-db/ptest"
   };

   // spoof fake program name (since CPython uses it to load relativ libs)
   std::string programName = "./python.aot";
   std::array wasmArgs{ programName.data() };

   wasm_runtime_set_wasi_args(module,
                              dirList.data(), dirList.size(),
                              mapDirList.data(), mapDirList.size(),
                              nullptr, 0,
                              wasmArgs.data(), wasmArgs.size());

   wasm_module_inst_t moduleInst = wasm_runtime_instantiate(module, stackSize, heapSize,
                                          errorBuf, sizeof(errorBuf));
   if (!moduleInst) {
      std::cerr << errorBuf << std::endl;
      wasm_runtime_unload(module);
      return -1;
   }

   wasm_exec_env_t execEnv = wasm_runtime_create_exec_env(moduleInst, stackSize);
   if (!execEnv) {
      std::cerr << "Create exec env failed" << std::endl;
      return -1;
   }
   using PyObjectPtr = uint32_t;
   call_py_func<void>(execEnv, moduleInst, "Py_Initialize");
   const auto init = call_py_func<bool>(execEnv, moduleInst, "Py_IsInitialized").at(0).of.i32;
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
      void* nativeBufAddr = nullptr;
      uint64_t instBufAddr = wasm_runtime_module_malloc(moduleInst, strlen(script) + 1, &nativeBufAddr);
      if (!nativeBufAddr) {
         throw std::runtime_error{"Failed to malloc wasm buffer for script"};
      }
      char* nativeCharBuf = std::bit_cast<char*>(nativeBufAddr);

      // Copy script into wasm buffer
      memcpy(nativeCharBuf, script, strlen(script) + 1);

      // Call PyRun_SimpleString(script)
      auto result = call_py_func<int>(execEnv, moduleInst, "PyRun_SimpleString", (uint32_t)instBufAddr).at(0).of.i32;

      // Free wasm buffer
      wasm_runtime_module_free(moduleInst, instBufAddr);

      if (result != 0) {
         call_py_func<void>(execEnv, moduleInst, "PyErr_Print");
         throw std::runtime_error{"Failed to run sys.path.append script"};
      }
      std::cerr << "Successfully added /app to sys.path" << std::endl;
   }

   {
      std::string funcName = "os";
      void* native_buf_addr = nullptr;
      uint64_t inst_buf_addr = wasm_runtime_module_malloc(moduleInst, 200, &native_buf_addr);
      char* native_char_buf = std::bit_cast<char*>(native_buf_addr);


      memcpy(native_char_buf, funcName.c_str(), funcName.size() + 1);
      uint32_t fNameWasmStr = static_cast<uint32_t>(inst_buf_addr);
      uint32_t arg1 = 10;
      uint32_t arg2 = 20;


      // pName = PyUnicode_DecodeFSDefault(argv[1]);
      auto pName = call_py_func<PyObjectPtr>(execEnv, moduleInst, "PyUnicode_DecodeFSDefault", fNameWasmStr).at(0).of.i32;
      //pModule = PyImport_Import(pName);
      auto pModule = call_py_func<PyObjectPtr>(execEnv, moduleInst, "PyImport_Import", pName).at(0).of.i32;
      if (!pModule) {
         call_py_func<void>(execEnv, moduleInst, "PyErr_Print");
         throw std::runtime_error{"Module not found"};
      }

      // auto pFunc = call_py_func<PyObjectPtr>(execEnv, moduleInst, "PyObject_GetAttrString", pModule,fNameWasmStr).at(0).of.i32;
      // //Check callable
      // std::cerr << "pFunc: " << pFunc << std::endl;
      // if (!pFunc || call_py_func<bool>(execEnv, moduleInst, "PyCallable_Check", pFunc).at(0).of.i32 == 0) {
      //    throw std::runtime_error{"Function is not callable"};
      // }
      // auto pArgs = call_py_func<PyObjectPtr>(execEnv, moduleInst, "PyTuple_New", 2).at(0).of.i32;
      // if (!pArgs) {
      //    throw std::runtime_error{"Failed to create args tuple"};
      // }
      // auto pArg1 = call_py_func<PyObjectPtr>(execEnv, moduleInst, "PyLong_FromLong", arg1).at(0).of.i32;
      // auto pArg2 = call_py_func<PyObjectPtr>(execEnv, moduleInst, "PyLong_FromLong", arg2).at(0).of.i32;
      // if (!pArg1 || !pArg2) {
      //    throw std::runtime_error{"Failed to create args"};
      // }
      //
      // call_py_func<int>(execEnv, moduleInst, "PyTuple_SetItem", pArgs, 0, pArg1);
      // call_py_func<int>(execEnv, moduleInst, "PyTuple_SetItem", pArgs, 1, pArg2);
      //
      // PyObjectPtr resultObj = call_py_func<PyObjectPtr>(execEnv, moduleInst, "PyObject_CallObject", pFunc, pArgs).at(0).of.i32;
      //
      // if (!resultObj) {
      //    throw std::runtime_error{"Failed to call function"};
      // }
      // auto result = call_py_func<int64_t>(execEnv, moduleInst, "PyLong_AsLong", resultObj).at(0).of.i64;
      // std::cerr << "Result: " << result << std::endl;




   }


   // free all
   wasm_runtime_destroy_exec_env(execEnv);
   wasm_runtime_deinstantiate(moduleInst);
   wasm_runtime_unload(module);
   wasm_runtime_destroy();
}
