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

   wasm_exec_env_t exec_env = wasm_runtime_create_exec_env(module_inst, stack_size);
   if (!exec_env) {
      std::cerr << "Create exec env failed" << std::endl;
      return -1;
   }

   // test if interpreter is initialized
   auto results = call_py_func<char*>(exec_env, module_inst, "Py_GetVersion");
   std::string python_version{std::bit_cast<char*>(wasm_runtime_addr_app_to_native(module_inst, results[0].of.i32))};
   std::cout << std::format("Python version: {}\n", python_version);

   // hello world
   using PyObjectPtr = uint32_t;
   void* native_buf_addr = nullptr;
   uint64_t inst_buf_addr = wasm_runtime_module_malloc(module_inst, 200, &native_buf_addr);

   std::string part1{"Hello "};
   std::string part2{"World!"};

   char* native_char_buf = std::bit_cast<char*>(native_buf_addr);
   memcpy(native_char_buf, part1.c_str(), part1.size() + 1);
   uint32_t wasm_part1_str = static_cast<uint32_t>(inst_buf_addr);
   memcpy(native_char_buf + part1.size() + 1, part2.c_str(), part2.size() + 1);
   uint32_t wasm_part2_str = static_cast<uint32_t>(inst_buf_addr + part1.size() + 1);

   PyObjectPtr part1_obj = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyUnicode_FromString", wasm_part1_str)[0].of.i32;
   PyObjectPtr part2_obj = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyUnicode_FromString", wasm_part2_str)[0].of.i32;
   PyObjectPtr joined_obj = call_py_func<PyObjectPtr>(exec_env, module_inst, "PyUnicode_Concat", part1_obj, part2_obj)[0].of.i32;
   call_py_func<void>(exec_env, module_inst, "Py_DecRef", part1_obj);
   call_py_func<void>(exec_env, module_inst, "Py_DecRef", part2_obj);
   uint32_t wasm_result_str = call_py_func<char*>(exec_env, module_inst, "PyUnicode_AsUTF8", joined_obj)[0].of.i32;
   std::string result_str{std::bit_cast<char*>(wasm_runtime_addr_app_to_native(module_inst, wasm_result_str))};
   std::cout << std::format("Result string: {}\n", result_str);
   call_py_func<void>(exec_env, module_inst, "Py_DecRef", joined_obj);

   // free all
   wasm_runtime_destroy_exec_env(exec_env);
   wasm_runtime_deinstantiate(module_inst);
   wasm_runtime_unload(module);
   wasm_runtime_destroy();
}
