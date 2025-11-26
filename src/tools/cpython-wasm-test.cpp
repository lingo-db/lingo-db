#include "bh_platform.h"
#include "bh_read_file.h"
#include "wasm_c_api_internal.h"
#include "wasm_export.h"
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <memory>

#ifndef CPYTHON_WASM_SOURCE_DIR
#error CPYTHON_WASM_SOURCE_DIR not defined
#endif

class PyObjectRef;

wasm_function_inst_t get_py_func(wasm_module_inst_t module_inst, const char* func_name) {
   wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, func_name);
   if (!func)
      throw std::runtime_error{std::format("Function '{}' not found in module", func_name)};
   return func;
}

template <typename T>
void pack_val(void* args_void, uint32_t& idx, T v)
   requires(std::is_same_v<T, PyObjectRef>)
{
   static_cast<wasm_val_t*>(args_void)[idx++].of.i32 = static_cast<int32_t>(v);
}
template <typename T>
void pack_val(void* args_void, uint32_t& idx, T v)
   requires(std::integral<T> && sizeof(T) == 4)
{
   static_cast<wasm_val_t*>(args_void)[idx++].of.i32 = static_cast<int32_t>(v);
}
template <typename T>
void pack_val(void* args_void, uint32_t& idx, T v)
   requires(std::integral<T> && sizeof(T) == 8)
{
   static_cast<wasm_val_t*>(args_void)[idx++].of.i64 = static_cast<int64_t>(v);
}
inline void pack_val(void* args_void, uint32_t& idx, float v) {
   static_cast<wasm_val_t*>(args_void)[idx++].of.f32 = v;
}
inline void pack_val(void* args_void, uint32_t& idx, double v) {
   static_cast<wasm_val_t*>(args_void)[idx++].of.f64 = v;
}

// Fallback to produce a clear compile-time error for unsupported types.
template <typename T>
void pack_val(void* /*args*/, uint32_t& /*idx*/, T /*v*/) {
   static_assert(sizeof(T) == 0, "pack_val: unsupported argument type");
}

template <typename T, typename... Rest>
int count_results() {
   int count = 0;
   if constexpr (!std::is_same_v<T, void>) count = 1;
   if constexpr (sizeof...(Rest) == 0)
      return count;
   else
      return count + count_results<Rest...>();
}

template <typename... Outs, typename... Ins>
void serialize_args(wasm_val_t* args, uint32_t& num_args, uint32_t& num_results, Ins&&... ins) {
   num_args = 0;
   num_results = count_results<Outs...>();
   // Use fold expression to pack each input in order
   (pack_val(static_cast<void*>(args), num_args, std::forward<Ins>(ins)), ...);
}

class WasmRuntime {
   public:
   WasmRuntime() {
      wasm_runtime_init();
   }
   ~WasmRuntime() {
      wasm_runtime_destroy();
   }
};

class CPythonModule {
   public:
   CPythonModule(std::unique_ptr<WasmRuntime> runtime) : runtime{std::move(runtime)} {
      /* read WASM file into a memory buffer */
      uint32_t bufSize;
      uint8_t* buffer = reinterpret_cast<uint8_t*>(bh_read_file_to_buffer(CPYTHON_WASM_AOT_FILE, &bufSize));
      if (!buffer) {
         throw std::runtime_error{"Failed to read WASM file"};
      }

      /* parse the WASM file from buffer and create a WASM module */
      char errorBuf[128] = {0};
      module = wasm_runtime_load(buffer, bufSize, errorBuf, sizeof(errorBuf));
      if (!module) {
         throw std::runtime_error{std::format("Failed to load WASM module: {}", errorBuf)};
      }
   }

   ~CPythonModule() {
      wasm_runtime_unload(module);
   }

   wasm_module_t module;

   private:
   std::unique_ptr<WasmRuntime> runtime;
};

class CPythonInst {
   public:
   constexpr static uint32 stackSize = 1677721600; // 16MB
   constexpr static uint32 heapSize = 8092; // 8KB

   CPythonInst(std::unique_ptr<CPythonModule> cpythonModule) : cpythonModule{std::move(cpythonModule)} {
      char errorBuf[128] = {0};

      /* --- 1. Setup 'dir_list' (for --dir=.) --- */
      // This sets the CWD for python.aot (This is correct)
      std::array dirList{
         CPYTHON_WASM_BUILD_DIR};

      /* --- 2. Setup 'map_dir_list' (for --map-dir=...) --- */
      std::array mapDirList{
         "/::" CPYTHON_WASM_SOURCE_DIR,
         "/lib/python3.14::" CPYTHON_WASM_SOURCE_DIR "/Lib",
      };

      // spoof fake program name (since CPython uses it to load relativ libs)
      std::string programName = "./python.aot";
      std::array wasmArgs{programName.data()};

      wasm_runtime_set_wasi_args(this->cpythonModule->module,
                                 dirList.data(), dirList.size(),
                                 mapDirList.data(), mapDirList.size(),
                                 nullptr, 0,
                                 wasmArgs.data(), wasmArgs.size());
      moduleInst = wasm_runtime_instantiate(this->cpythonModule->module, stackSize, heapSize,
                                            errorBuf, sizeof(errorBuf));
      if (!moduleInst) {
         throw std::runtime_error{std::format("Failed to instantiate CPython module: {}", errorBuf)};
      }
   }

   ~CPythonInst() {
      wasm_runtime_deinstantiate(moduleInst);
   }

   wasm_module_inst_t moduleInst;

   private:
   std::unique_ptr<CPythonModule> cpythonModule;
};

class PyEnv {
   public:
   // - Outs... must be supplied explicitly
   // - Ins... are deduced from the provided inputs.
   template <typename... Outs, typename... Ins>
   std::vector<wasm_val_t> callPyFunc(const std::string& funcName, Ins&&... ins) {
      wasm_function_inst_t func = get_py_func(moduleInst, funcName.c_str());
      std::vector<wasm_val_t> args(sizeof...(Ins));
      uint32_t nameArgs = 0, numResults = 0;
      serialize_args<Outs...>(args.data(), nameArgs, numResults, ins...);
      std::vector<wasm_val_t> results(numResults);
      assert(nameArgs == sizeof...(Ins));
      bool success = wasm_runtime_call_wasm_a(execEnv, func, numResults, results.data(), nameArgs, args.data());
      if (!success) {
         /* exception is thrown if call fails */
         throw std::runtime_error{wasm_runtime_get_exception(moduleInst)};
      }
      return results;
   }

   PyEnv(std::unique_ptr<CPythonInst> cpythonInst) : moduleInst{cpythonInst->moduleInst}, cpythonInst{std::move(cpythonInst)} {
      execEnv = wasm_runtime_create_exec_env(this->cpythonInst->moduleInst, this->cpythonInst->stackSize);
      if (!execEnv) {
         throw std::runtime_error{"Failed to create execution environment"};
      }
      callPyFunc<void>("Py_Initialize");
   }

   ~PyEnv() {
      wasm_runtime_destroy_exec_env(execEnv);
   }

   void testSetup() {
      const auto init = callPyFunc<bool>("Py_IsInitialized").at(0).of.i32;
      if (!init) {
         throw std::runtime_error{std::format("Py_Initialize failed")};
      }
      std::cout << "Python initialized successfully" << std::endl;
   }

   wasm_exec_env_t execEnv;

   // for conveniance
   wasm_module_inst_t moduleInst;

   private:
   std::unique_ptr<CPythonInst> cpythonInst;
};

class PyObjectRef {
   public:
   PyObjectRef(const int32_t ptr, std::shared_ptr<PyEnv> pyEnv) : ptr{ptr}, pyEnv{pyEnv} {}
   ~PyObjectRef() {
      pyEnv->callPyFunc<void>("Py_DecRef", ptr);
   }

   operator uint32_t() const { return ptr; }

   private:
   int32_t ptr;
   std::shared_ptr<PyEnv> pyEnv;
};

static PyObjectRef importModule(const std::string& moduleName, std::shared_ptr<PyEnv> pyEnv) {
   void* nativeBuf = nullptr;
   uint64_t wasmBuf = wasm_runtime_module_malloc(pyEnv->moduleInst, 200, &nativeBuf);
   auto* nativeCharBuf = std::bit_cast<char*>(nativeBuf);

   memcpy(nativeCharBuf, moduleName.c_str(), moduleName.size() + 1);
   // pName = PyUnicode_DecodeFSDefault(moduleName);
   auto pName = pyEnv->callPyFunc<PyObjectRef>("PyUnicode_DecodeFSDefault", wasmBuf).at(0).of.i32;
   //pModule = PyImport_Import(pName);
   auto pModule = pyEnv->callPyFunc<PyObjectRef>("PyImport_Import", pName).at(0).of.i32;
   if (!pModule) {
      pyEnv->callPyFunc<void>("PyErr_Print");
      throw std::runtime_error{"Module not found"};
   }
   return {pModule, pyEnv};
}

int main(int argc, char** argv) {
   std::unique_ptr<WasmRuntime> runtime = std::make_unique<WasmRuntime>();
   std::unique_ptr<CPythonModule> cpythonModule = std::make_unique<CPythonModule>(std::move(runtime));
   std::unique_ptr<CPythonInst> cpythonInst = std::make_unique<CPythonInst>(std::move(cpythonModule));
   std::shared_ptr<PyEnv> pyEnv = std::make_shared<PyEnv>(std::move(cpythonInst));
   pyEnv->testSetup();

   {
      PyObjectRef osModule = importModule(std::string{"os"}, pyEnv);
      const char* fName = "getcwd";
      // Malloc buffer in wasm for the function
      void* nativeBufAddr = nullptr;
      uint64_t fNameWasmAddr = wasm_runtime_module_malloc(pyEnv->moduleInst, strlen(fName) + 1, &nativeBufAddr);
      if (!nativeBufAddr) {
         throw std::runtime_error{"Failed to malloc wasm buffer for function name"};
      }
      char* nativeCharBuf = std::bit_cast<char*>(nativeBufAddr);
      // Copy function name into wasm buffer
      memcpy(nativeCharBuf, fName, strlen(fName) + 1);

      PyObjectRef pFunc = {pyEnv->callPyFunc<uint32_t>("PyObject_GetAttrString", osModule, fNameWasmAddr).at(0).of.i32, pyEnv};
      //Check callable
      std::cerr << "pFunc: " << pFunc << std::endl;
      if (!pFunc || pyEnv->callPyFunc<bool>("PyCallable_Check", pFunc).at(0).of.i32 == 0) {
         throw std::runtime_error{"Function is not callable"};
      }

      // call function
      PyObjectRef resultObj = {pyEnv->callPyFunc<uint32_t>("PyObject_CallObject", pFunc, 0).at(0).of.i32, pyEnv};
      if (!resultObj) {
         throw std::runtime_error{"Failed to call function"};
      }
      // Convert result to string
      auto resultStrObj = pyEnv->callPyFunc<PyObjectRef>("PyObject_Str", resultObj).at(0).of.i32;
      if (!resultStrObj) {
         throw std::runtime_error{"Failed to convert result to string"};
      }
      // Convert string object to C string
      auto cStrWasmAddr = pyEnv->callPyFunc<uint32_t>("PyUnicode_AsUTF8", resultStrObj).at(0).of.i32;
      void* nativecStr = wasm_runtime_addr_app_to_native(pyEnv->moduleInst, cStrWasmAddr);
      std::cout << std::bit_cast<char*>(nativecStr) << std::endl;
   }
}
