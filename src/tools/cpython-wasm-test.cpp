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
#include <cstring>
#include <array>
#include <cassert>

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
requires (std::is_same_v<T, PyObjectRef>)
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
   constexpr static uint32 heapSize = 8092;    // 8KB

   CPythonInst(std::unique_ptr<CPythonModule> cpythonModule) : cpythonModule{std::move(cpythonModule)} {
      char errorBuf[128] = {0};

      /* --- 1. Setup 'dir_list' (for --dir=.) --- */
      std::array dirList{
         CPYTHON_WASM_BUILD_DIR};

      /* --- 2. Setup 'map_dir_list' (for --map-dir=...) --- */
      std::array mapDirList{
         "/::" CPYTHON_WASM_SOURCE_DIR,
         "/lib/python3.14::" CPYTHON_WASM_SOURCE_DIR "/Lib",
      };

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
      if (!init) throw std::runtime_error{std::format("Py_Initialize failed")};
      std::cout << "Python initialized successfully" << std::endl;
   }

   void runSimpleString(const std::string& script) {
      // We must malloc the script into WASM memory
      void* nativeBuf = nullptr;
      uint64_t wasmBuf = wasm_runtime_module_malloc(moduleInst, script.size() + 1, &nativeBuf);
      if (!nativeBuf) throw std::runtime_error{"Malloc failed for script"};

      memcpy(nativeBuf, script.c_str(), script.size() + 1);

      // Call PyRun_SimpleString
      int ret = callPyFunc<int>("PyRun_SimpleString", (uint32_t)wasmBuf).at(0).of.i32;

      wasm_runtime_module_free(moduleInst, wasmBuf);

      if (ret != 0) {
         callPyFunc<void>("PyErr_Print");
         throw std::runtime_error{"Script execution failed"};
      }
   }

    // --- HELPER: Create a Python String in WASM Memory ---
    uint32_t createPyString(const std::string& str) {
        void* nativeBuf = nullptr;
        uint64_t wasmBuf = wasm_runtime_module_malloc(moduleInst, str.size() + 1, &nativeBuf);
        if (!nativeBuf) throw std::runtime_error{"Malloc failed for string"};
        memcpy(nativeBuf, str.c_str(), str.size() + 1);

        uint32_t pStr = callPyFunc<int32_t>("PyUnicode_DecodeFSDefault", (uint32_t)wasmBuf).at(0).of.i32;

        wasm_runtime_module_free(moduleInst, wasmBuf);
        if (pStr == 0) {
             callPyFunc<void>("PyErr_Print");
             throw std::runtime_error{"Failed to create Python string"};
        }
        return pStr;
    }

    // --- HELPER: Set Attribute using Python Objects ---
    void setAttr(uint32_t pObj, const std::string& name, uint32_t pVal) {
        uint32_t pName = createPyString(name);
        int ret = callPyFunc<int>("PyObject_SetAttr", pObj, pName, pVal).at(0).of.i32;
        callPyFunc<void>("Py_DecRef", pName);
        if (ret != 0) {
             callPyFunc<void>("PyErr_Print");
             throw std::runtime_error{"Failed to set attribute: " + name};
        }
    }

   // --- HELPER: Read a Python String (PyObject*) into C++ std::string ---
   std::string readPyString(uint32_t pStr) {
       if (!pStr) return "";

       // PyUnicode_AsUTF8 returns a const char* (which is an int32 pointer in Wasm)
       // Note: The buffer is owned by Python; it's valid as long as pStr exists.
       uint32_t wasmPtr = callPyFunc<uint32_t>("PyUnicode_AsUTF8", pStr).at(0).of.i32;
       if (!wasmPtr) {
           callPyFunc<void>("PyErr_Print");
           throw std::runtime_error("Failed to decode Python string via PyUnicode_AsUTF8");
       }

       // Convert Wasm pointer to Host pointer
       char* hostPtr = (char*)wasm_runtime_addr_app_to_native(moduleInst, wasmPtr);
       if (!hostPtr) {
            throw std::runtime_error("Invalid memory access: could not map WASM string to Host");
       }
       return std::string(hostPtr);
   }

   // --- FUNCTION: Get Inittab Entries as a String ---
   // Correctly uses a mix of Python Script (to create the string) and C-API (to read it).
   std::string getInittabString() {
       // 1. Run Python to create the string and attach it to 'sys' module
       // We attach it to 'sys' because it's easy to retrieve from C++.
       const char* script = R"(
import sys
# Create the string and save it as an attribute on sys
sys._inittab_debug_dump = "\n".join(sorted(sys.builtin_module_names))
)";
       runSimpleString(script);

       // 2. Import 'sys' module using C-API
       // We create the string "sys" first
       uint32_t pSysName = createPyString("sys");
       uint32_t pSys = callPyFunc<uint32_t>("PyImport_Import", pSysName).at(0).of.i32;
       callPyFunc<void>("Py_DecRef", pSysName); // Clean up name

       if (!pSys) {
           callPyFunc<void>("PyErr_Print");
           throw std::runtime_error("Failed to import 'sys'");
       }

       // 3. Get the attribute '_inittab_debug_dump'
       uint32_t pAttrName = createPyString("_inittab_debug_dump");
       uint32_t pStr = callPyFunc<uint32_t>("PyObject_GetAttr", pSys, pAttrName).at(0).of.i32;
       callPyFunc<void>("Py_DecRef", pAttrName); // Clean up name
       callPyFunc<void>("Py_DecRef", pSys);      // Clean up module

       if (!pStr) {
           callPyFunc<void>("PyErr_Print");
           throw std::runtime_error("Failed to get inittab string from sys");
       }

       // 4. Convert to C++ string
       std::string result = readPyString(pStr);

       // 5. Cleanup the result object
       callPyFunc<void>("Py_DecRef", pStr);

       return result;
   }

   // --- HELPER: Pre-register a module in sys.modules with a valid __spec__ ---
   void preRegisterModule(const std::string& moduleName) {
      std::string script = std::format(R"(
import sys
import importlib.util
import types

name = "{}"
# 1. Get or Create the Module
if name in sys.modules:
   mod = sys.modules[name]
else:
   mod = types.ModuleType(name)
   sys.modules[name] = mod

# 2. Patch Metadata if missing
# This is crucial: without __spec__, importlib treats the module as 'uninitialized'
# and will try to reload it from disk, failing with ModuleNotFoundError.
if not hasattr(mod, '__spec__') or mod.__spec__ is None:
   spec = importlib.util.spec_from_loader(name, loader=None, origin='pre-registered')
   spec.has_location = False
   mod.__spec__ = spec
   mod.__package__ = name.rpartition('.')[0]
)", moduleName);
      runSimpleString(script);
   }

  void injectExtensionModule(const std::string& moduleName, const std::string& initFuncName) {
    std::cout << "Injecting Extension: " << moduleName << "..." << std::endl;

    // 1. Pre-register the placeholder
    // This MUST happen before PyInit, because PyInit triggers the imports that cause the cycle.
    preRegisterModule(moduleName);

    // 2. Call PyInit to get the Module Definition
    wasm_function_inst_t initFunc = wasm_runtime_lookup_function(moduleInst, initFuncName.c_str());
    if (!initFunc) throw std::runtime_error{std::format("Init function '{}' not found.", initFuncName)};

    std::vector<wasm_val_t> results(1);
    std::vector<wasm_val_t> args(0);
    bool success = wasm_runtime_call_wasm_a(execEnv, initFunc, 1, results.data(), 0, args.data());

    if (!success) {
        callPyFunc<void>("PyErr_Print");
        throw std::runtime_error{wasm_runtime_get_exception(moduleInst)};
    }

    uint32_t pDef = results[0].of.i32;
    if (pDef == 0) {
        callPyFunc<void>("PyErr_Print");
        throw std::runtime_error{"PyInit returned NULL"};
    }

    // 3. Retrieve the Placeholder Module (Object Identity Preservation)
    int32_t pSysModules = callPyFunc<int32_t>("PyImport_GetModuleDict").at(0).of.i32;
    uint32_t pName = createPyString(moduleName);

    // Borrowed reference
    uint32_t pModule = callPyFunc<uint32_t>("PyDict_GetItem", pSysModules, pName).at(0).of.i32;
    callPyFunc<void>("Py_DecRef", pName);

    if (pModule == 0) {
        throw std::runtime_error("Pre-registered module disappeared from sys.modules!");
    }

    // 4. Execute the Definition into the Existing Module
    int execRet = callPyFunc<int>("PyModule_ExecDef", pModule, pDef).at(0).of.i32;

    if (execRet == -1) {
       callPyFunc<void>("PyErr_Print");
       throw std::runtime_error{"Failed to execute module definition"};
    }

    std::cout << "Success: " << moduleName << " loaded and populated." << std::endl;
}

   wasm_exec_env_t execEnv;
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
   uint64_t wasmBuf = wasm_runtime_module_malloc(pyEnv->moduleInst, moduleName.size() + 1, &nativeBuf);
   if (!nativeBuf) throw std::runtime_error{"Malloc failed"};
   memcpy(nativeBuf, moduleName.c_str(), moduleName.size() + 1);

   auto pName = pyEnv->callPyFunc<PyObjectRef>("PyUnicode_DecodeFSDefault", (uint32_t)wasmBuf).at(0).of.i32;
   wasm_runtime_module_free(pyEnv->moduleInst, wasmBuf);

   auto pModule = pyEnv->callPyFunc<PyObjectRef>("PyImport_Import", pName).at(0).of.i32;
   if (!pModule) {
      pyEnv->callPyFunc<void>("PyErr_Print");
      throw std::runtime_error{"Module not found: " + moduleName};
   }
   return {pModule, pyEnv};
}



int main(int argc, char** argv) {
   std::unique_ptr<WasmRuntime> runtime = std::make_unique<WasmRuntime>();
   std::unique_ptr<CPythonModule> cpythonModule = std::make_unique<CPythonModule>(std::move(runtime));
   std::unique_ptr<CPythonInst> cpythonInst = std::make_unique<CPythonInst>(std::move(cpythonModule));
   std::shared_ptr<PyEnv> pyEnv = std::make_shared<PyEnv>(std::move(cpythonInst));
   pyEnv->testSetup();

   // --- CHECK INITTAB ---
   try {
      std::cout << "--- Registered Inittab Modules ---" << std::endl;
      std::string modules = pyEnv->getInittabString();
      std::cout << modules << std::endl;
      std::cout << "----------------------------------" << std::endl;
   } catch (const std::exception& e) {
      std::cerr << "Error checking inittab: " << e.what() << std::endl;
   }

   try {
      // --- MANUAL EXTENSION LOADING ---
      // We attempt to manually inject the NumPy extension.
      // This function contains the crucial fix for circular dependencies (pre-registering __spec__).
      // NOTE: If you see "numpy._core._multiarray_umath" in the Inittab output above,
      // you can comment this line out, as Python will load it automatically!
      pyEnv->injectExtensionModule("numpy._core._multiarray_umath", "PyInit__multiarray_umath");

      // Optional: Inject other modules if needed
      // pyEnv->injectExtensionModule("numpy.linalg._umath_linalg", "PyInit__umath_linalg");

   } catch (const std::exception& e) {
      std::cerr << "Fatal Error injecting extensions: " << e.what() << std::endl;
      return 1;
   }

   // --- Suffix Hack ---
   const char* addSuffixScript =
   "import importlib.machinery\n"
   "importlib.machinery.EXTENSION_SUFFIXES.append('.cpython-313-wasm32-emscripten.so')\n";

   void* nativeBuf = nullptr;
   uint64_t instBufAddr = wasm_runtime_module_malloc(pyEnv->moduleInst, strlen(addSuffixScript) + 1, &nativeBuf);
   if (nativeBuf) {
       memcpy(nativeBuf, addSuffixScript, strlen(addSuffixScript) + 1);
       int runResult = pyEnv->callPyFunc<int>("PyRun_SimpleString", static_cast<uint32_t>(instBufAddr)).at(0).of.i32;
       wasm_runtime_module_free(pyEnv->moduleInst, instBufAddr);
       if (runResult != 0) pyEnv->callPyFunc<void>("PyErr_Print");
   }

   // --- Import NumPy ---
   try {
      std::cerr << "Importing numpy module..." << std::endl;
      PyObjectRef numpyModule = importModule(std::string{"numpy"}, pyEnv);
      std::cout << "NumPy imported successfully!" << std::endl;
   } catch (const std::exception& e) {
       std::cerr << "Failed to import numpy: " << e.what() << std::endl;
   }

   return 0;
}
