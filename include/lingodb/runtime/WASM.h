#ifndef LINGODB_WASM_H
#define LINGODB_WASM_H

#include "lingodb/catalog/Catalog.h"

#include "wasm_c_api_internal.h"
#include "wasm_export.h"

#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <assert.h>

namespace lingodb::wasm {

enum CommonPyFunc : uint8_t {
   Py_Initialize = 0,
   PyImport_GetModuleDict = 1,
   PyDict_GetItemString = 2,
   Py_IncRef = 3,
   PyModule_New = 4,
   PyModule_GetDict = 5,
   PyEval_GetBuiltins = 6,
   PyDict_SetItemString = 7,
   Py_DECREF = 8,
   PyRun_String = 9,
   PyObject_GetAttrString = 10,
   PyErr_Print = 11,
   PyObject_GetAttr = 12,
   PyObject_SetAttrString = 13,
   PyObject_Vectorcall = 14,
   PyLong_AsLongLong = 15,
   PyLong_FromLongLong = 16,
   PyObject_IsTrue = 17,
   PyBool_FromLong = 18,
   PyUnicode_AsUTF8 = 19,
   PyUnicode_GetLength = 20,
   PyUnicode_FromStringAndSize = 21,
   PyFloat_AsDouble = 22,
   PyFloat_FromDouble = 23,
   PyImport_ImportModule = 24,
};
static std::vector<std::pair<CommonPyFunc, std::string>> commonPyFuncNames = {
   {CommonPyFunc::Py_Initialize, "Py_Initialize"},
   {CommonPyFunc::PyImport_GetModuleDict, "PyImport_GetModuleDict"},
   {CommonPyFunc::PyDict_GetItemString, "PyDict_GetItemString"},
   {CommonPyFunc::Py_IncRef, "Py_IncRef"},
   {CommonPyFunc::PyModule_New, "PyModule_New"},
   {CommonPyFunc::PyModule_GetDict, "PyModule_GetDict"},
   {CommonPyFunc::PyEval_GetBuiltins, "PyEval_GetBuiltins"},
   {CommonPyFunc::PyDict_SetItemString, "PyDict_SetItemString"},
   {CommonPyFunc::Py_DECREF, "Py_DecRef"},
   {CommonPyFunc::PyRun_String, "PyRun_String"},
   {CommonPyFunc::PyObject_GetAttrString, "PyObject_GetAttrString"},
   {CommonPyFunc::PyErr_Print, "PyErr_Print"},
   {CommonPyFunc::PyObject_GetAttr, "PyObject_GetAttr"},
   {CommonPyFunc::PyObject_SetAttrString, "PyObject_SetAttrString"},
   {CommonPyFunc::PyObject_Vectorcall, "PyObject_Vectorcall"},
   {CommonPyFunc::PyLong_AsLongLong, "PyLong_AsLongLong"},
   {CommonPyFunc::PyLong_FromLongLong, "PyLong_FromLongLong"},
   {CommonPyFunc::PyObject_IsTrue, "PyObject_IsTrue"},
   {CommonPyFunc::PyBool_FromLong, "PyBool_FromLong"},
   {CommonPyFunc::PyUnicode_AsUTF8, "PyUnicode_AsUTF8"},
   {CommonPyFunc::PyUnicode_GetLength, "PyUnicode_GetLength"},
   {CommonPyFunc::PyUnicode_FromStringAndSize, "PyUnicode_FromStringAndSize"},
   {CommonPyFunc::PyFloat_AsDouble, "PyFloat_AsDouble"},
   {CommonPyFunc::PyFloat_FromDouble, "PyFloat_FromDouble"},
   {CommonPyFunc::PyImport_ImportModule, "PyImport_ImportModule"},
};
struct WASMSession {
   std::vector<wasm_function_inst_t> funcs;
   size_t allocatedTmpSpace = 128 * 1024; //128 KiB
   size_t tmpSpaceAvailable = allocatedTmpSpace;
   uint32_t tmpSpaceAddr = 0;
   uint8_t* tmpSpaceTmpAddrNative = nullptr;
   uint32_t currTmpSpaceAddr = 0;
   uint8_t* currTmpSpaceAddrNative = nullptr;

   std::pair<uint32_t, void*> allocateFromTmpSpace(size_t size) {
      if (size > tmpSpaceAvailable) {
         return {0, nullptr};
      }
      uint32_t addr = currTmpSpaceAddr;
      currTmpSpaceAddr += size;
      void* nativeAddr = currTmpSpaceAddrNative;
      currTmpSpaceAddrNative += size;
      tmpSpaceAvailable -= size;
      return {addr, nativeAddr};
   }
   void resetTmpSpace() {
      tmpSpaceAvailable = allocatedTmpSpace;
      currTmpSpaceAddr = tmpSpaceAddr;
      currTmpSpaceAddrNative = tmpSpaceTmpAddrNative;
   }
   void allocateTmpSpace() {
      tmpSpaceAddr = createWasmBuffer(allocatedTmpSpace);

      tmpSpaceAddr = wasm_runtime_module_malloc_internal(moduleInst, execEnv, allocatedTmpSpace, (void**) &tmpSpaceTmpAddrNative);
      currTmpSpaceAddr = tmpSpaceAddr;
      currTmpSpaceAddrNative = tmpSpaceTmpAddrNative;
   }

   public:
   std::vector<uint32_t> cachedObjects;
   uint32_t get(size_t idx) {
      if (idx >= cachedObjects.size()) {
         cachedObjects.resize(idx + 1, 0);
      }
      return cachedObjects[idx];
   }
   void set(size_t idx, uint32_t obj) {
      if (idx >= cachedObjects.size()) {
         cachedObjects.resize(idx + 1, 0);
      }
      cachedObjects[idx] = obj;
   }
   void clearCache() {
      cachedObjects.clear();
   }
   class WASMTmpString {
      WASMSession& session;
      uint32_t addr;
      uint8_t* nativeAddr;
      bool owned = true;

      public:
      WASMTmpString(WASMSession& session, uint32_t addr, uint8_t* nativeAddr, bool owned = true)
         : session(session), addr(addr), nativeAddr(nativeAddr), owned(owned) {}
      uint32_t getAddr() const {
         return addr;
      }
      uint8_t* getNativeAddr() const {
         return nativeAddr;
      }
      ~WASMTmpString() {
         if (owned) {
            session.freeWasmBuffer(addr);
         }
      }
   };
   class WasmSessionTmpScope {
      WASMSession& session;

      public:
      WasmSessionTmpScope(WASMSession& session)
         : session(session) {
      }
      WASMTmpString allocateRaw(size_t size) {
         auto [addr, nativeAddr] = session.allocateFromTmpSpace(size);
         if (addr == 0) {
            uint8_t* nativeBufAddr = nullptr;
            uint64_t instBufAddr = wasm_runtime_module_malloc_internal(session.moduleInst, session.execEnv, size, (void**) &nativeBufAddr);

            return WASMTmpString(session, addr, nativeBufAddr, true);
         }
         return WASMTmpString(session, addr, (uint8_t*) nativeAddr, false);
      }
      WASMTmpString allocateString(std::string_view strView) {
         auto tmpStr = allocateRaw(strView.size() + 1);
         memcpy(tmpStr.getNativeAddr(), strView.data(), strView.size());
         tmpStr.getNativeAddr()[strView.size()] = '\0';
         return tmpStr;
      }
      ~WasmSessionTmpScope() {
         session.resetTmpSpace();
      }
   };
   WASMSession(wasm_exec_env_t execEnv, wasm_module_inst_t moduleInst)
      : execEnv(execEnv),
        moduleInst(moduleInst) {
      for (const auto& [funcId, funcName] : commonPyFuncNames) {
         wasm_function_inst_t func = wasm_runtime_lookup_function(moduleInst, funcName.c_str());
         if (!func) {
            throw std::runtime_error(std::format("Failed to lookup function '{}'", funcName));
         }
         if (funcId >= funcs.size()) {
            funcs.resize(funcId + 1);
         }
         funcs[funcId] = func;
      }
      allocateTmpSpace();
   }

   WasmSessionTmpScope createTmpScope() {
      return WasmSessionTmpScope{*this};
   }
   wasm_exec_env_t execEnv;
   wasm_module_inst_t moduleInst;
   // - Outs... must be supplied explicitly
   // - Ins... are deduced from the provided inputs.
   template <typename... Out, typename... Ins>
   auto callPyFunc(CommonPyFunc funcId, Ins&&... ins) {
      wasm_function_inst_t func = funcs[funcId];
      std::array<wasm_val_t, sizeof...(Ins)> args;
      uint32_t numArgs = 0;
      constexpr size_t num_results = countResults<Out...>();
      serializeArgs(args.data(), numArgs, ins...);
      std::array<wasm_val_t, num_results> results;
      assert(numArgs == sizeof...(Ins));
      bool success = wasm_runtime_call_wasm_a(execEnv, func, num_results, results.data(), numArgs, args.data());
      if (!success) {
         /* exception is thrown if call fails */
         throw std::runtime_error{wasm_runtime_get_exception(moduleInst)};
      }
      return results;
   }
   template <class T>
   T* getAddr(uint32_t addr) {
      return reinterpret_cast<T*>(wasm_runtime_addr_app_to_native(moduleInst, addr));
   }

   uint32_t createWasmBuffer(size_t size) {
      void* nativeBufAddr = nullptr;

      uint64_t instBufAddr = wasm_runtime_module_malloc_internal(moduleInst, execEnv, size, &nativeBufAddr);

      return instBufAddr;
   }
   void freeWasmBuffer(uint32_t addr) {
      wasm_runtime_module_free_internal(moduleInst, execEnv, addr);
   }

   public:
   private:
   template <typename T>
   void packVal(void* argsVoid, uint32_t& idx, T v)
      requires(std::integral<T> && sizeof(T) == 4)
   {
      auto& entry = static_cast<wasm_val_t*>(argsVoid)[idx++];
      entry.of.i32 = v;
      entry.kind = WASM_I32;
   }
   template <typename T>
   void packVal(void* argsVoid, uint32_t& idx, T v)
      requires(std::integral<T> && sizeof(T) == 8)
   {
      auto& entry = static_cast<wasm_val_t*>(argsVoid)[idx++];
      entry.of.i64 = static_cast<int64_t>(v);
      entry.kind = WASM_I64;
   }
   inline void packVal(void* argsVoid, uint32_t& idx, float v) {
      auto& entry = static_cast<wasm_val_t*>(argsVoid)[idx++];
      entry.of.f32 = v;
      entry.kind = WASM_F32;
   }
   inline void packVal(void* argsVoid, uint32_t& idx, double v) {
      auto& entry = static_cast<wasm_val_t*>(argsVoid)[idx++];
      entry.of.f64 = v;
      entry.kind = WASM_F64;
   }

   // Fallback to produce a clear compile-time error for unsupported types.
   template <typename T>
   void packVal(void* /*args*/, uint32_t& /*idx*/, T /*v*/) {
      static_assert(sizeof(T) == 0, "pack_val: unsupported argument type");
   }

   template <typename T, typename... Rest>
   static constexpr int countResults() {
      int count = 0;
      if constexpr (!std::is_same_v<T, void>) count = 1;
      if constexpr (sizeof...(Rest) == 0)
         return count;
      else
         return count + countResults<Rest...>();
   }

   template <typename... Ins>
   void serializeArgs(wasm_val_t* args, uint32_t& numArgs, Ins&&... ins) {
      numArgs = 0;
      // Use fold expression to pack each input in order
      (packVal(static_cast<void*>(args), numArgs, std::forward<Ins>(ins)), ...);
   }
};
class WASM {
   public:
   static std::vector<std::shared_ptr<WASMSession>> localWasmSessions;
   static WASMSession* initializeWASM();
};
} // namespace lingodb::wasm

#endif //LINGODB_WASM_H
