#ifndef LINGODB_WASM_H
#define LINGODB_WASM_H

#include "lingodb/catalog/Catalog.h"

#include "wasix_python_bridge.h"

#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lingodb::wasm {

// Stable host-side IDs for the CPython C-API functions we look up by name in
// the guest. Numbering is internal — only used to index `WASMSession::funcs`.
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
   lingodb_wasix_session_t* shim = nullptr;
   // Cached shim function indices, indexed by CommonPyFunc.
   std::vector<int32_t> funcs;
   int32_t guestMallocIdx = -1;
   int32_t guestFreeIdx = -1;

   // Pre-allocated bump arena in the guest's linear memory; reset (bump
   // pointer rewinds) per call via WasmSessionTmpScope.
   size_t allocatedTmpSpace = 128 * 1024;
   size_t tmpSpaceAvailable = allocatedTmpSpace;
   uint32_t tmpSpaceAddr = 0;
   uint32_t currTmpSpaceAddr = 0;

   uint32_t allocateFromTmpSpace(size_t size) {
      if (size > tmpSpaceAvailable) return 0;
      uint32_t addr = currTmpSpaceAddr;
      currTmpSpaceAddr += size;
      tmpSpaceAvailable -= size;
      return addr;
   }
   void resetTmpSpace() {
      tmpSpaceAvailable = allocatedTmpSpace;
      currTmpSpaceAddr = tmpSpaceAddr;
   }

   public:
   // Mirrors PythonExtState in the CPython embed path; currently unused on
   // the WASIX side but kept on the public surface for parity.
   std::vector<uint32_t> cachedObjects;
   uint32_t get(size_t idx) {
      if (idx >= cachedObjects.size()) cachedObjects.resize(idx + 1, 0);
      return cachedObjects[idx];
   }
   void set(size_t idx, uint32_t obj) {
      if (idx >= cachedObjects.size()) cachedObjects.resize(idx + 1, 0);
      cachedObjects[idx] = obj;
   }
   void clearCache() { cachedObjects.clear(); }

   // Translate a guest linear-memory offset to a native pointer. Cheap, but
   // invalidated if the guest grows its memory — recompute at use, never
   // cache across wasm calls.
   uint8_t* nativeAddr(uint32_t guestOffset) {
      return lingodb_wasix_memory_base(shim) + guestOffset;
   }

   class WASMTmpString {
      WASMSession& session;
      uint32_t addr;
      bool owned;

      public:
      WASMTmpString(WASMSession& session, uint32_t addr, bool owned)
         : session(session), addr(addr), owned(owned) {}
      uint32_t getAddr() const { return addr; }
      uint8_t* getNativeAddr() const { return session.nativeAddr(addr); }
      ~WASMTmpString() {
         if (owned) session.freeWasmBuffer(addr);
      }
   };
   class WasmSessionTmpScope {
      WASMSession& session;

      public:
      WasmSessionTmpScope(WASMSession& session) : session(session) {}
      WASMTmpString allocateRaw(size_t size) {
         uint32_t addr = session.allocateFromTmpSpace(size);
         if (addr == 0) {
            uint32_t guestPtr = session.createWasmBuffer(size);
            return WASMTmpString(session, guestPtr, /*owned=*/true);
         }
         return WASMTmpString(session, addr, /*owned=*/false);
      }
      WASMTmpString allocateString(std::string_view strView) {
         auto tmpStr = allocateRaw(strView.size() + 1);
         std::memcpy(tmpStr.getNativeAddr(), strView.data(), strView.size());
         tmpStr.getNativeAddr()[strView.size()] = '\0';
         return tmpStr;
      }
      ~WasmSessionTmpScope() { session.resetTmpSpace(); }
   };

   explicit WASMSession(lingodb_wasix_session_t* shim) : shim(shim) {
      // Resolve the CPython C-API exports + the guest libc allocator.
      for (const auto& [funcId, funcName] : commonPyFuncNames) {
         int32_t idx = lingodb_wasix_lookup_func(shim, funcName.c_str());
         if (idx < 0) {
            throw std::runtime_error(std::format("missing wasm export: {}", funcName));
         }
         if (funcId >= funcs.size()) funcs.resize(funcId + 1, -1);
         funcs[funcId] = idx;
      }
      guestMallocIdx = lingodb_wasix_lookup_func(shim, "malloc");
      guestFreeIdx = lingodb_wasix_lookup_func(shim, "free");
      if (guestMallocIdx < 0 || guestFreeIdx < 0) {
         throw std::runtime_error("wasm module is missing `malloc`/`free` exports");
      }

      // Note: Py_Initialize is already called by the shim's session_new
      // (mirrors embed-python.rs). No need to call it again here.

      allocateTmpSpace();
   }

   ~WASMSession() {
      if (shim) lingodb_wasix_session_free(shim);
   }

   WasmSessionTmpScope createTmpScope() {
      return WasmSessionTmpScope{*this};
   }

   // - Outs... must be supplied explicitly (the result-type list).
   // - Ins... are deduced from the provided inputs.
   template <typename... Out, typename... Ins>
   auto callPyFunc(CommonPyFunc funcId, Ins&&... ins) {
      int32_t funcIdx = funcs[funcId];
      std::array<lingodb_wasm_val_t, sizeof...(Ins) == 0 ? 1 : sizeof...(Ins)> argsBuf{};
      uint32_t numArgs = 0;
      constexpr size_t num_results = countResults<Out...>();
      serializeArgs(argsBuf.data(), numArgs, ins...);
      std::array<lingodb_wasm_val_t, num_results == 0 ? 1 : num_results> resultsBuf{};

      int rc = lingodb_wasix_call(
         shim, funcIdx,
         argsBuf.data(), sizeof...(Ins),
         resultsBuf.data(), num_results);
      if (rc != 0) {
         const char* msg = lingodb_wasix_last_error();
         throw std::runtime_error(std::format("wasm trap in funcId={}: {}",
                                              static_cast<int>(funcId),
                                              msg ? msg : "<no message>"));
      }
      std::array<lingodb_wasm_val_t, num_results> out{};
      for (size_t i = 0; i < num_results; ++i) out[i] = resultsBuf[i];
      return out;
   }

   template <class T>
   T* getAddr(uint32_t addr) {
      return reinterpret_cast<T*>(nativeAddr(addr));
   }

   uint32_t createWasmBuffer(size_t size) {
      lingodb_wasm_val_t arg{.kind = LINGODB_WASM_I32, ._pad = {}, .of = {.i32 = static_cast<int32_t>(size)}};
      lingodb_wasm_val_t res{};
      int rc = lingodb_wasix_call(shim, guestMallocIdx, &arg, 1, &res, 1);
      if (rc != 0) {
         const char* msg = lingodb_wasix_last_error();
         throw std::runtime_error(std::format("guest malloc trapped: {}", msg ? msg : ""));
      }
      return static_cast<uint32_t>(res.of.i32);
   }
   void freeWasmBuffer(uint32_t addr) {
      lingodb_wasm_val_t arg{.kind = LINGODB_WASM_I32, ._pad = {}, .of = {.i32 = static_cast<int32_t>(addr)}};
      int rc = lingodb_wasix_call(shim, guestFreeIdx, &arg, 1, nullptr, 0);
      (void) rc; // best-effort free
   }

   private:
   void allocateTmpSpace() {
      tmpSpaceAddr = createWasmBuffer(allocatedTmpSpace);
      currTmpSpaceAddr = tmpSpaceAddr;
   }

   template <typename T>
   void packVal(void* argsVoid, uint32_t& idx, T v)
      requires(std::integral<T> && sizeof(T) == 4)
   {
      auto& entry = static_cast<lingodb_wasm_val_t*>(argsVoid)[idx++];
      entry.of.i32 = static_cast<int32_t>(v);
      entry.kind = LINGODB_WASM_I32;
   }
   template <typename T>
   void packVal(void* argsVoid, uint32_t& idx, T v)
      requires(std::integral<T> && sizeof(T) == 8)
   {
      auto& entry = static_cast<lingodb_wasm_val_t*>(argsVoid)[idx++];
      entry.of.i64 = static_cast<int64_t>(v);
      entry.kind = LINGODB_WASM_I64;
   }
   inline void packVal(void* argsVoid, uint32_t& idx, float v) {
      auto& entry = static_cast<lingodb_wasm_val_t*>(argsVoid)[idx++];
      entry.of.f32 = v;
      entry.kind = LINGODB_WASM_F32;
   }
   inline void packVal(void* argsVoid, uint32_t& idx, double v) {
      auto& entry = static_cast<lingodb_wasm_val_t*>(argsVoid)[idx++];
      entry.of.f64 = v;
      entry.kind = LINGODB_WASM_F64;
   }
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
   static constexpr int countResults() { return 0; }

   template <typename... Ins>
   void serializeArgs(lingodb_wasm_val_t* args, uint32_t& numArgs, Ins&&... ins) {
      numArgs = 0;
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
