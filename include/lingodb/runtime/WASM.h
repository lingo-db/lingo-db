#ifndef LINGODB_WASM_H
#define LINGODB_WASM_H

#include "lingodb/catalog/Catalog.h"

#include "wasm.h"
#include "wasmer.h"

#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
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
   // Wasmer ownership graph: store owns instance/wasiEnv; module is shareable
   // across stores (engine-bound) but we hold one per session for now.
   wasm_store_t* store = nullptr;
   wasm_module_t* module = nullptr;
   wasm_instance_t* instance = nullptr;
   wasi_env_t* wasiEnv = nullptr;
   wasm_memory_t* memory = nullptr;     // borrowed from `exports`
   wasm_func_t* guestMalloc = nullptr;  // borrowed from `exports`
   wasm_func_t* guestFree = nullptr;    // borrowed from `exports`
   wasm_extern_vec_t exports{};
   wasm_exporttype_vec_t exportTypes{};

   // Cached CPython C-API entries indexed by CommonPyFunc — borrowed pointers
   // into `exports` (no separate ownership).
   std::vector<wasm_func_t*> funcs;

   // Pre-allocated scratch buffer in the guest's linear memory used for
   // short-lived host->guest string copies. Allocated once at session creation
   // by calling the guest-exported malloc; reset (bump-pointer) per call.
   size_t allocatedTmpSpace = 128 * 1024; // 128 KiB
   size_t tmpSpaceAvailable = allocatedTmpSpace;
   uint32_t tmpSpaceAddr = 0;
   uint32_t currTmpSpaceAddr = 0;

   uint32_t allocateFromTmpSpace(size_t size) {
      if (size > tmpSpaceAvailable) {
         return 0;
      }
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
   // Cache used by the WASM PythonRuntime path (mirrors PythonExtState in the
   // CPython embed path — kept for parity, currently unused on the WASM side).
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

   // Translate a guest linear-memory offset to a native pointer. Cheap (one
   // addition), but the result is invalidated if the guest grows its memory —
   // do not cache across wasm calls. Use at the moment of access.
   uint8_t* nativeAddr(uint32_t guestOffset) {
      return reinterpret_cast<uint8_t*>(wasm_memory_data(memory)) + guestOffset;
   }

   class WASMTmpString {
      WASMSession& session;
      uint32_t addr;
      bool owned;

      public:
      WASMTmpString(WASMSession& session, uint32_t addr, bool owned)
         : session(session), addr(addr), owned(owned) {}
      uint32_t getAddr() const { return addr; }
      // Recompute on demand because wasm_memory_data() can be invalidated by
      // memory.grow between calls — never cache the native pointer.
      uint8_t* getNativeAddr() const { return session.nativeAddr(addr); }
      ~WASMTmpString() {
         if (owned) {
            session.freeWasmBuffer(addr);
         }
      }
   };
   class WasmSessionTmpScope {
      WASMSession& session;

      public:
      WasmSessionTmpScope(WASMSession& session) : session(session) {}
      WASMTmpString allocateRaw(size_t size) {
         uint32_t addr = session.allocateFromTmpSpace(size);
         if (addr == 0) {
            // Tmp arena exhausted — fall back to a one-off guest malloc, freed
            // on WASMTmpString destruction.
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

   WASMSession(wasm_store_t* store, wasm_module_t* module,
               wasm_instance_t* instance, wasi_env_t* wasiEnv,
               wasm_extern_vec_t exports, wasm_exporttype_vec_t exportTypes)
      : store(store), module(module), instance(instance), wasiEnv(wasiEnv),
        exports(exports), exportTypes(exportTypes) {
      // Build a name -> export-index map. The two vecs are parallel (same
      // ordering by spec).
      std::unordered_map<std::string, size_t> nameToIdx;
      nameToIdx.reserve(exportTypes.size);
      for (size_t i = 0; i < exportTypes.size; ++i) {
         const wasm_name_t* n = wasm_exporttype_name(exportTypes.data[i]);
         nameToIdx.emplace(std::string(n->data, n->size), i);
      }
      auto lookupExtern = [&](const char* nm) -> wasm_extern_t* {
         auto it = nameToIdx.find(nm);
         return it == nameToIdx.end() ? nullptr : exports.data[it->second];
      };
      auto lookupFunc = [&](const char* nm) -> wasm_func_t* {
         wasm_extern_t* e = lookupExtern(nm);
         return e ? wasm_extern_as_func(e) : nullptr;
      };

      if (auto* m = lookupExtern("memory")) {
         memory = wasm_extern_as_memory(m);
      }
      if (!memory) {
         throw std::runtime_error("WASM module exports no `memory`");
      }

      // wasi-sdk CPython exports its libc allocator. Used both for the tmp
      // arena and for one-off allocations beyond the arena.
      guestMalloc = lookupFunc("malloc");
      guestFree = lookupFunc("free");
      if (!guestMalloc || !guestFree) {
         throw std::runtime_error("WASM module is missing `malloc`/`free` exports");
      }

      for (const auto& [funcId, funcName] : commonPyFuncNames) {
         wasm_func_t* func = lookupFunc(funcName.c_str());
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

   ~WASMSession() {
      if (exports.data) wasm_extern_vec_delete(&exports);
      if (exportTypes.data) wasm_exporttype_vec_delete(&exportTypes);
      if (instance) wasm_instance_delete(instance);
      if (wasiEnv) wasi_env_delete(wasiEnv);
      if (module) wasm_module_delete(module);
      if (store) wasm_store_delete(store);
   }

   WasmSessionTmpScope createTmpScope() {
      return WasmSessionTmpScope{*this};
   }

   // - Outs... must be supplied explicitly (the result-type list).
   // - Ins... are deduced from the provided inputs.
   template <typename... Out, typename... Ins>
   auto callPyFunc(CommonPyFunc funcId, Ins&&... ins) {
      wasm_func_t* func = funcs[funcId];
      std::array<wasm_val_t, sizeof...(Ins) == 0 ? 1 : sizeof...(Ins)> argsBuf;
      uint32_t numArgs = 0;
      constexpr size_t num_results = countResults<Out...>();
      serializeArgs(argsBuf.data(), numArgs, ins...);
      std::array<wasm_val_t, num_results == 0 ? 1 : num_results> resultsBuf;

      wasm_val_vec_t argsVec{
         .size = sizeof...(Ins),
         .data = argsBuf.data(),
      };
      wasm_val_vec_t resultsVec{
         .size = num_results,
         .data = resultsBuf.data(),
      };
      wasm_trap_t* trap = wasm_func_call(func, &argsVec, &resultsVec);
      if (trap) {
         wasm_message_t msg;
         wasm_trap_message(trap, &msg);
         std::string err(msg.data ? msg.data : "<no message>", msg.size);
         wasm_byte_vec_delete(&msg);
         wasm_trap_delete(trap);
         throw std::runtime_error(std::format("wasm trap: {}", err));
      }

      // Return a fixed-size std::array<wasm_val_t, num_results>; callers index
      // via .at(0).of.{i32,i64,f64} same as before.
      std::array<wasm_val_t, num_results> out{};
      for (size_t i = 0; i < num_results; ++i) out[i] = resultsBuf[i];
      return out;
   }

   template <class T>
   T* getAddr(uint32_t addr) {
      return reinterpret_cast<T*>(nativeAddr(addr));
   }

   uint32_t createWasmBuffer(size_t size) {
      wasm_val_t args[1] = {WASM_I32_VAL(static_cast<int32_t>(size))};
      wasm_val_t res[1] = {WASM_INIT_VAL};
      wasm_val_vec_t a{.size = 1, .data = args};
      wasm_val_vec_t r{.size = 1, .data = res};
      wasm_trap_t* trap = wasm_func_call(guestMalloc, &a, &r);
      if (trap) {
         wasm_trap_delete(trap);
         throw std::runtime_error("guest malloc trapped");
      }
      return static_cast<uint32_t>(res[0].of.i32);
   }
   void freeWasmBuffer(uint32_t addr) {
      wasm_val_t args[1] = {WASM_I32_VAL(static_cast<int32_t>(addr))};
      wasm_val_vec_t a{.size = 1, .data = args};
      wasm_val_vec_t r = WASM_EMPTY_VEC;
      wasm_trap_t* trap = wasm_func_call(guestFree, &a, &r);
      if (trap) wasm_trap_delete(trap);
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
   void serializeArgs(wasm_val_t* args, uint32_t& numArgs, Ins&&... ins) {
      numArgs = 0;
      (packVal(static_cast<void*>(args), numArgs, std::forward<Ins>(ins)), ...);
   }
};

class WASM {
   public:
   // One engine shared across all sessions / workers — Engine compiles wasm
   // and is thread-safe; Stores (and below) are per-thread.
   static wasm_engine_t* engine;
   static std::vector<std::shared_ptr<WASMSession>> localWasmSessions;
   static WASMSession* initializeWASM();
};
} // namespace lingodb::wasm

#endif //LINGODB_WASM_H
