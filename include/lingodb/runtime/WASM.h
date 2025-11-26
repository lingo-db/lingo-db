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
struct WASMSession {
   public:
   WASMSession(wasm_exec_env_t execEnv, wasm_module_inst_t moduleInst)
      : execEnv(execEnv),
        moduleInst(moduleInst) {}
   wasm_exec_env_t execEnv;
   wasm_module_inst_t moduleInst;
   // - Outs... must be supplied explicitly
   // - Ins... are deduced from the provided inputs.
   template <typename... Out, typename... Ins>
   std::vector<wasm_val_t> callPyFunc(std::string_view funcName, Ins&&... ins) {
      wasm_function_inst_t func = getPyFunc(funcName.data());
      std::vector<wasm_val_t> args(sizeof...(Ins));
      uint32_t numArgs = 0, num_results = 0;
      serializeArgs<Out...>(args.data(), numArgs, num_results, ins...);
      std::vector<wasm_val_t> results(num_results);
      assert(numArgs == sizeof...(Ins));
      bool success = wasm_runtime_call_wasm_a(execEnv, func, num_results, results.data(), numArgs, args.data());
      if (!success) {
         /* exception is thrown if call fails */
         throw std::runtime_error{wasm_runtime_get_exception(moduleInst)};
      }
      return results;
   }

   uint32_t createWasmStringBuffer(std::string str) {
      void* nativeBufAddr = nullptr;

      uint64_t instBufAddr = wasm_runtime_module_malloc_internal(moduleInst, execEnv, strlen(str.c_str()) + 1, &nativeBufAddr);
      if (!nativeBufAddr) {
         throw std::runtime_error(wasm_runtime_get_exception(moduleInst));
      }
      char* nativeCharBuf = std::bit_cast<char*>(nativeBufAddr);
      memcpy(nativeCharBuf, str.c_str(), strlen(str.c_str()) + 1);
      return instBufAddr;
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
   int countResults() {
      int count = 0;
      if constexpr (!std::is_same_v<T, void>) count = 1;
      if constexpr (sizeof...(Rest) == 0)
         return count;
      else
         return count + countResults<Rest...>();
   }

   template <typename... Outs, typename... Ins>
   void serializeArgs(wasm_val_t* args, uint32_t& numArgs, uint32_t& numResults, Ins&&... ins) {
      numArgs = 0;
      numResults = countResults<Outs...>();
      // Use fold expression to pack each input in order
      (packVal(static_cast<void*>(args), numArgs, std::forward<Ins>(ins)), ...);
   }

   wasm_function_inst_t getPyFunc(const char* funcName) {
      wasm_function_inst_t func = wasm_runtime_lookup_function(moduleInst, funcName);
      if (!func)
         throw std::runtime_error{std::format("Function '{}' not found in module", funcName)};
      return func;
   }
};
class WASM {
   public:
   static std::vector<std::shared_ptr<WASMSession>> localWasmSessions;
   static WASMSession initializeWASM();
};
} // namespace lingodb::wasm

#endif //LINGODB_WASM_H
