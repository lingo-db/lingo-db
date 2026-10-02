#ifndef LINGODB_RUNTIME_EXECUTIONCONTEXT_H
#define LINGODB_RUNTIME_EXECUTIONCONTEXT_H
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>

#include "ConcurrentMap.h"
#include "Session.h"
#include <lingodb/scheduler/Scheduler.h>
#ifdef USE_CPYTHON_WASM_RUNTIME
namespace lingodb::wasm {
struct WASMSession;
} // namespace lingodb::wasm
#endif
namespace lingodb::runtime {
class VarLen32;
class Database;
//some state required for query processing;
struct State {
   void* ptr = nullptr;
   std::function<void(void*)> freeFn;
};
struct Arena {
   static constexpr size_t thresholdDirectAlloc = 16 * 1024; // 16 KiB
   static constexpr size_t chunkSize = 1024 * 1024; // 1 MiB
   // chunk allocations + direct allocations
   std::vector<uint8_t*> allocations;
   // remaining from current Chunk
   size_t remainingBytes = 0;
   // pointer to still free space in current chunk
   uint8_t* currentChunkStart = nullptr;

   uint8_t* alloc(size_t bytes) {
      //case 1: bytes are larger than threshold, allocate directly
      if (bytes >= thresholdDirectAlloc) {
         uint8_t* ptr = (uint8_t*) malloc(bytes);
         allocations.push_back(ptr);
         return ptr;
      } else {
         if (remainingBytes < bytes) {
            //case 2: not enough space in current chunk, allocate new chunk
            uint8_t* newChunk = (uint8_t*) malloc(chunkSize);
            allocations.push_back(newChunk);
            currentChunkStart = newChunk;
            remainingBytes = chunkSize;
         }
         // now we have enough space in the current chunk
         uint8_t* ptr = currentChunkStart;
         currentChunkStart += bytes;
         remainingBytes -= bytes;
         return ptr;
      }
   }

   ~Arena() {
      for (auto& alloc : allocations) {
         free(alloc);
      }
   }
};

class ExecutionContext {
   std::unordered_map<uint32_t, uint8_t*> results;
   std::unordered_map<uint32_t, int64_t> tupleCounts;
   std::vector<std::unordered_map<size_t, State>> allocators;
   std::vector<std::vector<State>> perWorkerStates;
   std::vector<Arena> stringArenas;
   Session& session;

   public:
   ExecutionContext(Session& session) : session(session) {
      allocators.resize(lingodb::scheduler::getNumWorkers());
      stringArenas.resize(lingodb::scheduler::getNumWorkers());
      perWorkerStates.resize(lingodb::scheduler::getNumWorkers());
#ifdef USE_CPYTHON_RUNTIME
      resetPythonSessionCache();
#endif
   }
#ifdef USE_CPYTHON_RUNTIME
   void resetPythonSessionCache();
#endif
   Session& getSession() {
      return session;
   }
   template <class T>
   std::optional<T*> getResultOfType(uint32_t id) {
      if (results.contains(id)) {
         return (T*) results[id];
      } else {
         return {};
      }
   }
   std::optional<int64_t> getTupleCount(uint32_t id) {
      if (tupleCounts.contains(id)) {
         return tupleCounts[id];
      } else {
         return {};
      }
   }
   const std::unordered_map<uint32_t, int64_t>& getTupleCounts() const {
      return tupleCounts;
   }

   uint8_t* allocString(size_t bytes) {
      return stringArenas[lingodb::scheduler::currentWorkerId()].alloc(bytes);
   }
   static void setResult(uint32_t id, uint8_t* ptr);
   static uint8_t* allocStateRaw(size_t size);
   static void clearResult(uint32_t id);
   static void setTupleCount(uint32_t id, int64_t tupleCount);
   // throws a std::runtime_error, reported as a runtime error of the query
   static void raiseError(VarLen32 message);
   void registerState(const State& s) {
      perWorkerStates[lingodb::scheduler::currentWorkerId()].push_back(s);
   }
   State& getAllocator(size_t group) {
      return allocators[lingodb::scheduler::currentWorkerId()][group];
   }
#ifdef USE_CPYTHON_RUNTIME
   // Set before execution when the compiled plan contains py_interp ops, i.e.
   // the query actually runs the embedded interpreter (runtime python /
   // hipy_fallback / tabular UDFs). The many parallel pipeline tasks
   // (TaskWithImplicitContext) only attach a worker to CPython when this is set,
   // so plain SQL and pure ahead-of-time hipy queries pay no per-task Python
   // overhead — which otherwise measurably hurts their parallel scaling in a
   // Python-enabled build. The single main query task (TaskWithContext) always
   // attaches, since the flag is only known once execution has started.
   bool pythonUsed = false;
   void markPythonUsed() { pythonUsed = true; }
   bool isPythonUsed() const { return pythonUsed; }
   void setupPython();
   void teardownPython();
#endif
#ifdef USE_CPYTHON_WASM_RUNTIME
   void setupWasm();
   void teardownWasm();
   wasm::WASMSession* getWasmSession();
#endif
   ~ExecutionContext();
};

void setCurrentExecutionContext(ExecutionContext* context);
ExecutionContext* getCurrentExecutionContext();
} // end namespace lingodb::runtime

#endif // LINGODB_RUNTIME_EXECUTIONCONTEXT_H
