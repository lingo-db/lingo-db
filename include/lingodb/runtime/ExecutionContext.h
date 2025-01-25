#ifndef LINGODB_RUNTIME_EXECUTIONCONTEXT_H
#define LINGODB_RUNTIME_EXECUTIONCONTEXT_H
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>

#include "Session.h"

#include <lingodb/scheduler/Scheduler.h>
#include "ConcurrentMap.h"
namespace lingodb::runtime {
class Database;
//some state required for query processing;
struct State {
   void* ptr = nullptr;
   std::function<void(void*)> freeFn;
};
class ExecutionContext {
   std::unordered_map<uint32_t, uint8_t*> results;
   std::unordered_map<uint32_t, int64_t> tupleCounts;
   ConcurrentMap<void*, State> states;
   std::vector<std::unordered_map<size_t, State>> allocators;
   Session& session;

   public:
   ExecutionContext(Session& session) : session(session) { allocators.resize(lingodb::scheduler::getNumWorkers()); }
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
   void setResult(uint32_t id, uint8_t* ptr);
   void clearResult(uint32_t id);
   void setTupleCount(uint32_t id, int64_t tupleCount);
   void registerState(const State& s) {
      states.insert(s.ptr, s);
   }
   State& getAllocator(size_t group) {
      return allocators[lingodb::scheduler::currentWorkerId()][group];
   }
   void reset();
   ~ExecutionContext();
};
} // end namespace lingodb::runtime

#endif // LINGODB_RUNTIME_EXECUTIONCONTEXT_H
