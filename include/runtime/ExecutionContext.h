#ifndef RUNTIME_EXECUTIONCONTEXT_H
#define RUNTIME_EXECUTIONCONTEXT_H
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>

#include <oneapi/tbb.h>
namespace runtime {
class Database;
//some state required for query processing;
struct State {
   void* ptr;
   std::function<void(void*)> freeFn;
};
class ExecutionContext {
   std::unordered_map<uint32_t, uint8_t*> results;
   std::unordered_map<uint32_t, int64_t> tupleCounts;
   tbb::concurrent_hash_map<void*, State> states;

   public:
   int id;
   std::unique_ptr<Database> db;
   Database* getDatabase();
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
   void setResult(uint32_t id, uint8_t* ptr);
   void setTupleCount(uint32_t id, int64_t tupleCount);
   void registerState(const State& s) {
      states.insert({s.ptr, s});
   }
   void reset();
   ~ExecutionContext();
};
} // end namespace runtime

#endif // RUNTIME_EXECUTIONCONTEXT_H
