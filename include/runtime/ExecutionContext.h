#ifndef RUNTIME_EXECUTIONCONTEXT_H
#define RUNTIME_EXECUTIONCONTEXT_H
#include "Database.h"
namespace runtime {
class ExecutionContext {
   std::unordered_map<uint32_t, uint8_t*> results;
   std::unordered_map<uint32_t, int64_t> tupleCounts;

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
};
} // end namespace runtime

#endif // RUNTIME_EXECUTIONCONTEXT_H
