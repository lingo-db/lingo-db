#ifndef RUNTIME_EXECUTIONCONTEXT_H
#define RUNTIME_EXECUTIONCONTEXT_H
#include "Database.h"
namespace runtime {
class ExecutionContext {
   std::unordered_map<uint32_t, uint8_t*> results;

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
   void setResult(uint32_t, uint8_t* ptr);
};
} // end namespace runtime

#endif // RUNTIME_EXECUTIONCONTEXT_H
