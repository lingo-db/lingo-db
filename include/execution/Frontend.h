#ifndef EXECUTION_FRONTEND_H
#define EXECUTION_FRONTEND_H
#include "Error.h"
#include "runtime/Database.h"
namespace mlir {
class ModuleOp;
} // namespace mlir
namespace execution {
class Frontend {
   protected:
   runtime::Database* database;
   Error error;

   std::unordered_map<std::string, double> timing;

   public:
   runtime::Database* getDatabase() const {
      return database;
   }
   void setDatabase(runtime::Database* db) {
      Frontend::database = db;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   virtual void loadFromFile(std::string fileName) = 0;
   virtual void loadFromString(std::string data) = 0;
   virtual mlir::ModuleOp* getModule() = 0;
   virtual ~Frontend() {}
};
std::unique_ptr<Frontend> createMLIRFrontend();
std::unique_ptr<Frontend> createSQLFrontend();

} //namespace execution

#endif //EXECUTION_FRONTEND_H
