#ifndef LINGODB_EXECUTION_FRONTEND_H
#define LINGODB_EXECUTION_FRONTEND_H
#include "Error.h"
#include "lingodb/catalog/Catalog.h"
namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir
namespace lingodb::execution {
class Frontend {
   protected:
   catalog::Catalog* catalog;
   Error error;

   std::unordered_map<std::string, double> timing;

   public:
   catalog::Catalog* getCatalog() const {
      return catalog;
   }
   void setCatalog(catalog::Catalog* catalog) {
      Frontend::catalog = catalog;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   virtual void loadFromFile(std::string fileName) = 0;
   virtual void loadFromString(std::string data) = 0;
   virtual bool isParallelismAllowed() { return true; }
   virtual mlir::ModuleOp* getModule() = 0;
   virtual ~Frontend() {}
};
std::unique_ptr<Frontend> createMLIRFrontend();
std::unique_ptr<Frontend> createSQLFrontend();
std::unique_ptr<Frontend> createNewSQLFrontend();
void initializeContext(mlir::MLIRContext& context);

} //namespace lingodb::execution

#endif //LINGODB_EXECUTION_FRONTEND_H