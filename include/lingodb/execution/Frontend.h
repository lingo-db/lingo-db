#ifndef LINGODB_EXECUTION_FRONTEND_H
#define LINGODB_EXECUTION_FRONTEND_H
#include "Error.h"
#include "lingodb/catalog/Catalog.h"
#include "lingodb/catalog/Types.h"

#include <memory>
#include <optional>
#include <vector>
namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir
namespace lingodb {
class NullableType;
} // namespace lingodb
namespace lingodb::ast {
class Value;
} // namespace lingodb::ast
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
   virtual void setContext(mlir::MLIRContext* context) = 0;
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   virtual void loadFromFile(std::string fileName) = 0;
   virtual void loadFromString(std::string data) = 0;
   /// Bind values for `?` placeholders in the SQL. Values are 1-indexed as
   /// they appear in the text. Default is a no-op (e.g. MLIR frontend has no
   /// placeholders). Must be called before `loadFromString` / `loadFromFile`.
   virtual void setParameters(std::vector<std::shared_ptr<ast::Value>> /*values*/) {}
   /// Types the analyzer inferred for each `?` placeholder after the most
   /// recent `loadFromString` / `loadFromFile`. Index matches the placeholder
   /// order in the SQL (1-based = entry 0). Entry may be `std::nullopt` if
   /// the placeholder was unused or its type could not be inferred.
   virtual std::vector<std::optional<catalog::Type>> getInferredParameterTypes() const { return {}; }
   virtual bool isParallelismAllowed() { return true; }
   virtual mlir::ModuleOp* getModule() = 0;
   virtual ~Frontend() {}
};
std::unique_ptr<Frontend> createMLIRFrontend();
std::unique_ptr<Frontend> createSQLFrontend();
void initializeContext(mlir::MLIRContext& context, bool includeLLVM = true);

} //namespace lingodb::execution

#endif //LINGODB_EXECUTION_FRONTEND_H
