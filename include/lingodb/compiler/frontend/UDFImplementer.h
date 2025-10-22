#ifndef LINGODB_COMPILER_FRONTEND_UDFIMPLEMENTER_H
#define LINGODB_COMPILER_FRONTEND_UDFIMPLEMENTER_H

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/MLIRTypes.h"

#include <lingodb/catalog/FunctionCatalogEntry.h>
#include <lingodb/catalog/Types.h>
namespace lingodb::catalog {
class MLIRUDFImplementor {
   public:
   virtual mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args, lingodb::catalog::Catalog* catalog) = 0;
   virtual ~MLIRUDFImplementor() = default;
};
} // namespace lingodb::catalog

namespace lingodb::compiler::frontend {
std::shared_ptr<catalog::MLIRUDFImplementor> getUDFImplementer(std::shared_ptr<catalog::FunctionCatalogEntry> entry);
std::shared_ptr<catalog::MLIRUDFImplementor> createCUDFImplementer(
   std::string funcName, std::string cCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType);

} // namespace lingodb::compiler::frontend
#endif
