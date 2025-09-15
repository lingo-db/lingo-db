#pragma once
#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/MLIRTypes.h"

#include <lingodb/catalog/Types.h>
#include <lingodb/catalog/FunctionCatalogEntry.h>
namespace lingodb::catalog {
class MLIRUDFImplementor {
   public:
   virtual mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) = 0;
   virtual ~MLIRUDFImplementor() = default;
};
} // namespace lingodb::catalog

namespace lingodb::compiler::frontend {
std::shared_ptr<catalog::MLIRUDFImplementor> getUDFImplementer(std::shared_ptr<catalog::FunctionCatalogEntry> entry);
std::shared_ptr<catalog::MLIRUDFImplementor> createCUDFImplementer(
   std::string funcName, std::string cCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType);

} // namespace lingodb::compiler::frontend