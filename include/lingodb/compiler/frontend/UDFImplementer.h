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

// Lowering hook for tabular UDFs. The translator handles the SQL scaffolding
// around the call (the relalg.nested wrapper, arrow.table ↔ subop.local_table
// bridges, scalar-arg expression translation); the implementor is responsible
// only for the language-specific bridge: take an `arrow.table` SSA value
// carrying the input rows plus already-cast scalar args, emit whatever ops
// invoke the UDF, and return an `arrow.table` SSA value carrying the output.
class MLIRTableUDFImplementor {
   public:
   virtual mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc,
                                    mlir::Value inputArrowTable, mlir::ValueRange scalarArgs,
                                    lingodb::catalog::Catalog* catalog) = 0;
   virtual ~MLIRTableUDFImplementor() = default;
};
} // namespace lingodb::catalog

namespace lingodb::compiler::frontend {
std::shared_ptr<catalog::MLIRUDFImplementor> getUDFImplementer(std::shared_ptr<catalog::FunctionCatalogEntry> entry);
std::shared_ptr<catalog::MLIRUDFImplementor> createCUDFImplementer(
   std::string funcName, std::string cCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType);
std::shared_ptr<catalog::MLIRUDFImplementor> createPythonUDFImplementer(
   std::string funcName, std::string pyCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType);

std::shared_ptr<catalog::MLIRTableUDFImplementor> getTableUDFImplementer(std::shared_ptr<catalog::TableFunctionCatalogEntry> entry);
std::shared_ptr<catalog::MLIRTableUDFImplementor> createPythonTableUDFImplementer(
   std::string funcName, std::string pyCode, std::vector<lingodb::catalog::Type> scalarArgumentTypes);

} // namespace lingodb::compiler::frontend
#endif
