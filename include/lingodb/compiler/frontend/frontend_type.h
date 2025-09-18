#pragma once
#include "lingodb/catalog/TableCatalogEntry.h"

#include <mlir/IR/MLIRContext.h>
#include "lingodb/catalog/MLIRTypes.h"
namespace lingodb {
class NullableType {
   public:
   NullableType(catalog::Type type);
   NullableType(catalog::Type type, bool isNullable);
   catalog::Type type;
   std::shared_ptr<NullableType> castType = nullptr;
   bool isNullable;
   bool useZeroInsteadOfNull = false;
   mlir::Type toMlirType(mlir::MLIRContext* context) const;
   mlir::Value castValueToThisType(mlir::OpBuilder& builder, mlir::Value valueToCast, bool valueNullable) const;
   mlir::Value castValue(mlir::OpBuilder& builder, mlir::Value valueToCast) const;
   bool isNumeric() const;

   bool operator==(NullableType&);
   bool operator!=(NullableType&);
};
}