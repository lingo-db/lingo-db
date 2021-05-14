#ifndef MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
#define MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::db {
class CollectionIterationImpl {
   public:
   virtual std::vector<Value> implementLoop(mlir::TypeRange iterArgTypes,  mlir::TypeConverter& typeConverter,OpBuilder builder, mlir::ModuleOp parentModule, std::function<std::vector<Value>(ValueRange,OpBuilder)> bodyBuilder) = 0;
   virtual ~CollectionIterationImpl(){

   }
};
class CollectionType : public mlir::Type {
   public:
   using Type::Type;
   Type getElementType() const;
   std::unique_ptr<CollectionIterationImpl> getIterationImpl(Value collection) const;
};
} // namespace mlir::db

#endif // MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
