#ifndef MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
#define MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::db {

class CollectionType : public mlir::Type {
   public:
   using Type::Type;
   Type getElementType() const;
};
} // namespace mlir::db

#endif // MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
