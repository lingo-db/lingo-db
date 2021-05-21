#ifndef MLIR_DIALECT_DB_IR_DBBUILDERTYPE_H
#define MLIR_DIALECT_DB_IR_DBBUILDERTYPE_H
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::db {

class BuilderType : public mlir::Type {
   public:
   using Type::Type;
   Type getInsertableType() const;
   Type getBuildResultType();
};
} // namespace mlir::db

#endif // MLIR_DIALECT_DB_IR_DBBUILDERTYPE_H
