#ifndef MLIR_DIALECT_DB_IR_DBTYPE_H
#define MLIR_DIALECT_DB_IR_DBTYPE_H
#include "mlir/IR/Types.h"
namespace mlir::db {
class DBType : public mlir::Type {
   public:
   using Type::Type;

   bool isNullable();
   DBType getBaseType() const;
   DBType asNullable() const;
   bool isVarLen() const;
   static bool classof(Type);
};
} // namespace mlir::db

#endif // MLIR_DIALECT_DB_IR_DBTYPE_H
