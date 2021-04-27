#ifndef MLIR_GOES_RELATIONAL_DBTYPE_H
#define MLIR_GOES_RELATIONAL_DBTYPE_H
#include "mlir/IR/Types.h"
namespace mlir::db {
class DBType : public mlir::Type {
   public:
   using Type::Type;

   bool isNullable();
   DBType getBaseType() const;
   DBType asNullable() const;
   static bool classof(Type);
};
}

#endif // MLIR_GOES_RELATIONAL_DBTYPE_H
