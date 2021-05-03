#ifndef MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
#define MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
#include "mlir/IR/Types.h"
namespace mlir::db {
class CollectionType : public mlir::Type {
   public:
   using Type::Type;

   bool isImplicit();
   bool hasSize();
   bool hasMaxSize();
   bool canReadMultipleTimes();
   bool offersRandomAccess();
   bool offersParallelIteration();
   bool isOrdered();
};
} // namespace mlir::db

#endif // MLIR_DIALECT_DB_IR_DBCOLLECTIONTYPE_H
