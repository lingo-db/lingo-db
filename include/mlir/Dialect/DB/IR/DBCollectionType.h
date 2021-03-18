#ifndef MLIR_GOES_RELATIONAL_DBCOLLECTIONTYPE_H
#define MLIR_GOES_RELATIONAL_DBCOLLECTIONTYPE_H
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
}

#endif // MLIR_GOES_RELATIONAL_DBCOLLECTIONTYPE_H
