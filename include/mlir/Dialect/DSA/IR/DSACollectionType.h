#ifndef MLIR_DIALECT_DSA_IR_DSACOLLECTIONTYPE_H
#define MLIR_DIALECT_DSA_IR_DSACOLLECTIONTYPE_H
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::dsa {

class CollectionType : public mlir::Type {
   public:
   using Type::Type;
   Type getElementType() const;
   static bool classof(Type);
};
} // namespace mlir::dsa

#endif // MLIR_DIALECT_DSA_IR_DSACOLLECTIONTYPE_H
