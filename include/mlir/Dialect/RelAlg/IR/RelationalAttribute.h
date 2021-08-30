#ifndef MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTE_H
#define MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTE_H
#include "mlir/Dialect/DB/IR/DBType.h"

namespace mlir::relalg {
struct RelationalAttribute {
  mlir::Type type;
};
} // namespace mlir::relalg

#endif // MLIR_DIALECT_RELALG_IR_RELATIONALATTRIBUTE_H
