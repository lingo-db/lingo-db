#ifndef MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTE_H
#define MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTE_H
#include "mlir/Dialect/DB/IR/DBType.h"

namespace mlir::relalg {
struct RelationalAttribute {
  mlir::db::DBType type;
};
} // namespace mlir::relalg

#endif // MLIR_GOES_RELATIONAL_RELATIONALATTRIBUTE_H
