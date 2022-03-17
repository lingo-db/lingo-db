#ifndef MLIR_DIALECT_RELALG_IR_COLUMN_H
#define MLIR_DIALECT_RELALG_IR_COLUMN_H
#include <mlir/IR/Types.h>
namespace mlir::relalg {
struct Column {
  mlir::Type type;
};
} // namespace mlir::relalg

#endif // MLIR_DIALECT_RELALG_IR_COLUMN_H
