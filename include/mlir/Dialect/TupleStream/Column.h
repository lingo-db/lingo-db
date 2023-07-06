#ifndef MLIR_DIALECT_TUPLESTREAM_COLUMN_H
#define MLIR_DIALECT_TUPLESTREAM_COLUMN_H
#include <mlir/IR/Types.h>
namespace mlir::tuples {
struct Column {
   mlir::Type type;
};
} // namespace mlir::tuples

#endif // MLIR_DIALECT_TUPLESTREAM_COLUMN_H
