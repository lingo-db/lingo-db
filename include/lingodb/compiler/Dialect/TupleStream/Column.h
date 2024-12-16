#ifndef LINGODB_COMPILER_DIALECT_TUPLESTREAM_COLUMN_H
#define LINGODB_COMPILER_DIALECT_TUPLESTREAM_COLUMN_H
#include <mlir/IR/Types.h>
namespace lingodb::compiler::dialect::tuples {
struct Column {
   mlir::Type type;
};
} // namespace lingodb::compiler::dialect::tuples

#endif //LINGODB_COMPILER_DIALECT_TUPLESTREAM_COLUMN_H
