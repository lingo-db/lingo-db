#ifndef MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNFOLDING_H
#define MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNFOLDING_H
#include "mlir/Dialect/TupleStream/Column.h"
namespace mlir::subop {

struct ColumnFoldInfo {
   std::unordered_map<mlir::tuples::Column*, mlir::tuples::Column*> directMappings;
};
} // namespace mlir::subop

#endif //MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNFOLDING_H
