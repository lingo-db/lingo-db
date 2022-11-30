#ifndef LINGODB_COLUMNFOLDING_H
#define LINGODB_COLUMNFOLDING_H
#include "mlir/Dialect/TupleStream/Column.h"
namespace mlir::relalg {

struct ColumnFoldInfo {
   std::unordered_map<mlir::tuples::Column*, mlir::tuples::Column*> directMappings;
};
} // namespace mlir::relalg

#endif //LINGODB_COLUMNFOLDING_H
