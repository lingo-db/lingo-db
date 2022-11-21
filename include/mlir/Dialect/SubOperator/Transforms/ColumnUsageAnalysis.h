#ifndef MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
#define MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
#include "mlir/Dialect/TupleStream/Column.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"

#include <unordered_set>
namespace mlir::subop {

struct ColumnUsageAnalysis {
   std::unordered_map<mlir::Operation*, std::unordered_set<mlir::tuples::Column*>> usedColumns;
   ColumnUsageAnalysis(mlir::Operation* op);
   void analyze(mlir::Operation* op, mlir::Attribute attr);
   const std::unordered_set<mlir::tuples::Column*>& getUsedColumns(mlir::Operation* op) {
      return usedColumns[op];
   }
};
} // namespace mlir::subop
#endif // MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
