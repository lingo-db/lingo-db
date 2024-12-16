#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"

#include <unordered_set>
namespace lingodb::compiler::dialect::subop {

struct ColumnUsageAnalysis {
   std::unordered_map<mlir::Operation*, std::unordered_set<dialect::tuples::Column*>> usedColumns;
   std::unordered_map<dialect::tuples::Column*, std::unordered_set<mlir::Operation*>> operationsUsingColumn;
   ColumnUsageAnalysis(mlir::Operation* op);
   void analyze(mlir::Operation* op, mlir::Attribute attr);
   const std::unordered_set<dialect::tuples::Column*>& getUsedColumns(mlir::Operation* op) {
      return usedColumns[op];
   }
   const std::unordered_set<mlir::Operation*> findOperationsUsing(dialect::tuples::Column* column) const {
      if (operationsUsingColumn.contains(column)) {
         return operationsUsingColumn.at(column);
      }
      return {};
   }
};
} // namespace lingodb::compiler::dialect::subop
#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
