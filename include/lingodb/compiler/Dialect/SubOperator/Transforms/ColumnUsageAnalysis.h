#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"

#include <unordered_set>
namespace lingodb::compiler::dialect::subop {

struct ColumnUsageAnalysis {
   const llvm::SmallVector<mlir::Operation*> emptyOpVector;
   llvm::DenseMap<mlir::Operation*, llvm::DenseSet<dialect::tuples::Column*>> usedColumns;
   llvm::DenseMap<dialect::tuples::Column*, llvm::SmallVector<mlir::Operation*>> operationsUsingColumn;
   ColumnUsageAnalysis(mlir::Operation* op);
   void analyze(mlir::Operation* op, mlir::Attribute attr);
   auto& getUsedColumns(mlir::Operation* op) {
      return usedColumns[op];
   }
   const llvm::SmallVector<mlir::Operation*>& findOperationsUsing(dialect::tuples::Column* column) const {
      if (operationsUsingColumn.contains(column)) {
         return operationsUsingColumn.at(column);
      }
      return emptyOpVector;
   }
};
} // namespace lingodb::compiler::dialect::subop
#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNUSAGEANALYSIS_H
