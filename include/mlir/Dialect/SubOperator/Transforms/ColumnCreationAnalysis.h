#ifndef MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNCREATIONANALYSIS_H
#define MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNCREATIONANALYSIS_H
#include "mlir/Dialect/TupleStream/Column.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"

#include <unordered_set>
namespace mlir::subop {

struct ColumnCreationAnalysis {
   std::unordered_map<mlir::Operation*, std::unordered_set<mlir::tuples::Column*>> createdColumns;
   ColumnCreationAnalysis(mlir::Operation* op);
   void analyze(mlir::Operation* op, mlir::Attribute attr);
   const std::unordered_set<mlir::tuples::Column*>& getCreatedColumns(mlir::Operation* op) {
      return createdColumns[op];
   }
};
} // namespace mlir::subop
#endif // MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNCREATIONANALYSIS_H
