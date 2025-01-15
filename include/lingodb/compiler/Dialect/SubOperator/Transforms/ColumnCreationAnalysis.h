#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNCREATIONANALYSIS_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNCREATIONANALYSIS_H
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"

#include <unordered_set>
namespace lingodb::compiler::dialect::subop {

struct ColumnCreationAnalysis {
   std::unordered_map<mlir::Operation*, std::unordered_set<dialect::tuples::Column*>> createdColumns;

   std::unordered_map<dialect::tuples::Column*, mlir::Operation*> columnCreators;
   ColumnCreationAnalysis(mlir::Operation* op);
   void analyze(mlir::Operation* op, mlir::Attribute attr);
   const std::unordered_set<dialect::tuples::Column*>& getCreatedColumns(mlir::Operation* op) {
      return createdColumns[op];
   }

   mlir::Operation* getColumnCreator(dialect::tuples::Column* column) {
      assert(columnCreators.count(column) && "Column not created by any operation");
      return columnCreators[column];
   }
   void update(mlir::Operation* op);
};
} // namespace lingodb::compiler::dialect::subop
#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_COLUMNCREATIONANALYSIS_H
