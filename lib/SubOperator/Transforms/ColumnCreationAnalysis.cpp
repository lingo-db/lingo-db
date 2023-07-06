#include "mlir/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
void mlir::subop::ColumnCreationAnalysis::analyze(mlir::Operation* op, mlir::Attribute attr) {
   if (!attr) return;
   if (auto arrayAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
      for (auto x : arrayAttr) {
         analyze(op, x);
      }
   } else if (auto dictionaryAttr = attr.dyn_cast_or_null<mlir::DictionaryAttr>()) {
      for (auto x : dictionaryAttr) {
         analyze(op, x.getValue());
      }
   } else if (auto columnDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>()) {
      createdColumns[op].insert(&columnDefAttr.getColumn());
      analyze(op, columnDefAttr.getFromExisting());
   }
}
mlir::subop::ColumnCreationAnalysis::ColumnCreationAnalysis(mlir::Operation* op) {
   op->walk([&](mlir::Operation* curr) {
      for (auto attr : curr->getAttrs()) {
         analyze(curr, attr.getValue());
      }
      if (auto *parentOp = curr->getParentOp()) {
         auto& currUsedColumns = getCreatedColumns(curr);
         createdColumns[parentOp].insert(currUsedColumns.begin(), currUsedColumns.end());
      }
   });
}