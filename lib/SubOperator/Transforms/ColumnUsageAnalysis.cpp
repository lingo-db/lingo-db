#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
void mlir::subop::ColumnUsageAnalysis::analyze(mlir::Operation* op, mlir::Attribute attr) {
   if (!attr) return;
   if (auto arrayAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
      for (auto x : arrayAttr) {
         analyze(op, x);
      }
   } else if (auto dictionaryAttr = attr.dyn_cast_or_null<mlir::DictionaryAttr>()) {
      for (auto x : dictionaryAttr) {
         analyze(op, x.getValue());
      }
   } else if (auto columnRefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>()) {
      usedColumns[op].insert(&columnRefAttr.getColumn());
   } else if (auto columnDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>()) {
      analyze(op, columnDefAttr.getFromExisting());
   }
}
mlir::subop::ColumnUsageAnalysis::ColumnUsageAnalysis(mlir::Operation* op) {
   op->walk([&](mlir::Operation* curr) {
      for (auto attr : curr->getAttrs()) {
         analyze(curr, attr.getValue());
      }
      if (auto *parentOp = curr->getParentOp()) {
         auto& currUsedColumns = getUsedColumns(curr);
         usedColumns[parentOp].insert(currUsedColumns.begin(), currUsedColumns.end());
      }
   });
}