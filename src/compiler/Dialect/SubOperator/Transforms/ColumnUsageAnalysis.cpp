#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
using namespace lingodb::compiler::dialect;

void subop::ColumnUsageAnalysis::analyze(mlir::Operation* op, mlir::Attribute attr) {
   if (!attr) return;
   if (auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
      for (auto x : arrayAttr) {
         analyze(op, x);
      }
   } else if (auto dictionaryAttr = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
      for (auto x : dictionaryAttr) {
         analyze(op, x.getValue());
      }
   } else if (auto columnRefAttr = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(attr)) {
      usedColumns[op].insert(&columnRefAttr.getColumn());
      operationsUsingColumn[&columnRefAttr.getColumn()].insert(op);

   } else if (auto columnDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr)) {
      analyze(op, columnDefAttr.getFromExisting());
   }
}
subop::ColumnUsageAnalysis::ColumnUsageAnalysis(mlir::Operation* op) {
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