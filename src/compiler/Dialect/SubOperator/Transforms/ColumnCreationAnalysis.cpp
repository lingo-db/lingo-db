#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

using namespace lingodb::compiler::dialect;
void subop::ColumnCreationAnalysis::analyze(mlir::Operation* op, mlir::Attribute attr) {
   if (!attr) return;
   if (auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
      for (auto x : arrayAttr) {
         analyze(op, x);
      }
   } else if (auto mappingDefAttr = mlir::dyn_cast_or_null<subop::ColumnDefMemberMappingAttr>(attr)) {
      for (auto x : mappingDefAttr.getMapping()) {
         analyze(op, x.second);
      }
   } else if (auto mappingRefAttr = mlir::dyn_cast_or_null<subop::ColumnRefMemberMappingAttr>(attr)) {
      for (auto x : mappingRefAttr.getMapping()) {
         analyze(op, x.second);
      }
   } else if (auto columnDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr)) {
      createdColumns[op].insert(&columnDefAttr.getColumn());
      columnCreators[&columnDefAttr.getColumn()] = op;
      analyze(op, columnDefAttr.getFromExisting());
   }
}
subop::ColumnCreationAnalysis::ColumnCreationAnalysis(mlir::Operation* op) {
   op->walk([&](mlir::Operation* curr) {
      for (auto attr : curr->getAttrs()) {
         analyze(curr, attr.getValue());
      }
      if (auto* parentOp = curr->getParentOp()) {
         auto& currUsedColumns = getCreatedColumns(curr);
         createdColumns[parentOp].insert(currUsedColumns.begin(), currUsedColumns.end());
      }
   });
}
void subop::ColumnCreationAnalysis::update(mlir::Operation* op) {
   for (auto attr : op->getAttrs()) {
      analyze(op, attr.getValue());
   }
}