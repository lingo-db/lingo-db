#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
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
      if (mlir::isa<subop::SubOperator>(curr)) {
         for (auto attr : curr->getAttrs()) {
            analyze(curr, attr.getValue());
         }
         mlir::Operation* subopParentOp = curr->getParentOp();
         while (subopParentOp && !mlir::isa<subop::SubOperator>(subopParentOp)) {
            subopParentOp = subopParentOp->getParentOp();
         }
         if (subopParentOp) {
            auto currUsedColumns = getCreatedColumns(curr);
            createdColumns[subopParentOp].insert(currUsedColumns.begin(), currUsedColumns.end());
         }
      }
   });
}
void subop::ColumnCreationAnalysis::update(mlir::Operation* op) {
   for (auto attr : op->getAttrs()) {
      analyze(op, attr.getValue());
   }
}
namespace {
void analyze(mlir::Operation* op, mlir::Attribute attr, std::unordered_set<tuples::Column*>& createdColumns) {
   if (!attr) return;
   if (auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
      for (auto x : arrayAttr) {
         analyze(op, x, createdColumns);
      }
   } else if (auto mappingDefAttr = mlir::dyn_cast_or_null<subop::ColumnDefMemberMappingAttr>(attr)) {
      for (auto x : mappingDefAttr.getMapping()) {
         analyze(op, x.second, createdColumns);
      }
   } else if (auto mappingRefAttr = mlir::dyn_cast_or_null<subop::ColumnRefMemberMappingAttr>(attr)) {
      for (auto x : mappingRefAttr.getMapping()) {
         analyze(op, x.second, createdColumns);
      }
   } else if (auto columnDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr)) {
      createdColumns.insert(&columnDefAttr.getColumn());
      analyze(op, columnDefAttr.getFromExisting(), createdColumns);
   }
}
} // end anonymous namespace
std::unordered_set<tuples::Column*> subop::ColumnCreationAnalysis::getCreatedColumnsForOp(mlir::Operation* op) {
   std::unordered_set<tuples::Column*> createdCols;
   op->walk([&](mlir::Operation* curr) {
      for (auto attr : curr->getAttrs()) {
         ::analyze(curr, attr.getValue(), createdCols);
      }
   });
   return createdCols;
}
