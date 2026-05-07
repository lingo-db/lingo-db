#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"

#include "lingodb/compiler/Dialect/SubOperator/ColumnUsageHelpers.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsInterfaces.cpp.inc"

namespace lingodb::compiler::dialect::subop {
namespace {

void collectUsed(mlir::Attribute attr, llvm::DenseSet<tuples::Column*>& out) {
   if (!attr) return;
   if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      for (auto x : arrayAttr) collectUsed(x, out);
   } else if (auto mappingDefAttr = mlir::dyn_cast<subop::ColumnDefMemberMappingAttr>(attr)) {
      for (auto x : mappingDefAttr.getMapping()) collectUsed(x.second, out);
   } else if (auto mappingRefAttr = mlir::dyn_cast<subop::ColumnRefMemberMappingAttr>(attr)) {
      for (auto x : mappingRefAttr.getMapping()) collectUsed(x.second, out);
   } else if (auto refAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(attr)) {
      out.insert(&refAttr.getColumn());
   } else if (auto defAttr = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
      collectUsed(defAttr.getFromExisting(), out);
   }
}

void collectCreated(mlir::Attribute attr, llvm::DenseSet<tuples::Column*>& out) {
   if (!attr) return;
   if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      for (auto x : arrayAttr) collectCreated(x, out);
   } else if (auto mappingDefAttr = mlir::dyn_cast<subop::ColumnDefMemberMappingAttr>(attr)) {
      for (auto x : mappingDefAttr.getMapping()) collectCreated(x.second, out);
   } else if (auto mappingRefAttr = mlir::dyn_cast<subop::ColumnRefMemberMappingAttr>(attr)) {
      for (auto x : mappingRefAttr.getMapping()) collectCreated(x.second, out);
   } else if (auto defAttr = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
      out.insert(&defAttr.getColumn());
      collectCreated(defAttr.getFromExisting(), out);
   }
}

} // namespace

void collectUsedColumnsFromAttrs(mlir::Operation* op,
                                 llvm::DenseSet<tuples::Column*>& out) {
   for (auto namedAttr : op->getAttrs()) {
      collectUsed(namedAttr.getValue(), out);
   }
   // Property-backed column refs are not visible via getAttrs(); pick them
   // up explicitly for the ops we know about.
   if (auto getColumnOp = mlir::dyn_cast<tuples::GetColumnOp>(op)) {
      if (auto* col = getColumnOp.getAttr()) out.insert(col);
   }
}

void collectCreatedColumnsFromAttrs(mlir::Operation* op,
                                    llvm::DenseSet<tuples::Column*>& out) {
   for (auto namedAttr : op->getAttrs()) {
      collectCreated(namedAttr.getValue(), out);
   }
}

} // namespace lingodb::compiler::dialect::subop
