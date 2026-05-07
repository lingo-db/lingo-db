#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/ColumnUsageHelpers.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
using namespace lingodb::compiler::dialect;

subop::ColumnUsageAnalysis::ColumnUsageAnalysis(mlir::Operation* op) {
   op->walk([&](mlir::Operation* curr) {
      auto subOp = mlir::dyn_cast<subop::SubOperator>(curr);
      if (!subOp) return;
      llvm::DenseSet<tuples::Column*> directlyUsed;
      subOp.addUsedColumns(directlyUsed);
      for (auto* col : directlyUsed) {
         operationsUsingColumn[col].push_back(curr);
      }
      // Post-order walk: children of curr have already merged their transitive
      // columns into usedColumns[curr] before we visit curr. Combining those
      // with curr's directly-used columns gives the full transitive set, which
      // we then propagate to curr's parent SubOperator.
      llvm::DenseSet<tuples::Column*> transitive(directlyUsed);
      auto it = usedColumns.find(curr);
      if (it != usedColumns.end()) {
         transitive.insert(it->second.begin(), it->second.end());
      }
      usedColumns[curr] = transitive;
      mlir::Operation* subopParentOp = curr->getParentOp();
      while (subopParentOp && !mlir::isa<subop::SubOperator>(subopParentOp)) {
         subopParentOp = subopParentOp->getParentOp();
      }
      if (subopParentOp) {
         usedColumns[subopParentOp].insert(transitive.begin(), transitive.end());
      }
   });
}

llvm::DenseSet<tuples::Column*> subop::ColumnUsageAnalysis::getUsedColumnsForOp(mlir::Operation* op) {
   llvm::DenseSet<tuples::Column*> usedcolumns;
   op->walk([&](mlir::Operation* curr) {
      if (auto subOp = mlir::dyn_cast<subop::SubOperator>(curr)) {
         subOp.addUsedColumns(usedcolumns);
      } else {
         // Non-SubOp operations inside a SubOp body (e.g. tuples.getcolumn in
         // lambda regions) can still carry ColumnRefAttrs. The default helper
         // covers those via plain attribute walk.
         subop::collectUsedColumnsFromAttrs(curr, usedcolumns);
      }
   });
   return usedcolumns;
}
