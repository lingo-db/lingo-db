#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/ColumnUsageHelpers.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
using namespace lingodb::compiler::dialect;

subop::ColumnCreationAnalysis::ColumnCreationAnalysis(mlir::Operation* op) {
   op->walk([&](mlir::Operation* curr) {
      auto subOp = mlir::dyn_cast<subop::SubOperator>(curr);
      if (!subOp) return;
      llvm::DenseSet<tuples::Column*> directlyCreated;
      subOp.addCreatedColumns(directlyCreated);
      for (auto* col : directlyCreated) {
         columnCreators[col] = curr;
      }
      // Post-order walk: by now createdColumns[curr] holds the columns created
      // by all SubOp descendants. Merge curr's direct creations on top, then
      // propagate the full transitive set to curr's parent SubOp.
      auto& currSet = createdColumns[curr];
      currSet.insert(directlyCreated.begin(), directlyCreated.end());
      std::unordered_set<tuples::Column*> snapshot(currSet);
      mlir::Operation* subopParentOp = curr->getParentOp();
      while (subopParentOp && !mlir::isa<subop::SubOperator>(subopParentOp)) {
         subopParentOp = subopParentOp->getParentOp();
      }
      if (subopParentOp) {
         createdColumns[subopParentOp].insert(snapshot.begin(), snapshot.end());
      }
   });
}

void subop::ColumnCreationAnalysis::update(mlir::Operation* op) {
   auto subOp = mlir::dyn_cast<subop::SubOperator>(op);
   if (!subOp) return;
   llvm::DenseSet<tuples::Column*> cols;
   subOp.addCreatedColumns(cols);
   auto& set = createdColumns[op];
   for (auto* col : cols) {
      set.insert(col);
      columnCreators[col] = op;
   }
}

std::unordered_set<tuples::Column*> subop::ColumnCreationAnalysis::getCreatedColumnsForOp(mlir::Operation* op) {
   std::unordered_set<tuples::Column*> createdCols;
   op->walk([&](mlir::Operation* curr) {
      llvm::DenseSet<tuples::Column*> cols;
      if (auto subOp = mlir::dyn_cast<subop::SubOperator>(curr)) {
         subOp.addCreatedColumns(cols);
      } else {
         subop::collectCreatedColumnsFromAttrs(curr, cols);
      }
      for (auto* col : cols) createdCols.insert(col);
   });
   return createdCols;
}
