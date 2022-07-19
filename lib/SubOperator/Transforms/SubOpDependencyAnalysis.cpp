#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"

#include <queue>
mlir::subop::SubOpRootAnalysis::SubOpRootAnalysis(mlir::Operation* op) {
   op->walk([&](mlir::subop::SubOperator subop) {
      mlir::Operation* root = subop.getOperation();
      for (auto x : subop->getOperands()) {
         if (auto pred = mlir::dyn_cast_or_null<mlir::subop::SubOperator>(x.getDefiningOp())) {
            root = this->root[pred];
         }
      }
      this->root[subop] = root;
   });
}
mlir::subop::SubOpDependencyAnalysis::SubOpDependencyAnalysis(mlir::Operation* op, AnalysisManager& am) {
   SubOpRootAnalysis& rootAnalysis = am.getAnalysis<SubOpRootAnalysis>();
   std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> pipelines;
   std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> pipelineRequirements;
   std::unordered_map<mlir::Operation*, size_t> dependCount;
   std::queue<mlir::Operation*> queue;
   op->walk([&](mlir::subop::SubOperator subop) {
      auto* subopRoot = rootAnalysis.getRoot(subop);
      for (auto x : subop->getOperands()) {
         if (!x.getType().isa<mlir::tuples::TupleStreamType>()) {
            if (auto* definingOp = x.getDefiningOp()) {
               if (subopRoot->getBlock() == definingOp->getBlock()) {
                  pipelineRequirements[subopRoot].push_back(definingOp);
               }
            }
         }
      }
      for (auto readMember : subop.getReadMembers()) {
         for (auto* conflict : writtenMembers[readMember]) {
            addDependency(subopRoot, conflict);
         }
      }
      for (auto writtenMember : subop.getReadMembers()) {
         for (auto* conflict : writtenMembers[writtenMember]) {
            addDependency(subopRoot, conflict);
         }
         for (auto* conflict : readMembers[writtenMember]) {
            addDependency(subopRoot, conflict);
         }
      }
      for (auto readMember : subop.getReadMembers()) {
         readMembers[readMember].insert(subopRoot);
      }
      for (auto writtenMember : subop.getWrittenMembers()) {
         writtenMembers[writtenMember].insert(subopRoot);
      }

      pipelines[subopRoot].push_back(subop);
   });
   for (auto x : pipelines) {
      dependCount[x.first] = dependencies[x.first].size();
      if (dependCount[x.first] == 0) {
         queue.push(x.first);
      }
   }
   std::unordered_set<mlir::Operation*> availableRequirements;
   while (!queue.empty()) {
      auto* currRoot = queue.front();
      //llvm::dbgs() << "curr :" << currRoot << "\n";
      queue.pop();
      for (auto* otherRoot : inverseDependencies[currRoot]) {
         if (dependCount[otherRoot] > 0 && otherRoot != currRoot) {
            dependCount[otherRoot]--;
            if (dependCount[otherRoot] == 0) {
               queue.push(otherRoot);
            }
         }
      }
      auto& localOrdering = validOrder[currRoot->getBlock()];
      for (auto* requirement : pipelineRequirements[currRoot]) {
         if (!availableRequirements.contains(requirement)) {
            availableRequirements.insert(requirement);
            localOrdering.push_back(requirement);
         }
      }
      localOrdering.insert(localOrdering.end(), pipelines[currRoot].begin(), pipelines[currRoot].end());
   }
}