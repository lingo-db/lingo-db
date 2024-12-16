#include "llvm/Support/Debug.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"

#include <queue>

using namespace lingodb::compiler::dialect;

subop::SubOpRootAnalysis::SubOpRootAnalysis(mlir::Operation* op) {
   op->walk([&](subop::SubOperator subop) {
      for (auto x : subop->getOperands()) {
         if (auto pred = mlir::dyn_cast_or_null<subop::SubOperator>(x.getDefiningOp())) {
            if (mlir::isa<tuples::TupleStreamType>(x.getType())) {
               this->roots[subop].insert(this->roots[subop].end(), this->roots[pred].begin(), this->roots[pred].end());
            }
         }
      }
      if (this->roots[subop].empty()) {
         this->roots[subop].push_back(subop.getOperation());
      }
   });
}
bool subop::SubOpDependencyAnalysis::isDependentOn(mlir::Operation* curr, mlir::Operation* other) {
   for (auto* d : getDependenciesOf(curr)) {
      if (d == other) return true;
      if (isDependentOn(d, other)) {
         return true;
      }
   }
   return false;
}
bool subop::SubOpDependencyAnalysis::areIndependent(mlir::Operation* op, mlir::Operation* op2) {
   return !isDependentOn(op, op2) && !isDependentOn(op2, op);
}
void subop::SubOpDependencyAnalysis::addToRoot(mlir::Operation* root, mlir::Operation* previousRoot) {
   for (auto* dep : dependencies[previousRoot]) {
      addDependency(root, dep, {});
   }
   for (auto* dep : inverseDependencies[previousRoot]) {
      addDependency(dep, root, {});
   }
}
subop::SubOpDependencyAnalysis::SubOpDependencyAnalysis(mlir::Operation* op, mlir::AnalysisManager& am) {
   SubOpRootAnalysis& rootAnalysis = am.getAnalysis<SubOpRootAnalysis>();
   std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> pipelines;
   std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> pipelineRequirements;
   std::unordered_map<mlir::Operation*, size_t> dependCount;
   std::queue<mlir::Operation*> queue;
   op->walk([&](subop::SubOperator subop) {
      auto roots = rootAnalysis.getRoots(subop);
      for (auto* subopRoot : roots) {
         for (auto x : subop->getOperands()) {
            if (!mlir::isa<tuples::TupleStreamType>(x.getType())) {
               if (auto* definingOp = x.getDefiningOp()) {
                  if (mlir::dyn_cast_or_null<subop::SubOperator>(definingOp)) {
                     addDependency(subopRoot, definingOp, roots);
                  } else {
                     if (subopRoot->getBlock() == definingOp->getBlock()) {
                        pipelineRequirements[subopRoot].push_back(definingOp);
                     }
                  }
               }
            }
         }
         for (auto& region : subop->getRegions()) {
            for (auto& op : region.getOps()) {
               for (auto operand : op.getOperands()) {
                  if (operand.getParentRegion() == subopRoot->getParentRegion()) {
                     if (auto* definingOp = operand.getDefiningOp()) {
                        pipelineRequirements[subopRoot].push_back(definingOp);
                     }
                  }
               }
            }
         }
         for (auto readMember : subop.getReadMembers()) {
            for (auto* conflict : writtenMembers[readMember]) {
               addDependency(subopRoot, conflict, roots);
            }
         }
         for (auto writtenMember : subop.getWrittenMembers()) {
            for (auto* conflict : writtenMembers[writtenMember]) {
               addDependency(subopRoot, conflict, roots);
            }
            for (auto* conflict : readMembers[writtenMember]) {
               addDependency(subopRoot, conflict, roots);
            }
         }
         for (auto readMember : subop.getReadMembers()) {
            readMembers[readMember].insert(subopRoot);
         }
         for (auto writtenMember : subop.getWrittenMembers()) {
            writtenMembers[writtenMember].insert(subopRoot);
         }

         pipelines[subopRoot].push_back(subop);
      }
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
      availableRequirements.insert(currRoot);
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
   for (auto [root, c] : dependCount) {
      if (c != 0) {
         root->dump();
         llvm::dbgs() << "dependencies:\n";
         for (auto* dep : dependencies[root]) {
            if (dependCount[dep] > 0) {
               dep->dump();
            }
         }
         llvm::dbgs() << "-----------------------------------------------\n";
      }
   }
   for (auto [root, c] : dependCount) {
      if (c != 0) {
         assert(false && "could not find suitable order of sub-operators");
      }
   }
}