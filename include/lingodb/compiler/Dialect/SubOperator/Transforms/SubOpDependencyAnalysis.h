#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_SUBOPDEPENDENCYANALYSIS_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_SUBOPDEPENDENCYANALYSIS_H
#include "mlir/Pass/AnalysisManager.h"
#include <mlir/IR/Operation.h>

#include <unordered_map>
#include <unordered_set>
namespace lingodb::compiler::dialect::subop {

struct SubOpRootAnalysis {
   std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>> roots;
   SubOpRootAnalysis(mlir::Operation* op);
   const std::vector<mlir::Operation*>& getRoots(mlir::Operation* op) {
      return roots[op];
   }
};
struct SubOpDependencyAnalysis {
   std::unordered_map<mlir::Operation*, std::unordered_set<mlir::Operation*>> dependencies;
   std::unordered_map<mlir::Operation*, std::unordered_set<mlir::Operation*>> inverseDependencies;
   std::unordered_map<std::string, std::unordered_set<mlir::Operation*>> readMembers;
   std::unordered_map<std::string, std::unordered_set<mlir::Operation*>> writtenMembers;
   std::unordered_map<mlir::Block*, std::vector<mlir::Operation*>> validOrder;
   void addNonTupleStreamDependencies(mlir::Value x, std::vector<mlir::Operation*>& roots, mlir::Operation* subopRoot, std::unordered_map<mlir::Operation*, std::vector<mlir::Operation*>>& pipelineRequirements);
   SubOpDependencyAnalysis(mlir::Operation* op, mlir::AnalysisManager& am);
   bool isDependentOn(mlir::Operation* curr, mlir::Operation* other);
   bool areIndependent(mlir::Operation* op, mlir::Operation* op2);
   void addToRoot(mlir::Operation* root, mlir::Operation* previousRoot);
   void addDependency(mlir::Operation* a, mlir::Operation* b, std::vector<mlir::Operation*> exclude) {
      if (a == b) return;
      if (a->getBlock() != b->getBlock()) return; //todo: recheck
      if (std::find(exclude.begin(), exclude.end(), b) != exclude.end()) return;
      dependencies[a].insert(b);
      inverseDependencies[b].insert(a);
   }
   const std::unordered_set<mlir::Operation*>& getDependenciesOf(mlir::Operation* op) {
      return dependencies[op];
   }
   const std::vector<mlir::Operation*>& getValidOrder(mlir::Block* block) {
      return validOrder[block];
   }
};
} // namespace lingodb::compiler::dialect::subop

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_SUBOPDEPENDENCYANALYSIS_H
