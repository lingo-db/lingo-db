#ifndef MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_SUBOPDEPENDENCYANALYSIS_H
#define MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_SUBOPDEPENDENCYANALYSIS_H
#include "mlir/Pass/AnalysisManager.h"
#include <mlir/IR/Operation.h>

#include <unordered_map>
#include <unordered_set>
namespace mlir::subop {

struct SubOpRootAnalysis {
   std::unordered_map<mlir::Operation*, mlir::Operation*> root;
   SubOpRootAnalysis(mlir::Operation* op);
   mlir::Operation* getRoot(mlir::Operation* op) {
      return root[op];
   }
};
struct SubOpDependencyAnalysis {
   std::unordered_map<mlir::Operation*, std::unordered_set<mlir::Operation*>> dependencies;
   std::unordered_map<mlir::Operation*, std::unordered_set<mlir::Operation*>> inverseDependencies;
   std::unordered_map<std::string, std::unordered_set<mlir::Operation*>> readMembers;
   std::unordered_map<std::string, std::unordered_set<mlir::Operation*>> writtenMembers;
   std::unordered_map<mlir::Block*, std::vector<mlir::Operation*>> validOrder;
   SubOpDependencyAnalysis(mlir::Operation* op, AnalysisManager& am);
   void addDependency(mlir::Operation* a, mlir::Operation* b) {
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
} // namespace mlir::subop

#endif //MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_SUBOPDEPENDENCYANALYSIS_H
