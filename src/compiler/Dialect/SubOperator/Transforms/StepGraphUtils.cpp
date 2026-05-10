#include "lingodb/compiler/Dialect/SubOperator/Transforms/StepGraphUtils.h"

#include <queue>

namespace lingodb::compiler::dialect::subop {

std::vector<mlir::Operation*> kahnTopoSort(
   llvm::ArrayRef<mlir::Operation*> nodes,
   const llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>>& deps,
   const llvm::DenseMap<mlir::Operation*, size_t>& priority) {
   llvm::DenseMap<mlir::Operation*, size_t> indegree;
   llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> reverse;
   for (auto* n : nodes) indegree[n] = 0;
   for (auto& [n, ds] : deps) {
      for (auto* d : ds) {
         reverse[d].insert(n);
         indegree[n]++;
      }
   }
   auto cmp = [&](mlir::Operation* a, mlir::Operation* b) {
      return priority.lookup(a) > priority.lookup(b); // min-heap
   };
   std::priority_queue<mlir::Operation*, std::vector<mlir::Operation*>, decltype(cmp)> pq(cmp);
   for (auto& [n, d] : indegree) {
      if (d == 0) pq.push(n);
   }
   std::vector<mlir::Operation*> out;
   while (!pq.empty()) {
      auto* n = pq.top();
      pq.pop();
      out.push_back(n);
      for (auto* s : reverse[n]) {
         if (--indegree[s] == 0) pq.push(s);
      }
   }
   return out;
}

} // namespace lingodb::compiler::dialect::subop
