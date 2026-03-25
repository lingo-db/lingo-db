#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include <functional>

namespace {
using namespace lingodb::compiler::dialect;

class EnforceOrderPass : public mlir::PassWrapper<EnforceOrderPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnforceOrderPass)
   virtual llvm::StringRef getArgument() const override { return "subop-enforce-order"; }

   void runOnOperation() override {
      auto subOpDependencyAnalysis = getAnalysis<subop::SubOpDependencyAnalysis>();

      for (auto& localOrder : subOpDependencyAnalysis.validOrder) {
         if (localOrder.second.empty()) continue;

         mlir::Block* block = localOrder.first; // localOrder.first is the Block*

         std::vector<mlir::Operation*> originalOrder;
         for (auto& op : *block) {
            if (!op.hasTrait<mlir::OpTrait::IsTerminator>()) {
               originalOrder.push_back(&op);
            }
         }

         // Build dependency map
         llvm::DenseMap<mlir::Operation*, std::vector<mlir::Operation*>> deps;

         // 1. Enforce strict MLIR SSA Dominance
         for (auto* op : originalOrder) {
            // Direct operands
            for (auto operand : op->getOperands()) {
               if (auto* def = operand.getDefiningOp()) {
                  if (def->getBlock() == block) {
                     deps[op].push_back(def);
                  }
               }
            }
            // Implicit captures inside regions (e.g., inside subop.nested_map)
            op->walk([&](mlir::Operation* nested) {
               if (nested == op) return;
               for (auto operand : nested->getOperands()) {
                  if (auto* def = operand.getDefiningOp()) {
                     if (def->getBlock() == block && def != op) {
                        deps[op].push_back(def);
                     }
                  }
               }
            });
         }

         // Cycle detection helper to ensure we never violate SSA dominance
         // when adding the soft `localOrder` dependencies.
         auto createsCycle = [&](mlir::Operation* curr, mlir::Operation* prev) {
            if (curr == prev) return true;
            std::vector<mlir::Operation*> worklist;
            llvm::SmallPtrSet<mlir::Operation*, 8> visitedNodes;

            worklist.push_back(prev);
            visitedNodes.insert(prev);
            while (!worklist.empty()) {
               auto* node = worklist.back();
               worklist.pop_back();
               if (node == curr) return true;

               for (auto* dep : deps[node]) {
                  if (visitedNodes.insert(dep).second) {
                     worklist.push_back(dep);
                  }
               }
            }
            return false;
         };

         // 2. Enforce SubOp ValidOrder (add only if it does not introduce a cycle)
         for (size_t i = 1; i < localOrder.second.size(); i++) {
            mlir::Operation* prev = localOrder.second[i - 1];
            mlir::Operation* curr = localOrder.second[i];

            if (prev->getBlock() == block && curr->getBlock() == block) {
               if (!createsCycle(curr, prev)) {
                  deps[curr].push_back(prev);
               }
            }
         }

         // 3. DFS-based Stable Topological Sort
         llvm::DenseSet<mlir::Operation*> visited;
         llvm::DenseSet<mlir::Operation*> visiting;
         std::vector<mlir::Operation*> sorted;

         std::function<void(mlir::Operation*)> visit = [&](mlir::Operation* op) {
            if (visited.count(op)) return;
            if (visiting.count(op)) return; // Should not be hit anymore since deps forms a perfect DAG
            visiting.insert(op);

            // Visit all dependencies first ensuring they are scheduled before this op
            for (auto* dep : deps[op]) {
               visit(dep);
            }

            visiting.erase(op);
            visited.insert(op);
            sorted.push_back(op);
         };

         for (auto* op : originalOrder) {
            visit(op);
         }

         // 4. Apply the strictly legal sorted order back to the block
         mlir::Operation* terminator = block->empty() ? nullptr : &block->back();
         for (auto* op : sorted) {
            if (terminator && terminator->hasTrait<mlir::OpTrait::IsTerminator>()) {
               op->moveBefore(terminator);
            } else {
               op->moveBefore(block, block->end());
            }
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> subop::createEnforceOrderPass() { return std::make_unique<EnforceOrderPass>(); }
