#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
namespace {
class EnforceOrderPass : public mlir::PassWrapper<EnforceOrderPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnforceOrderPass)
   virtual llvm::StringRef getArgument() const override { return "subop-enforce-order"; }

   void runOnOperation() override {
      auto subOpDependencyAnalysis = getAnalysis<mlir::subop::SubOpDependencyAnalysis>();
      std::vector<std::pair<mlir::Operation*, mlir::Operation*>> otherOrdering;
      for (auto& localOrder : subOpDependencyAnalysis.validOrder) {
         mlir::Operation* last = nullptr;
         for (auto* x : localOrder.second) {
            if (!last) {
               x->moveAfter(localOrder.first, localOrder.first->begin());
            } else {
               x->moveAfter(last);
            }
            last = x;
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::subop::createEnforceOrderPass() { return std::make_unique<EnforceOrderPass>(); }
