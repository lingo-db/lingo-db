#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
namespace {

class FinalizePass : public mlir::PassWrapper<FinalizePass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizePass)
   virtual llvm::StringRef getArgument() const override { return "subop-finalize"; }
   void cloneRec(mlir::Operation* op, mlir::IRMapping mapping) {
      mlir::OpBuilder builder(op);
      builder.clone(*op, mapping);
      for (auto *user : op->getUsers()) {
         cloneRec(user, mapping);
      }
   }
   void runOnOperation() override {
      auto module = getOperation();
      std::vector<mlir::subop::UnionOp> unionOps;
      module->walk([&](mlir::subop::UnionOp unionOp) {
         unionOps.push_back(unionOp);
      });
      for (size_t i = 0; i < unionOps.size(); i++) {
         auto currentUnion = unionOps[unionOps.size() - 1 - i];
         bool first = true;
         for (auto operand : currentUnion.getOperands()) {
            if (first) {
               first = false;
               continue;
            }
            mlir::IRMapping mapping;
            mapping.map(currentUnion.getResult(), operand);
            for (auto *user : currentUnion.getResult().getUsers()) {
               cloneRec(user, mapping);
            }
         }
         currentUnion->replaceAllUsesWith(mlir::ValueRange{currentUnion.getOperand(0)});
         currentUnion->erase();

      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createFinalizePass() { return std::make_unique<FinalizePass>(); }