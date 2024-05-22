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

class InlineNestedMapPass : public mlir::PassWrapper<InlineNestedMapPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineNestedMapPass)
   virtual llvm::StringRef getArgument() const override { return "subop-nested-map-inline"; }

   void runOnOperation() override {
      std::vector<mlir::subop::NestedMapOp> nestedMapOps;
      getOperation()->walk([&](mlir::subop::NestedMapOp nestedMapOp) {
         nestedMapOps.push_back(nestedMapOp);
      });
      for (auto nestedMap : nestedMapOps) {
         auto returnOp = mlir::dyn_cast<mlir::tuples::ReturnOp>(nestedMap.getRegion().front().getTerminator());
         if (!returnOp) {
            nestedMap.emitError("NestedMapOp must be terminated by a ReturnOp");
            return signalPassFailure();
         }
         if (returnOp->getNumOperands() != 1) {
            continue;
         }

         mlir::Value streamResult = nestedMap.getResult();
         auto builder = mlir::OpBuilder(returnOp);
         mlir::Value replacement = builder.create<mlir::subop::CombineTupleOp>(nestedMap.getLoc(), returnOp.getOperand(0), nestedMap.getRegion().front().getArgument(0));
         std::vector<mlir::Operation*> opsToMove;
         std::queue<std::tuple<mlir::OpOperand&, mlir::Value, bool>> opsToProcess;
         for (auto& use : streamResult.getUses()) {
            opsToProcess.push({use, streamResult, false});
         }
         std::vector<mlir::subop::UnionOp> unions;
         while (!opsToProcess.empty()) {
            auto [currentUse, v, encounteredUnion] = opsToProcess.front();
            auto *op = currentUse.getOwner();
            opsToProcess.pop();

            if (auto unionOp = mlir::dyn_cast<mlir::subop::UnionOp>(op)) {
               for (auto& use : unionOp.getResult().getUses()) {
                  opsToProcess.push({use, v, true});
               }
               std::vector<mlir::Value> args(unionOp.getOperands().begin(), unionOp.getOperands().end());
               args.erase(args.begin() + currentUse.getOperandNumber());
               unionOp->setOperands(args);
               unions.push_back(unionOp);
               continue;
            }
            if (std::find(opsToMove.begin(), opsToMove.end(), op) != opsToMove.end()) {
               continue;
            }
            if (encounteredUnion) {
               mlir::OpBuilder builder(op);
               mlir::IRMapping mapping;
               mapping.map(currentUse.get(), v);
               auto *cloned=builder.clone(*op, mapping);
               opsToMove.push_back(cloned);
               for (auto& use : op->getUses()) {
                  if (use.get().getType().isa_and_nonnull<mlir::tuples::TupleStreamType>()) {
                     opsToProcess.push({use, mapping.lookup(use.get()), encounteredUnion});
                  }
               }
            } else {
               opsToMove.push_back(op);
               for (auto& use : op->getUses()) {
                  if (use.get().getType().isa_and_nonnull<mlir::tuples::TupleStreamType>()) {
                     opsToProcess.push({use, use.get(), encounteredUnion});
                  }
               }
            }

         }
         for (auto* op : opsToMove) {
            op->moveBefore(returnOp);
         }
         streamResult.replaceAllUsesWith(replacement);
         returnOp->setOperands({});
         for (auto unionOp : unions) {
            if (unionOp.getNumOperands() == 1) {
               unionOp->replaceAllUsesWith(unionOp.getOperands());
               unionOp->erase();
            } else if (unionOp.getNumOperands() == 0) {
               std::function<void(mlir::OpOperand&)> removeFn = [&](mlir::OpOperand& use) {
                  if (auto unionOp = mlir::dyn_cast<mlir::subop::UnionOp>(use.getOwner())) {
                     std::vector<mlir::Value> args(unionOp.getOperands().begin(), unionOp.getOperands().end());
                     args.erase(args.begin() + use.getOperandNumber());
                     unionOp->setOperands(args);
                     unions.push_back(unionOp);
                  } else {
                     for (auto& use : use.getOwner()->getUses()) {
                        removeFn(use);
                     }
                     use.getOwner()->erase();
                  }
               };
               for (auto& use : unionOp->getUses()) {
                  removeFn(use);
               }
               unionOp->erase();
            }
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createInlineNestedMapPass() { return std::make_unique<InlineNestedMapPass>(); }