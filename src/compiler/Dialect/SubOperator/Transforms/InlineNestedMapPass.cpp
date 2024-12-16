#include "llvm/Support/Debug.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <queue>
namespace {
using namespace lingodb::compiler::dialect;

class InlineNestedMapPass : public mlir::PassWrapper<InlineNestedMapPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineNestedMapPass)
   virtual llvm::StringRef getArgument() const override { return "subop-nested-map-inline"; }

   void runOnOperation() override {
      std::queue<subop::NestedMapOp> nestedMapOps;
      getOperation()->walk([&](subop::NestedMapOp nestedMapOp) {
         nestedMapOps.push(nestedMapOp);
      });
      while(!nestedMapOps.empty()) {
         auto nestedMap = nestedMapOps.front();
         nestedMapOps.pop();
         auto returnOp = mlir::dyn_cast<tuples::ReturnOp>(nestedMap.getRegion().front().getTerminator());
         if (!returnOp) {
            nestedMap.emitError("NestedMapOp must be terminated by a ReturnOp");
            return signalPassFailure();
         }
         if (returnOp->getNumOperands() != 1) {
            continue;
         }

         mlir::Value streamResult = nestedMap.getResult();
         auto builder = mlir::OpBuilder(returnOp);
         mlir::Value replacement = builder.create<subop::CombineTupleOp>(nestedMap.getLoc(), returnOp.getOperand(0), nestedMap.getRegion().front().getArgument(0));
         std::vector<mlir::Operation*> opsToMove;
         std::queue<std::tuple<mlir::OpOperand&, mlir::Value, bool, subop::ColumnMapping>> opsToProcess;
         for (auto& use : streamResult.getUses()) {
            opsToProcess.push({use, streamResult, false,{}});
         }
         std::vector<subop::UnionOp> unions;
         while (!opsToProcess.empty()) {
            auto [currentUse, v, encounteredUnion, columnMapping] = opsToProcess.front();
            auto* op = currentUse.getOwner();
            opsToProcess.pop();

            if (auto unionOp = mlir::dyn_cast<subop::UnionOp>(op)) {
               for (auto& use : unionOp.getResult().getUses()) {
                  opsToProcess.push({use, v, true, columnMapping});
               }
               if (!encounteredUnion) {
                  std::vector<mlir::Value> args(unionOp.getOperands().begin(), unionOp.getOperands().end());
                  args.erase(args.begin() + currentUse.getOperandNumber());
                  unionOp->setOperands(args);
               }
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
               auto* cloned =  mlir::cast<subop::SubOperator>(op).cloneSubOp(builder, mapping,  columnMapping);
               if(auto clonedNestedMap = mlir::dyn_cast<subop::NestedMapOp>(cloned)) {
                  nestedMapOps.push(clonedNestedMap);
               }
               opsToMove.push_back(cloned);
               for (auto& use : op->getUses()) {
                  if (mlir::isa_and_nonnull<tuples::TupleStreamType>(use.get().getType())) {
                     opsToProcess.push({use, mapping.lookup(use.get()), encounteredUnion, columnMapping});
                  }
               }
            } else {
               opsToMove.push_back(op);
               for (auto& use : op->getUses()) {
                  if (mlir::isa_and_nonnull<tuples::TupleStreamType>(use.get().getType())) {
                     opsToProcess.push({use, use.get(), encounteredUnion, columnMapping});
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
                  if (auto unionOp = mlir::dyn_cast<subop::UnionOp>(use.getOwner())) {
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
subop::createInlineNestedMapPass() { return std::make_unique<InlineNestedMapPass>(); }