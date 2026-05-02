#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGLOOPAGGREGATE
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

class GraphAlgLoopAggregate
   : public impl::GraphAlgLoopAggregateBase<GraphAlgLoopAggregate> {
   using impl::GraphAlgLoopAggregateBase<
      GraphAlgLoopAggregate>::GraphAlgLoopAggregateBase;

   void runOnOperation() final;
};

} // namespace

// If the loop body ends with an aggregation op, ensure the init arg is also an
// aggregation. Later on in the AvantGraph query pipeline, this will signal to
// the optimizer that the iteration state can be kept as an aggregate table.
static void addInitReduce(mlir::Operation* op, mlir::IRRewriter& rewriter) {
   mlir::Block* body;
   llvm::SmallVector<mlir::Value> newInitArgs;
   if (auto constOp = llvm::dyn_cast<ForConstOp>(op)) {
      body = &constOp.getBody().front();
      newInitArgs = constOp.getInitArgs();
   } else {
      auto dimOp = llvm::cast<ForDimOp>(op);
      body = &dimOp.getBody().front();
      newInitArgs = dimOp.getInitArgs();
   }

   auto yieldOp = llvm::cast<YieldOp>(body->getTerminator());
   for (auto [i, iterResult] : llvm::enumerate(yieldOp.getInputs())) {
      auto iterLastOp = iterResult.getDefiningOp();
      if (llvm::isa_and_present<PickAnyOp, DeferredReduceOp>(iterLastOp)) {
         auto oldArg = newInitArgs[i];
         newInitArgs[i] = rewriter.create<DeferredReduceOp>(
            oldArg.getLoc(), oldArg.getType(), oldArg);
      }
   }

   rewriter.modifyOpInPlace(op, [&]() {
      if (auto constOp = llvm::dyn_cast<ForConstOp>(op)) {
         constOp.getInitArgsMutable().assign(newInitArgs);
      } else {
         auto dimOp = llvm::cast<ForDimOp>(op);
         dimOp.getInitArgsMutable().assign(newInitArgs);
      }
   });
}

void GraphAlgLoopAggregate::runOnOperation() {
   llvm::SmallVector<mlir::Operation*> loopOps;
   getOperation()->walk([&](ForConstOp op) { loopOps.emplace_back(op); });
   getOperation()->walk([&](ForDimOp op) { loopOps.emplace_back(op); });

   mlir::IRRewriter rewriter(&getContext());
   for (auto op : loopOps) {
      rewriter.setInsertionPoint(op);
      addInitReduce(op, rewriter);
   }
}

} // namespace graphalg
