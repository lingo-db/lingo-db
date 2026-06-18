#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>

namespace {
using namespace lingodb::compiler::dialect;

// Reuse per-iteration scratch state across `subop.loop` iterations.
//
// The RelAlg->SubOp lowering of a graphalg loop re-creates, every iteration, the
// transient build/aggregation state a join or group-by needs (hash tables, hash
// build buffers, scratch materialize buffers). Each `subop.create` registers a
// fresh runtime allocation that is only freed at end-of-query, so an N-iteration
// loop accumulates N copies of all of that intermediate state and runs out of
// memory on large graphs.
//
// `subop.loop` is not IsolatedFromAbove, so a state value created *before* the
// loop can be referenced from the body and is the same holder on every iteration.
// We therefore move each iteration-local scratch `create` out in front of the
// loop and emit a `subop.clear` at the top of the loop body, which resets the
// state in place (dropping its contents but keeping the backing allocation) at the
// start of every iteration. The single hoisted state is reused instead of leaking
// one allocation per round.
//
// This must run before SplitIntoExecutionSteps (which threads the now-external
// state into the body's nested execution group) and before Parallelize (which can
// then thread-localize the hoisted create just like any other top-level state; the
// per-iteration clear keeps the worker-local instances from accumulating).
//
// Only states that are dead at the iteration boundary are eligible: a state that
// is carried to the next iteration (an operand of `subop.loop_continue`, e.g. the
// ping-pong frontier buffer) must keep its contents and is left in place.
class ReuseLoopScratchPass : public mlir::PassWrapper<ReuseLoopScratchPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseLoopScratchPass)
   virtual llvm::StringRef getArgument() const override { return "subop-reuse-loop-scratch"; }

   // Only structures whose runtime backing can be reset in place (and that
   // dominate memory) are reused; small fixed-size states (simple_state) are left
   // alone since resetting them would require replaying their init region.
   static bool isReusableScratch(mlir::Type t) {
      return mlir::isa<subop::BufferType, subop::HashMapType, subop::HashMultiMapType>(t);
   }

   void processLoop(subop::LoopOp loopOp) {
      mlir::Region& loopRegion = loopOp.getRegion();
      mlir::Block* body = loopOp.getBody();
      mlir::Operation* terminator = body->getTerminator();

      // Values carried to the next iteration must not be cleared/reused.
      llvm::SmallPtrSet<mlir::Value, 8> carried;
      for (mlir::Value v : terminator->getOperands()) carried.insert(v);

      llvm::SmallVector<subop::GenericCreateOp> toHoist;
      for (mlir::Operation& op : *body) {
         auto createOp = mlir::dyn_cast<subop::GenericCreateOp>(&op);
         if (!createOp) continue;
         if (!isReusableScratch(createOp.getRes().getType())) continue;
         mlir::Value res = createOp.getRes();
         if (carried.contains(res)) continue;
         // Defensive: every use must stay inside the loop so referencing the
         // hoisted value from the body remains dominance-valid.
         bool allInLoop = true;
         for (mlir::Operation* u : res.getUsers())
            if (!loopRegion.isAncestor(u->getParentRegion())) { allInLoop = false; break; }
         if (!allInLoop) continue;
         toHoist.push_back(createOp);
      }
      if (toHoist.empty()) return;

      for (auto createOp : toHoist)
         createOp->moveBefore(loopOp);

      mlir::OpBuilder builder(loopOp.getContext());
      builder.setInsertionPointToStart(body);
      for (auto createOp : toHoist)
         builder.create<subop::ClearOp>(createOp.getLoc(), createOp.getRes());
   }

   void runOnOperation() override {
      llvm::SmallVector<subop::LoopOp> loops;
      getOperation()->walk([&](subop::LoopOp loopOp) { loops.push_back(loopOp); });
      for (subop::LoopOp loopOp : loops) processLoop(loopOp);
   }
};
} // end namespace

std::unique_ptr<mlir::Pass>
lingodb::compiler::dialect::subop::createReuseLoopScratchPass() {
   return std::make_unique<ReuseLoopScratchPass>();
}
