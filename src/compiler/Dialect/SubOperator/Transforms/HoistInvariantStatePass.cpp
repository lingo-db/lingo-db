#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>

namespace {
using namespace lingodb::compiler::dialect;

// Loop-invariant code motion for `subop.loop`.
//
// graphalg loops re-derive a join's build side (e.g. the adjacency matrix of a
// matmul/vxm) every iteration: the RelAlg->SubOp lowering emits, inside the loop
// body, a fresh `create buffer` + `materialize <edges>` + `create_hash_indexed_view`
// and only the lookup actually depends on the per-iteration frontier. Re-building
// that O(E) index each round is the dominant cost on large graphs. `subop.loop` is
// not IsolatedFromAbove, so any op whose inputs are all loop-invariant can simply
// be moved before the loop and referenced from the body; SplitIntoExecutionSteps
// later threads it in.
//
// Correctness hinges on state mutation. A subop state is mutated not only by ops
// that take the state value directly (materialize/insert/create_hash_indexed_view)
// but also indirectly: `lookup`/`lookup_or_insert`/`scan_refs` hand out a
// StateEntryReference (or a list of them), and a later `scatter`/`reduce` writes
// through that reference. So mutation has to be tracked transitively through
// ref/list-typed SSA values, not just at the immediate users of the state.
//
//   * An op reading a state defined OUTSIDE the loop is only hoistable if that
//     state is never mutated inside the loop (otherwise the moved read would see
//     stale data).
//   * An op CREATING a state may only be hoisted if every in-loop op that mutates
//     that state is hoisted as well, so that building it once equals rebuilding it
//     each iteration (e.g. the edge buffer + its materialize + hash-indexed-view
//     all move together, while a per-iteration accumulator whose `reduce` stays in
//     the loop is left in place).
class HoistInvariantStatePass : public mlir::PassWrapper<HoistInvariantStatePass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistInvariantStatePass)
   virtual llvm::StringRef getArgument() const override { return "subop-hoist-invariant-state"; }

   // Ops that take a state as a direct SSA operand and write to it.
   static bool isDirectStateWriter(mlir::Operation* op) {
      return mlir::isa<subop::MaterializeOp, subop::InsertOp,
                       subop::LookupOrInsertOp, subop::CreateHashIndexedView>(op);
   }
   // Ops that create a view aliasing their input state: a write through the view
   // is a write to the underlying state.
   static bool isViewCreator(mlir::Operation* op) {
      return mlir::isa<subop::CreateContinuousView, subop::CreateSortedViewOp,
                       subop::CreateSegmentTreeView, subop::CreateHashIndexedView>(op);
   }

   void processLoop(subop::LoopOp loopOp) {
      mlir::Region& loopRegion = loopOp.getRegion();
      mlir::Block* body = loopOp.getBody();

      auto inLoop = [&](mlir::Operation* op) {
         return loopRegion.isAncestor(op->getParentRegion());
      };
      auto definedOutside = [&](mlir::Value v) {
         return !loopRegion.isAncestor(v.getParentRegion());
      };
      // Is `v` an SSA value local to `op` (defined within one of its regions)?
      auto internalToOp = [&](mlir::Value v, mlir::Operation* op) {
         mlir::Operation* owner = v.getDefiningOp();
         if (!owner) owner = v.getParentBlock()->getParentOp();
         while (owner) {
            if (owner == op) return true;
            owner = owner->getParentOp();
         }
         return false;
      };

      // Collect every op inside the loop that mutates the state backing `stateVal`.
      //
      // A subop state is written either directly -- an op takes the state as an
      // SSA operand (materialize/insert/lookup_or_insert/create_hash_indexed_view)
      // -- or indirectly through a *reference column*: a `lookup`/`lookup_or_insert`/
      // `scan_refs` reading the state produces a ColumnDef reference, and a later
      // `scatter`/`reduce` writes through a ColumnRef to that same column. The
      // reference is carried as a column inside the tuplestream, not as an SSA
      // value, so mutation is tracked by column identity rather than def-use.
      auto collectMutators = [&](mlir::Value stateVal,
                                 llvm::SmallPtrSetImpl<mlir::Operation*>& mutators) {
         llvm::SmallPtrSet<const tuples::Column*, 8> tainted;
         // Seed: direct users of the state value and of any view aliasing it (a
         // write through a continuous/sorted/segment-tree/hash-indexed view is a
         // write to the underlying state).
         llvm::SmallVector<mlir::Value> stateWork{stateVal};
         llvm::SmallPtrSet<mlir::Value, 8> seenState;
         while (!stateWork.empty()) {
            mlir::Value s = stateWork.pop_back_val();
            if (!seenState.insert(s).second) continue;
            for (mlir::Operation* u : s.getUsers()) {
               if (!inLoop(u)) continue;
               if (isDirectStateWriter(u)) mutators.insert(u);
               if (isViewCreator(u))
                  for (mlir::Value r : u->getResults()) stateWork.push_back(r);
               if (auto rp = mlir::dyn_cast<subop::ReferenceProducer>(u))
                  tainted.insert(rp.getProducedReference().getColumnPtr().get());
            }
         }
         // Fixpoint: propagate taint through reference-deriving ops and flag every
         // scatter/reduce that writes through a tainted reference column.
         bool ch = true;
         while (ch) {
            ch = false;
            loopOp.getRegion().walk([&](mlir::Operation* o) {
               const tuples::Column* in = nullptr;
               const tuples::Column* out = nullptr;
               if (auto x = mlir::dyn_cast<subop::UnwrapOptionalRefOp>(o)) {
                  in = x.getOptionalRef().getColumnPtr().get();
                  out = x.getRef().getColumnPtr().get();
               } else if (auto x = mlir::dyn_cast<subop::OffsetReferenceBy>(o)) {
                  in = x.getRef().getColumnPtr().get();
                  out = x.getNewRef().getColumnPtr().get();
               }
               if (in && out && tainted.contains(in) && tainted.insert(out).second)
                  ch = true;
               const tuples::Column* wc = nullptr;
               if (auto s = mlir::dyn_cast<subop::ScatterOp>(o))
                  wc = s.getRef().getColumnPtr().get();
               else if (auto r = mlir::dyn_cast<subop::ReduceOp>(o))
                  wc = r.getRef().getColumnPtr().get();
               if (wc && tainted.contains(wc)) mutators.insert(o);
            });
         }
      };
      auto stateMutatedInLoop = [&](mlir::Value stateVal) {
         llvm::SmallPtrSet<mlir::Operation*, 8> mutators;
         collectMutators(stateVal, mutators);
         return !mutators.empty();
      };

      llvm::SmallPtrSet<mlir::Operation*, 16> cand;

      // Is `v` a dependency that prevents hoisting `op`? A value is fine if it is
      // local to `op`'s own regions, defined outside the loop (and, when a state,
      // never mutated in the loop), or produced by another current candidate.
      auto badDep = [&](mlir::Value v, mlir::Operation* op) {
         if (internalToOp(v, op)) return false;
         if (definedOutside(v)) {
            return mlir::isa<subop::State>(v.getType()) && stateMutatedInLoop(v);
         }
         mlir::Operation* def = v.getDefiningOp();
         return !def || !cand.contains(def);
      };
      // An op is hoistable when neither its direct operands nor any value captured
      // by its nested regions is a bad dependency.
      auto hoistable = [&](mlir::Operation* op) {
         for (mlir::Value v : op->getOperands())
            if (badDep(v, op)) return false;
         bool ok = true;
         op->walk([&](mlir::Operation* nested) {
            if (nested == op) return;
            for (mlir::Value v : nested->getOperands())
               if (badDep(v, op)) ok = false;
         });
         return ok;
      };

      // Greedy fixpoint: collect candidates over the loop body's top-level ops.
      bool changed = true;
      while (changed) {
         changed = false;
         for (mlir::Operation& op : *body) {
            if (op.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
            if (mlir::isa<subop::LoopOp>(op)) continue;
            if (cand.contains(&op)) continue;
            if (hoistable(&op)) {
               cand.insert(&op);
               changed = true;
            }
         }
      }

      // Validation fixpoint: demote until a stable, self-consistent candidate set.
      changed = true;
      while (changed) {
         changed = false;
         llvm::SmallVector<mlir::Operation*> snapshot(cand.begin(), cand.end());
         for (mlir::Operation* op : snapshot) {
            if (!cand.contains(op)) continue;
            // A dependency (direct operand or nested-region capture) was demoted,
            // so this op can no longer be hoisted either.
            bool demote = !hoistable(op);
            // Tuple streams are single-use and cannot cross the loop boundary: a
            // stream produced before the loop but consumed inside would be a
            // one-shot pipeline evaluated once instead of per iteration. Only hoist
            // a stream-producing op when its consumer is hoisted too, so whole
            // pipeline segments move together.
            if (!demote) {
               for (mlir::Value r : op->getResults()) {
                  if (!mlir::isa<tuples::TupleStreamType>(r.getType())) continue;
                  for (mlir::Operation* u : r.getUsers())
                     if (!cand.contains(u)) { demote = true; break; }
                  if (demote) break;
               }
            }
            // A created state may only be hoisted if every in-loop op that mutates
            // it is hoisted as well (tracked transitively through references).
            if (!demote) {
               for (mlir::Value r : op->getResults()) {
                  if (!mlir::isa<subop::State>(r.getType())) continue;
                  llvm::SmallPtrSet<mlir::Operation*, 8> mutators;
                  collectMutators(r, mutators);
                  for (mlir::Operation* m : mutators)
                     if (!cand.contains(m)) { demote = true; break; }
                  if (demote) break;
               }
            }
            if (demote) {
               cand.erase(op);
               changed = true;
            }
         }
      }

      if (cand.empty()) return;

      // Move hoisted ops (in original body order) to just before the loop.
      llvm::SmallVector<mlir::Operation*> toMove;
      for (mlir::Operation& op : *body)
         if (cand.contains(&op)) toMove.push_back(&op);
      for (mlir::Operation* op : toMove)
         op->moveBefore(loopOp);
   }

   void runOnOperation() override {
      llvm::SmallVector<subop::LoopOp> loops;
      getOperation()->walk([&](subop::LoopOp loopOp) { loops.push_back(loopOp); });
      // Outer loops first: hoisting out of an outer loop can expose invariance for
      // an inner one on a later run; a single pass suffices for the common
      // (single-level) graphalg loops.
      for (subop::LoopOp loopOp : loops) processLoop(loopOp);
   }
};
} // end namespace

std::unique_ptr<mlir::Pass>
lingodb::compiler::dialect::subop::createHoistInvariantStatePass() {
   return std::make_unique<HoistInvariantStatePass>();
}
