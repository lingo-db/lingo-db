#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGSEMINAIVE
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Semiring classification
//===----------------------------------------------------------------------===//

// The additive monoid of a semiring is idempotent (a + a == a) exactly for the
// OR / min / max semirings. Idempotency is what makes semi-naive evaluation
// sound: re-adding an already-settled cell leaves the state unchanged, so only
// the cells that actually changed (the delta) need to be propagated. The plain
// integer/real semirings use `+`, which is not idempotent, and are rejected.
bool hasIdempotentAdd(mlir::Type semiring) {
   if (semiring.isInteger(1) || semiring == SemiringTypes::forBool(semiring.getContext()))
      return true; // boolean OR
   if (llvm::isa<TropI64Type, TropF64Type>(semiring))
      return true; // tropical min
   if (semiring == SemiringTypes::forTropMaxInt(semiring.getContext()))
      return true; // tropical max
   return false;
}

bool carriesIdempotentSemiring(mlir::Value v) {
   auto mat = llvm::dyn_cast<MatrixType>(v.getType());
   return mat && hasIdempotentAdd(mat.getSemiring());
}

//===----------------------------------------------------------------------===//
// Linearity analysis
//===----------------------------------------------------------------------===//

// Classifies a value by how it depends on the loop-carried state arguments `M`:
//
//   Invariant  - independent of every state arg (loop-invariant constant).
//   Linear     - a degree-1 linear function g of the state with g(0) = 0, i.e.
//                g(X + Y) = g(X) + g(Y). Safe to evaluate on a delta of M.
//   Nonlinear  - anything else: products of two state-dependent factors,
//                affine expressions (linear + non-zero constant), values that
//                depend on the iteration index, or ops we cannot reason about.
//
// The classification is conservative: it only returns `Linear`/`Invariant` when
// it can prove the property, and falls back to `Nonlinear` otherwise.
enum class Linearity { Invariant, Linear, Nonlinear };

class LinearityAnalysis {
   public:
   LinearityAnalysis(mlir::Operation* loop, const llvm::SmallPtrSetImpl<mlir::Value>& stateArgs)
      : loop(loop), stateArgs(stateArgs) {}

   Linearity classify(mlir::Value v) {
      if (auto it = memo.find(v); it != memo.end())
         return it->second;
      // Guard against the (acyclic, but defensive) chance of revisiting.
      memo[v] = Linearity::Nonlinear;
      Linearity result = compute(v);
      memo[v] = result;
      return result;
   }

   private:
   // Additive combine (union / reduce / mask blend): a non-zero constant operand
   // turns the result affine, which is not linear.
   static Linearity joinAdditive(Linearity a, Linearity b) {
      if (a == Linearity::Nonlinear || b == Linearity::Nonlinear)
         return Linearity::Nonlinear;
      if (a == Linearity::Invariant && b == Linearity::Invariant)
         return Linearity::Invariant;
      if (a == Linearity::Linear && b == Linearity::Linear)
         return Linearity::Linear;
      // exactly one Linear and one Invariant -> linear + constant -> affine
      return Linearity::Nonlinear;
   }

   // Multiplicative combine (matmul): at most one factor may depend on the state.
   static Linearity joinMultiplicative(Linearity a, Linearity b) {
      if (a == Linearity::Invariant && b == Linearity::Invariant)
         return Linearity::Invariant;
      if ((a == Linearity::Linear && b == Linearity::Invariant) ||
          (a == Linearity::Invariant && b == Linearity::Linear))
         return Linearity::Linear;
      return Linearity::Nonlinear; // state*state (degree 2) or nonlinear factor
   }

   Linearity compute(mlir::Value v) {
      if (stateArgs.contains(v))
         return Linearity::Linear;

      if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
         // The loop body's own non-state argument is the induction index, which
         // varies per iteration -> non-linear. Any other block argument belongs
         // to an enclosing region (e.g. a function parameter) -> loop-invariant.
         if (blockArg.getOwner()->getParentOp() == loop)
            return Linearity::Nonlinear;
         return Linearity::Invariant;
      }

      auto* def = v.getDefiningOp();
      if (!loop->isAncestor(def))
         return Linearity::Invariant; // defined outside the loop

      return llvm::TypeSwitch<mlir::Operation*, Linearity>(def)
         .Case<MatMulJoinOp>([&](MatMulJoinOp op) {
            return joinMultiplicative(classify(op.getLhs()), classify(op.getRhs()));
         })
         .Case<TransposeOp, ReduceOp, PickAnyOp>(
            [&](mlir::Operation* op) { return classify(op->getOperand(0)); })
         .Case<UnionOp, DeferredReduceOp>([&](mlir::Operation* op) {
            return foldAdditive(op->getOperands());
         })
         .Case<MaskOp>([&](MaskOp op) {
            // Blend selects `input` where `mask` is set, else `base`. With a
            // loop-invariant mask this is a linear combination of base/input;
            // a state-dependent mask makes it non-linear.
            if (classify(op.getMask()) != Linearity::Invariant)
               return Linearity::Nonlinear;
            return joinAdditive(classify(op.getBase()), classify(op.getInput()));
         })
         .Default([&](mlir::Operation* op) {
            // Unknown op (apply, elementwise, broadcast, const, ...): linear only
            // if it touches no state at all (a loop-invariant subcomputation).
            for (mlir::Value operand : op->getOperands())
               if (classify(operand) != Linearity::Invariant)
                  return Linearity::Nonlinear;
            return Linearity::Invariant;
         });
   }

   // Classify an additive combination (union / reduce) of `operands`. A non-zero
   // invariant term mixed with a linear term yields an affine result, which is
   // not linear; all-linear stays linear; all-invariant stays invariant.
   Linearity foldAdditive(mlir::ValueRange operands) {
      bool anyLinear = false, anyInvariant = false;
      for (mlir::Value operand : operands) {
         switch (classify(operand)) {
            case Linearity::Nonlinear: return Linearity::Nonlinear;
            case Linearity::Linear: anyLinear = true; break;
            case Linearity::Invariant: anyInvariant = true; break;
         }
      }
      if (anyLinear && anyInvariant) return Linearity::Nonlinear; // affine
      if (anyLinear) return Linearity::Linear;
      return Linearity::Invariant;
   }

   mlir::Operation* loop;
   const llvm::SmallPtrSetImpl<mlir::Value>& stateArgs;
   llvm::DenseMap<mlir::Value, Linearity> memo;
};

//===----------------------------------------------------------------------===//
// Eligibility
//===----------------------------------------------------------------------===//

// True if `value` combines `arg` additively with other terms, i.e. it has the
// shape `arg (+) rest` (modulo reduce / pick_any wrappers and nested unions).
// This ensures the iteration *accumulates* onto the previous state rather than
// replacing it, which is the precondition for delta propagation.
bool accumulatesSelf(mlir::Value value, mlir::Value arg) {
   llvm::SmallVector<mlir::Value> worklist{value};
   llvm::SmallPtrSet<mlir::Value, 8> seen;
   while (!worklist.empty()) {
      mlir::Value cur = worklist.pop_back_val();
      if (!seen.insert(cur).second)
         continue;
      if (cur == arg)
         return true;
      auto* def = cur.getDefiningOp();
      if (!def)
         continue;
      // Only look through additive/aggregation wrappers; `arg` appearing only as
      // an input deeper inside a transform (e.g. a matmul factor) does not count
      // as a self-accumulation term.
      if (llvm::isa<ReduceOp, DeferredReduceOp, PickAnyOp, UnionOp>(def))
         worklist.append(def->getOperands().begin(), def->getOperands().end());
   }
   return false;
}

bool isEligible(ForDimOp loop) {
   // v1 only handles the fixed-trip-count form (empty `until`); a convergence
   // predicate would have to be rewritten in terms of the delta as well.
   if (!loop.getUntil().empty())
      return false;

   mlir::Block& body = loop.getBody().front();
   // Body arguments are [induction index, carried state matrices...].
   if (body.getNumArguments() < 2)
      return false;
   auto stateArgs = body.getArguments().drop_front(1);

   // Every carried state must use an idempotent additive monoid.
   for (mlir::Value arg : stateArgs)
      if (!carriesIdempotentSemiring(arg))
         return false;

   auto yield = llvm::cast<YieldOp>(body.getTerminator());
   if (yield.getInputs().size() != stateArgs.size())
      return false;

   llvm::SmallPtrSet<mlir::Value, 4> stateArgSet(stateArgs.begin(), stateArgs.end());
   LinearityAnalysis analysis(loop, stateArgSet);

   for (auto [arg, next] : llvm::zip_equal(stateArgs, yield.getInputs())) {
      // Each next-state must be M_next = M (+) g(M) with g linear.
      if (analysis.classify(next) != Linearity::Linear)
         return false;
      if (!accumulatesSelf(next, arg))
         return false;
      // The body's final combine must be an idempotent reducer; the rewrite
      // reuses it to accumulate the delta into the state.
      mlir::Operation* outer = next.getDefiningOp();
      if (!outer || !llvm::isa<PickAnyOp, DeferredReduceOp>(outer))
         return false;
   }
   return true;
}

// Builds a 1x1 boolean matrix that is true iff `d` has no stored cells. Mirrors
// the desugaring of `graphalg.nvals(d) == 0` (the form BFS's `until` uses):
//   isEmpty(d) = ( reduce_SUM(apply(d){ a -> cast<int>(a != identity) }) == 0 )
mlir::Value buildIsEmpty(mlir::OpBuilder& b, mlir::Location loc, mlir::Value d) {
   auto* ctx = b.getContext();
   auto dMat = llvm::cast<MatrixType>(d.getType());
   auto inputRing = llvm::cast<SemiringTypeInterface>(dMat.getSemiring());
   mlir::Type intRing = SemiringTypes::forInt(ctx);
   mlir::Type boolRing = SemiringTypes::forBool(ctx);
   mlir::Type scalarInt = MatrixType::scalarOf(intRing);
   mlir::Type scalarBool = MatrixType::scalarOf(boolRing);

   // nvals: count each present (non-identity) cell as 1, then sum.
   auto countApply = b.create<ApplyOp>(loc, dMat.withSemiring(intRing), mlir::ValueRange{d});
   {
      mlir::OpBuilder::InsertionGuard g(b);
      mlir::Block& body = countApply.createBody();
      b.setInsertionPointToStart(&body);
      auto arg = body.getArgument(0);
      auto zero = b.create<ConstantOp>(loc, inputRing.addIdentity());
      auto eqZero = b.create<EqOp>(loc, arg, zero.getResult());
      auto falseC = b.create<ConstantOp>(loc, b.getBoolAttr(false));
      auto nonZero = b.create<EqOp>(loc, eqZero.getResult(), falseC.getResult());
      auto asInt = b.create<CastScalarOp>(loc, intRing, nonZero.getResult());
      b.create<ApplyReturnOp>(loc, asInt.getResult());
   }
   mlir::Value count = b.create<ReduceOp>(loc, scalarInt, countApply.getResult());

   auto emptyApply = b.create<ApplyOp>(loc, scalarBool, mlir::ValueRange{count});
   {
      mlir::OpBuilder::InsertionGuard g(b);
      mlir::Block& body = emptyApply.createBody();
      b.setInsertionPointToStart(&body);
      auto arg = body.getArgument(0);
      auto zero = b.create<ConstantOp>(loc, b.getI64IntegerAttr(0));
      auto eq = b.create<EqOp>(loc, arg, zero.getResult());
      b.create<ApplyReturnOp>(loc, eq.getResult());
   }
   return emptyApply.getResult();
}

// Rewrites an eligible fixpoint loop into semi-naive form. Each carried state M
// is paired with a delta D; the loop body is re-run on the deltas (so only the
// changed cells are propagated), the result accumulated into M, and the next
// delta extracted:
//
//   for (M, D = init):
//     P_k       = body_k(D...)              // original body, M args -> D args
//     D_k_next  = delta(P_k, M_k)           // newly-improving cells
//     M_k_next  = reduce(union(M_k, D_k))   // accumulate
//
// Relies on the eligibility guarantee that body_k is linear over an idempotent
// semiring, so f(M) = M (+) g(M) and M (+) f(D) reproduces the next iterate.
void rewriteSemiNaive(ForDimOp loop) {
   mlir::OpBuilder builder(loop);
   mlir::Location loc = loop.getLoc();
   mlir::Block& oldBody = loop.getBody().front();
   auto oldYield = llvm::cast<YieldOp>(oldBody.getTerminator());
   unsigned n = oldBody.getNumArguments() - 1; // minus the induction index

   // New init/result types: the carried states M, then their deltas D (both
   // start from the original init values).
   llvm::SmallVector<mlir::Value> initArgs(loop.getInitArgs().begin(), loop.getInitArgs().end());
   initArgs.append(loop.getInitArgs().begin(), loop.getInitArgs().end());
   llvm::SmallVector<mlir::Type> resultTypes(loop.getResultTypes().begin(), loop.getResultTypes().end());
   resultTypes.append(loop.getResultTypes().begin(), loop.getResultTypes().end());

   auto newLoop = builder.create<ForDimOp>(loc, resultTypes, initArgs, loop.getDimAttr());
   mlir::Block* body = builder.createBlock(&newLoop.getBody());

   mlir::Value idx = body->addArgument(oldBody.getArgument(0).getType(), loc);
   llvm::SmallVector<mlir::Value> ms, ds;
   for (unsigned k = 0; k < n; ++k)
      ms.push_back(body->addArgument(oldBody.getArgument(1 + k).getType(), loc));
   for (unsigned k = 0; k < n; ++k)
      ds.push_back(body->addArgument(oldBody.getArgument(1 + k).getType(), loc));

   // Clone the original body, feeding the deltas where it used the states.
   mlir::IRMapping map;
   map.map(oldBody.getArgument(0), idx);
   for (unsigned k = 0; k < n; ++k)
      map.map(oldBody.getArgument(1 + k), ds[k]);
   builder.setInsertionPointToEnd(body);
   for (mlir::Operation& op : oldBody.without_terminator())
      builder.clone(op, map);

   llvm::SmallVector<mlir::Value> nextMs, nextDs;
   for (unsigned k = 0; k < n; ++k) {
      mlir::Value propagated = map.lookupOrDefault(oldYield.getInputs()[k]); // P = f(D)
      mlir::Type t = ms[k].getType();
      // M_next = OUTER(M (+) P), where OUTER is the body's own final reducer
      // (e.g. pick_any vs deferred_reduce) so the accumulated state keeps the
      // same canonical form the original loop produced.
      mlir::Value combined = builder.create<UnionOp>(loc, t, mlir::ValueRange{ms[k], propagated});
      mlir::Operation* outer = oldYield.getInputs()[k].getDefiningOp();
      mlir::Value stateNext;
      if (llvm::isa<PickAnyOp>(outer))
         stateNext = builder.create<PickAnyOp>(loc, t, combined);
      else
         stateNext = builder.create<DeferredReduceOp>(loc, t, mlir::ValueRange{combined});
      // Next delta = the cells of the new state that improved over the old.
      mlir::Value deltaNext = builder.create<DeltaOp>(loc, stateNext, ms[k]);
      nextMs.push_back(stateNext);
      nextDs.push_back(deltaNext);
   }
   llvm::SmallVector<mlir::Value> yielded(nextMs);
   yielded.append(nextDs.begin(), nextDs.end());
   builder.create<YieldOp>(loc, yielded);

   // Early termination: stop once every delta is empty (fixpoint reached). The
   // until block receives (index, next-states..., next-deltas...).
   mlir::Block* untilBlk = builder.createBlock(&newLoop.getUntil());
   untilBlk->addArgument(oldBody.getArgument(0).getType(), loc); // index
   for (unsigned k = 0; k < n; ++k)
      untilBlk->addArgument(ms[k].getType(), loc); // next states (unused)
   llvm::SmallVector<mlir::Value> untilDeltas;
   for (unsigned k = 0; k < n; ++k)
      untilDeltas.push_back(untilBlk->addArgument(ms[k].getType(), loc));
   builder.setInsertionPointToEnd(untilBlk);
   mlir::Value stop = buildIsEmpty(builder, loc, untilDeltas[0]);
   for (unsigned k = 1; k < n; ++k) {
      // stop only when *all* deltas are empty: AND via boolean multiply.
      mlir::Value e = buildIsEmpty(builder, loc, untilDeltas[k]);
      auto andOp = builder.create<ApplyOp>(loc, stop.getType(), mlir::ValueRange{stop, e});
      mlir::OpBuilder::InsertionGuard g(builder);
      mlir::Block& body = andOp.createBody();
      builder.setInsertionPointToStart(&body);
      auto m = builder.create<MulOp>(loc, body.getArgument(0), body.getArgument(1));
      builder.create<ApplyReturnOp>(loc, m.getResult());
      stop = andOp.getResult();
   }
   builder.create<YieldOp>(loc, stop);

   // The states are the first n results; rewire users and drop the old loop.
   for (unsigned k = 0; k < n; ++k)
      loop.getResult(k).replaceAllUsesWith(newLoop.getResult(k));
   loop.erase();
}

class GraphAlgSemiNaive
   : public impl::GraphAlgSemiNaiveBase<GraphAlgSemiNaive> {
   using impl::GraphAlgSemiNaiveBase<GraphAlgSemiNaive>::GraphAlgSemiNaiveBase;

   void runOnOperation() final {
      llvm::SmallVector<ForDimOp> eligible;
      getOperation().walk([&](ForDimOp loop) {
         if (isEligible(loop))
            eligible.push_back(loop);
      });
      for (ForDimOp loop : eligible)
         rewriteSemiNaive(loop);
   }
};

} // namespace
} // namespace graphalg
