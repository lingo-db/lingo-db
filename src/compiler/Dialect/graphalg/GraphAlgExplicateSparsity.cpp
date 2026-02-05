#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/FoldUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/analysis/DenseAnalysis.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGEXPLICATESPARSITY
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

/**
 * Explicitly makes matrices dense if they are used by ops that have different
 * semantics for dense vs. sparse matrices.
 *
 * The semantics of graphalg operations are defined over dense matrices. When
 * lowering to IPR however, it is preferable to work with sparse matrices, as
 * graphs are often sparsely connected. The special property of sparse matrices
 * is that some positions do not store any value: They have an implicit 'zero'
 * value. Because graphalg requires matrices to be defined over semirings, most
 * operations have correct behaviour around such missing values. For example,
 * matrix multiplication skips positions without a value, which is correct
 * because the implicit zero is an annihilator under multiplication. The rule of
 * thumb is: If doing nothing is equivalent to processing '0', no special
 * treatment is needed for an operation.
 *
 * The notable exception is \c ApplyOp, which can produce a non-zero value even
 * if the input is zero. For example: apply((x){ return x + 1; }, M).
 *
 * This pass handles such cases by adding a \c MakeDenseOp on inputs of an
 * \c ApplyOp, signalling that the lowering is only correct if the input is
 * dense. The analysis is conservative, and there may be cases where the
 * \c MakeDenseOp could be omitted. For such cases, we rely on canonicalization
 * to optimize unnecessary \c MakeDenseOp away.
 */
class GraphAlgExplicateSparsity
   : public impl::GraphAlgExplicateSparsityBase<lingodb/compiler/Dialect/graphalgExplicateSparsity> {
   using impl::GraphAlgExplicateSparsityBase<
      GraphAlgExplicateSparsity>::GraphAlgExplicateSparsityBase;

   void runOnOperation() final;
};

struct AddMatcher {
   mlir::Value lhs;
   mlir::Value rhs;
   // RHS should be negated, i.e. this is a sub rather than add.
   bool negRhs;

   bool match(mlir::Operation* op);
};

} // namespace

bool AddMatcher::match(mlir::Operation* op) {
   if (auto addOp = llvm::dyn_cast<AddOp>(op)) {
      lhs = addOp.getLhs();
      rhs = addOp.getRhs();
      negRhs = false;
   } else if (auto subOp = llvm::dyn_cast<mlir::arith::SubIOp>(op)) {
      lhs = subOp.getLhs();
      rhs = subOp.getRhs();
      negRhs = true;
   } else if (auto subOp = llvm::dyn_cast<mlir::arith::SubFOp>(op)) {
      lhs = subOp.getLhs();
      rhs = subOp.getRhs();
      negRhs = true;
   } else {
      return false;
   }

   return true;
}

// Clone \c op partially so that it returns \c target.
static mlir::FailureOr<ApplyOp> cloneUpTo(ApplyOp op, mlir::Value target,
                                          bool negateTarget,
                                          mlir::PatternRewriter& rewriter) {
   auto newOp = rewriter.create<ApplyOp>(
      op->getLoc(), op.getType().withSemiring(target.getType()),
      op.getInputs());
   auto& newBody = newOp.createBody();

   // Map block args
   mlir::IRMapping mapping;
   auto& body = op.getBody().front();
   mapping.map(body.getArguments(), newBody.getArguments());

   mlir::OpBuilder::InsertionGuard guard(rewriter);
   rewriter.setInsertionPointToStart(&newBody);
   mlir::Value newTarget;
   for (mlir::Operation& op : body) {
      if (auto found = mapping.lookupOrNull(target)) {
         // Before cloning the op, check if we have the target value
         // available, either from a block argument or from a previously
         // cloned op.
         newTarget = found;

         // We already have the target value we were looking for, so we can
         // skip cloning the remaining ops.
         break;
      }

      auto* newOp = rewriter.clone(op, mapping);
      mapping.map(&op, newOp);
   }

   if (!newTarget) {
      op->emitOpError("attempt to clone until value")
         << target << ", but it is not produced by this op";
   }

   if (negateTarget) {
      auto sring = llvm::cast<SemiringTypeInterface>(newTarget.getType());
      auto zeroOp =
         rewriter.create<ConstantOp>(op->getLoc(), sring.addIdentity());
      auto subOp = createScalarOpFor(op->getLoc(), BinaryOp::SUB, zeroOp,
                                     newTarget, rewriter);
      if (mlir::failed(subOp)) {
         return mlir::failure();
      }

      newTarget = *subOp;
   }

   rewriter.create<ApplyReturnOp>(op.getLoc(), newTarget);
   return newOp;
}

static mlir::LogicalResult
extractElementWiseAdd(ApplyOp op, mlir::PatternRewriter& rewriter) {
   auto returnOp =
      llvm::cast<ApplyReturnOp>(op.getBody().front().getTerminator());
   AddMatcher matcher;
   if (!mlir::matchPattern(returnOp.getValue(), matcher)) {
      return mlir::failure();
   }

   // Not useful to extract if one of the two sides is a constant.
   // Skipping that also avoids recursively calling this on a negated RHS.
   if (mlir::matchPattern(matcher.lhs, mlir::m_Constant()) ||
       mlir::matchPattern(matcher.rhs, mlir::m_Constant())) {
      return mlir::failure();
   }

   auto lhs = cloneUpTo(op, matcher.lhs, false, rewriter);
   if (mlir::failed(lhs)) {
      return mlir::failure();
   }

   auto rhs = cloneUpTo(op, matcher.rhs, matcher.negRhs, rewriter);
   if (mlir::failed(rhs)) {
      return mlir::failure();
   }

   rewriter.replaceOpWithNewOp<ElementWiseAddOp>(op, *lhs, *rhs);
   return mlir::success();
}

static MakeDenseOp makeDense(ApplyOp op, mlir::OpOperand& operand,
                             mlir::IRRewriter& rewriter) {
   auto input = operand.get();
   auto denseOp = rewriter.create<MakeDenseOp>(op->getLoc(), input);
   rewriter.cloneRegionBefore(op.getBody(), denseOp.getBody(),
                              denseOp.getBody().begin());

   // Set the input for this operand's arg to the additive identity.
   auto argNum = operand.getOperandNumber();
   auto& body = denseOp.getBody().front();
   auto arg = body.getArgument(argNum);
   auto argRing = llvm::cast<SemiringTypeInterface>(arg.getType());

   mlir::OpBuilder::InsertionGuard guard(rewriter);
   rewriter.setInsertionPointToStart(&body);
   auto zeroOp =
      rewriter.create<ConstantOp>(op->getLoc(), argRing.addIdentity());
   rewriter.replaceAllUsesWith(arg, zeroOp);

   // Replace the ApplyReturnOp with MakeDenseReturnOp.
   auto returnOp = llvm::cast<ApplyReturnOp>(body.getTerminator());
   rewriter.setInsertionPoint(returnOp);
   rewriter.replaceOpWithNewOp<MakeDenseReturnOp>(returnOp, returnOp.getValue());

   return denseOp;
}

static void makeInputsDense(ApplyOp op, mlir::IRRewriter& rewriter,
                            RunDenseAnalysis& denseAnalysis) {
   llvm::SmallVector<mlir::Value> denseInputs;
   for (auto& input : op->getOpOperands()) {
      mlir::Value denseInput;
      if (denseAnalysis.getFor(input.get())->isDense()) {
         // Already dense
         denseInput = input.get();
      } else {
         denseInput = makeDense(op, input, rewriter);
      }

      denseInputs.emplace_back(denseInput);
   }

   rewriter.modifyOpInPlace(
      op, [&]() { op.getInputsMutable().assign(denseInputs); });
}

static bool mayHaveSparseInputs(ApplyOp op,
                                const RunDenseAnalysis& denseAnalysis) {
   for (auto input : op.getInputs()) {
      if (!denseAnalysis.getFor(input)->isDense()) {
         return true;
      }
   }

   return false;
}

void GraphAlgExplicateSparsity::runOnOperation() {
   // Apply patterns that avoid making inputs dense in special cases.
   mlir::RewritePatternSet patterns(&getContext());
   patterns.add(extractElementWiseAdd);

   if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      getOperation()->emitError("failed to make sparsity explicit");
      signalPassFailure();
   }

   // Run the dense analysis
   RunDenseAnalysis denseAnalysis(getOperation());
   if (mlir::failed(denseAnalysis.status())) {
      return signalPassFailure();
   }

   // Add MakeDenseOp on potentially sparse inputs of every ApplyOp.
   //
   // NOTE: We manage our own rewriting here rather than using
   // mlir::applyPatternsAndFoldGreedily. Exhaustive pattern matching risks
   // adding MakeDenseOp on the inputs multiple times, especially because the
   // MakeDenseOp can sometimes be folded away immediately.
   llvm::SmallVector<ApplyOp> applyOps;
   getOperation()->walk([&](ApplyOp op) {
      if (mayHaveSparseInputs(op, denseAnalysis)) {
         applyOps.emplace_back(op);
      }
   });

   mlir::IRRewriter rewriter(&getContext());
   for (auto op : applyOps) {
      rewriter.setInsertionPoint(op);
      makeInputsDense(op, rewriter, denseAnalysis);
   }
}

} // namespace graphalg
