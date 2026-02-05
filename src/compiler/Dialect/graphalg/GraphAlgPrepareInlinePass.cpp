#include <llvm/ADT/SCCIterator.h>
#include <mlir/Analysis/CallGraph.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGPREPAREINLINE
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

/**
 * Prepares the IR for function inlining.
 *
 * This pass first verifies that there are no recursive calls which would
 * prevent exhaustive inlining.
 * Next, it transforms all variants of \c apply_* and \c select_* to a
 * \c ApplyInlineOp containing a \c CallOp that is ready to be inlined.
 *
 * The actual inlining should be performed using the standard \c Inline pass.
 */
class GraphAlgPrepareInline
   : public impl::GraphAlgPrepareInlineBase<GraphAlgPrepareInline> {
   private:
   mlir::LogicalResult verifyAcyclic(const mlir::CallGraph& graph);

   public:
   using impl::GraphAlgPrepareInlineBase<
      GraphAlgPrepareInline>::GraphAlgPrepareInlineBase;

   void runOnOperation() final;
};

/*
 * Generic container for the properties of:
 * - ApplyUnaryOp
 * - ApplyBinaryOp
 * - ApplyElementWiseOp
 * - SelectUnaryOp
 * - SelectBinaryOp
 */
struct ApplyAdaptor {
   mlir::Location loc;
   bool isSelect;
   MatrixType resultType;
   llvm::StringRef func;
   mlir::TypedValue<MatrixType> lhs;
   // Note: nullptr if unary
   mlir::TypedValue<MatrixType> rhs;

   llvm::SmallVector<mlir::Value, 2> inputs() {
      if (rhs) {
         return {lhs, rhs};
      } else {
         return {lhs};
      }
   }

   mlir::Value populateApply(mlir::ValueRange bodyArgs,
                             mlir::PatternRewriter& rewriter);
   mlir::LogicalResult replaceWithApply(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter);
};

} // namespace

mlir::Value ApplyAdaptor::populateApply(mlir::ValueRange bodyArgs,
                                        mlir::PatternRewriter& rewriter) {
   if (isSelect) {
      // (a : R, b) { c = f(a, b); return cast<R>(c) * a }
      auto callOp = rewriter.create<mlir::func::CallOp>(
         loc, func,
         MatrixType::scalarOf(SemiringTypes::forBool(rewriter.getContext())),
         bodyArgs);

      auto arg0 = bodyArgs[0];
      auto castOp =
         rewriter.create<CastOp>(loc, arg0.getType(), callOp->getResult(0));

      auto mulOp =
         rewriter.create<ElementWiseOp>(loc, castOp, BinaryOp::MUL, arg0);

      return mulOp;
   } else {
      // (a, b) { f(a, b) }
      auto callOp = rewriter.create<mlir::func::CallOp>(
         loc, func, resultType.asScalar(), bodyArgs);
      return callOp->getResult(0);
   }
}

mlir::LogicalResult
ApplyAdaptor::replaceWithApply(mlir::Operation* op,
                               mlir::PatternRewriter& rewriter) {
   if (resultType.isScalar()) {
      // Apply over scalar inputs is redundant.
      // Skip it, and write the apply body directly into the current block.
      auto result = populateApply(inputs(), rewriter);
      rewriter.replaceOp(op, result);
      return mlir::success();
   }

   if (rhs) {
      // If there are 2 inputs, handle different cases.
      auto lhsType = lhs.getType();
      auto rhsType = rhs.getType();

      if (rhsType.isScalar()) {
         // For ApplyBinaryOp: broadcast scalar to match LHS dimensions.
         rhsType = lhsType.withSemiring(rhsType.getSemiring());
         rhs = rewriter.create<BroadcastOp>(loc, rhsType, rhs);
      } else {
         // For ApplyElementWiseOp: both inputs already have matching
         // dimensions, no broadcasting needed.
         assert(lhsType.getDims() == rhsType.getDims());
      }
   }

   auto applyOp = rewriter.replaceOpWithNewOp<ApplyInlineOp>(
      op, inputs(), resultType.getSemiring());

   auto& body = applyOp.getBody().front();
   rewriter.setInsertionPointToStart(&body);

   auto result = populateApply(body.getArguments(), rewriter);
   rewriter.create<ApplyInlineReturnOp>(loc, result);
   return mlir::success();
}

static mlir::LogicalResult convertApplyUnary(ApplyUnaryOp op,
                                             mlir::PatternRewriter& rewriter) {
   return ApplyAdaptor{
      .loc = op->getLoc(),
      .isSelect = false,
      .resultType = op.getType(),
      .func = op.getFunc(),
      .lhs = op.getInput(),
   }
      .replaceWithApply(op, rewriter);
}

static mlir::LogicalResult convertApplyBinary(ApplyBinaryOp op,
                                              mlir::PatternRewriter& rewriter) {
   return ApplyAdaptor{
      .loc = op->getLoc(),
      .isSelect = false,
      .resultType = op.getType(),
      .func = op.getFunc(),
      .lhs = op.getLhs(),
      .rhs = op.getRhs(),
   }
      .replaceWithApply(op, rewriter);
}

static mlir::LogicalResult convertSelectUnary(SelectUnaryOp op,
                                              mlir::PatternRewriter& rewriter) {
   return ApplyAdaptor{
      .loc = op->getLoc(),
      .isSelect = true,
      .resultType = op.getType(),
      .func = op.getFunc(),
      .lhs = op.getInput(),
   }
      .replaceWithApply(op, rewriter);
}

static mlir::LogicalResult
convertSelectBinary(SelectBinaryOp op, mlir::PatternRewriter& rewriter) {
   return ApplyAdaptor{
      .loc = op->getLoc(),
      .isSelect = true,
      .resultType = op.getType(),
      .func = op.getFunc(),
      .lhs = op.getLhs(),
      .rhs = op.getRhs(),
   }
      .replaceWithApply(op, rewriter);
}

static mlir::LogicalResult
convertApplyElementWise(ApplyElementWiseOp op,
                        mlir::PatternRewriter& rewriter) {
   return ApplyAdaptor{
      .loc = op->getLoc(),
      .isSelect = false,
      .resultType = op.getType(),
      .func = op.getFunc(),
      .lhs = op.getLhs(),
      .rhs = op.getRhs(),
   }
      .replaceWithApply(op, rewriter);
}

mlir::LogicalResult
GraphAlgPrepareInline::verifyAcyclic(const mlir::CallGraph& graph) {
   // If a cycle exists, it must be contained within a strongly connected
   // component, since nodes forming a cycle are strongly connected.
   // By doing SCC analysis first we partition the graph, allowing for more
   // efficient cycle detection.
   for (auto cgi = llvm::scc_begin(&graph); !cgi.isAtEnd(); ++cgi) {
      if (!cgi.hasCycle()) {
         continue;
      }

      llvm::SmallVector<mlir::Location> locs;
      llvm::SmallVector<mlir::func::FuncOp> calledOps;
      const auto& cycle = *cgi;
      for (auto node : cycle) {
         auto funcOp =
            node->getCallableRegion()->getParentOfType<mlir::func::FuncOp>();
         if (!funcOp) {
            return node->getCallableRegion()->getParentOp()->emitOpError(
                      "is part of a function call cycle, but is not a ")
               << mlir::func::FuncOp::getOperationName();
         }

         calledOps.emplace_back(funcOp);
         locs.emplace_back(funcOp->getLoc());
      }

      auto report = mlir::emitError(mlir::FusedLoc::get(&getContext(), locs))
         << "The program contains a cycle";
      for (auto op : calledOps) {
         report.attachNote(op->getLoc()) << "Part of the cycle";
      }

      return report;
   }

   return mlir::success();
}

void GraphAlgPrepareInline::runOnOperation() {
   // Verify that the call graph is acyclic, so we can safely inline
   // everything.
   const auto& callGraph = getAnalysis<mlir::CallGraph>();
   if (mlir::failed(verifyAcyclic(callGraph))) {
      return signalPassFailure();
   }

   mlir::ConversionTarget target(getContext());
   target.addLegalDialect<GraphAlgDialect>();
   target.addLegalDialect<mlir::func::FuncDialect>();
   // Rewrite all variants of apply_* and select_* to ApplyInlineOp.
   target.addIllegalOp<ApplyUnaryOp, ApplyBinaryOp, ApplyElementWiseOp,
                       SelectUnaryOp, SelectBinaryOp>();
   mlir::RewritePatternSet patterns(&getContext());
   patterns.add(convertApplyUnary);
   patterns.add(convertApplyBinary);
   patterns.add(convertApplyElementWise);
   patterns.add(convertSelectUnary);
   patterns.add(convertSelectBinary);

   if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                 std::move(patterns)))) {
      return signalPassFailure();
   }
}

} // namespace graphalg
