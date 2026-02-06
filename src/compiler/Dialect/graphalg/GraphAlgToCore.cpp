#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

#include "lingodb/compiler/Conversion/RelAlgToSubOp/OrderedAttributes.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/runtime/ExternalDataSourceProperty.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGTOCORE
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

class GraphAlgToCore : public impl::GraphAlgToCoreBase<GraphAlgToCore> {
   public:
   using impl::GraphAlgToCoreBase<GraphAlgToCore>::GraphAlgToCoreBase;

   void runOnOperation() final;
};

} // namespace

static mlir::LogicalResult convertVecMatMul(VecMatMulOp op,
                                            mlir::PatternRewriter& rewriter) {
   // For lhs : Matrix<s1, 1, T> and rhs : Matrix<s1, s2, T>
   // lhs * rhs => rhs.T * lhs
   auto rhsT = rewriter.create<TransposeOp>(op->getLoc(), op.getRhs());
   rewriter.replaceOpWithNewOp<MatMulOp>(op, rhsT, op.getLhs());
   return mlir::success();
}

static mlir::LogicalResult convertElementWise(ElementWiseOp op,
                                              mlir::PatternRewriter& rewriter) {
   // Wrap in Apply to make the arguments scalar.
   auto applyOp =
      rewriter.replaceOpWithNewOp<ApplyOp>(op, op.getType(), op->getOperands());

   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto& body = applyOp.createBody();
      rewriter.setInsertionPointToStart(&body);

      assert(body.getNumArguments() == 2);
      auto lhs = body.getArgument(0);
      auto rhs = body.getArgument(1);
      auto scalarOp =
         createScalarOpFor(op->getLoc(), op.getOp(), lhs, rhs, rewriter);
      if (mlir::failed(scalarOp)) {
         return mlir::failure();
      }

      rewriter.create<ApplyReturnOp>(op->getLoc(), *scalarOp);
   }

   return mlir::success();
}

static mlir::LogicalResult convertCast(CastOp op,
                                       mlir::PatternRewriter& rewriter) {
   // Apply scalar casting to each element of the matrix.
   auto applyOp =
      rewriter.replaceOpWithNewOp<ApplyOp>(op, op.getType(), op.getInput());

   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto& body = applyOp.createBody();
      rewriter.setInsertionPointToStart(&body);

      auto inputArg = body.getArgument(0);
      auto outputType = op.getType().getSemiring();
      auto castOp =
         rewriter.create<CastScalarOp>(op->getLoc(), outputType, inputArg);
      rewriter.create<ApplyReturnOp>(op->getLoc(), castOp);
   }

   return mlir::success();
}

static mlir::LogicalResult convertNVals(NValsOp op,
                                        mlir::PatternRewriter& rewriter) {
   // e.nvals => reduce(apply((a){ cast<int>(a != zero) }, e))

   // The semiring used for counting
   auto countRing = SemiringTypes::forInt(rewriter.getContext());
   auto inputType = op.getInput().getType();
   auto inputRing = llvm::cast<SemiringTypeInterface>(inputType.getSemiring());
   auto applyOp = rewriter.create<ApplyOp>(
      op->getLoc(), op.getInput().getType().withSemiring(countRing),
      mlir::ValueRange{op.getInput()});

   // (a) { cast<int>(a != zero) }
   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto& body = applyOp.createBody();
      rewriter.setInsertionPointToStart(&body);
      auto arg = body.getArgument(0);

      // (arg == zero) == false
      auto zero =
         rewriter.create<ConstantOp>(op->getLoc(), inputRing.addIdentity());
      auto eqZero = rewriter.create<EqOp>(op->getLoc(), arg, zero);
      auto falseOp =
         rewriter.create<ConstantOp>(op->getLoc(), rewriter.getBoolAttr(false));
      auto neZero = rewriter.create<EqOp>(op->getLoc(), eqZero, falseOp);

      // cast<int>
      auto castOp =
         rewriter.create<CastScalarOp>(op->getLoc(), countRing, neZero);

      rewriter.create<ApplyReturnOp>(op->getLoc(), castOp);
   }

   // reduce(..)
   rewriter.replaceOpWithNewOp<ReduceOp>(op, MatrixType::scalarOf(countRing),
                                         applyOp);
   return mlir::success();
}

static mlir::LogicalResult convertNot(NotOp op,
                                      mlir::PatternRewriter& rewriter) {
   auto applyOp =
      rewriter.replaceOpWithNewOp<ApplyOp>(op, op.getType(), op.getInput());

   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto& body = applyOp.createBody();
      rewriter.setInsertionPointToStart(&body);

      auto inputArg = body.getArgument(0);
      auto falseOp =
         rewriter.create<ConstantOp>(op->getLoc(), rewriter.getBoolAttr(false));
      auto notOp = rewriter.create<EqOp>(op->getLoc(), inputArg, falseOp);
      rewriter.create<ApplyReturnOp>(op->getLoc(), notOp);
   }

   return mlir::success();
}

static mlir::LogicalResult convertNeg(NegOp op,
                                      mlir::PatternRewriter& rewriter) {
   auto inputRing =
      llvm::cast<SemiringTypeInterface>(op.getInput().getType().getSemiring());
   auto applyOp =
      rewriter.replaceOpWithNewOp<ApplyOp>(op, op.getType(), op.getInput());

   {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto& body = applyOp.createBody();
      rewriter.setInsertionPointToStart(&body);

      auto inputArg = body.getArgument(0);
      auto zeroOp =
         rewriter.create<ConstantOp>(op->getLoc(), inputRing.addIdentity());
      auto subOp = createScalarOpFor(op->getLoc(), BinaryOp::SUB, zeroOp,
                                     inputArg, rewriter);
      if (mlir::failed(subOp)) {
         return mlir::failure();
      }

      rewriter.create<ApplyReturnOp>(op->getLoc(), *subOp);
   }

   return mlir::success();
}

// Keeps all entries from \c input where \c mask is true, resets to zero
// otherwise. If \c complement is set, the mask is inverted first.
static mlir::TypedValue<MatrixType>
applyMask(mlir::Operation* op, mlir::TypedValue<MatrixType> mask,
          mlir::TypedValue<MatrixType> input, bool complement,
          mlir::PatternRewriter& rewriter) {
   // TODO: if we need to be more flexible, we can cast to bool first.
   assert(mask.getType().isBoolean());
   auto applyOp = rewriter.create<ApplyOp>(op->getLoc(), input.getType(),
                                           mlir::ValueRange{mask, input});

   mlir::OpBuilder::InsertionGuard guard(rewriter);
   auto& body = applyOp.createBody();
   rewriter.setInsertionPointToStart(&body);
   mlir::Value maskArg = body.getArgument(0);
   auto inputArg = body.getArgument(1);

   if (complement) {
      // Simulate (!mask) with (mask == false)
      auto falseOp =
         rewriter.create<ConstantOp>(op->getLoc(), rewriter.getBoolAttr(false));
      maskArg = rewriter.create<EqOp>(op->getLoc(), maskArg, falseOp);
   }

   // cast<R>(mask) * input
   auto castOp = rewriter.create<CastScalarOp>(
      op->getLoc(), input.getType().getSemiring(), maskArg);
   auto mulOp = rewriter.create<MulOp>(op->getLoc(), castOp, inputArg);
   rewriter.create<ApplyReturnOp>(op->getLoc(), mulOp);

   return applyOp;
}

static mlir::LogicalResult convertMask(MaskOp op,
                                       mlir::PatternRewriter& rewriter) {
   auto base = op.getBase();
   auto mask = op.getMask();
   auto input = op.getInput();

   // (cast<R>(!mask) * base) + (cast<R>(mask) * input)
   auto lhs = applyMask(op, mask, base, !op.getComplement(), rewriter);
   auto rhs = applyMask(op, mask, input, op.getComplement(), rewriter);
   rewriter.replaceOpWithNewOp<ElementWiseOp>(op, lhs, BinaryOp::ADD, rhs);
   return mlir::success();
}

static mlir::LogicalResult convertTriu(TriuOp op,
                                       mlir::PatternRewriter& rewriter) {
   // triu(e) => tril(e.T).T
   auto argTrans = rewriter.create<TransposeOp>(op->getLoc(), op.getInput());
   auto trilOp = rewriter.create<TrilOp>(op->getLoc(), argTrans);
   rewriter.replaceOpWithNewOp<TransposeOp>(op, trilOp);
   return mlir::success();
}

static mlir::LogicalResult convertLiteral(LiteralOp op,
                                          mlir::PatternRewriter& rewriter) {
   rewriter.replaceOpWithNewOp<ConstantMatrixOp>(op, op.getType(),
                                                 op.getValue());
   return mlir::success();
}

void GraphAlgToCore::runOnOperation() {
   mlir::ConversionTarget target(getContext());

   // Programs are still ModuleOps
   target.addLegalOp<mlir::ModuleOp>();
   // Functions still use the FuncDialect
   target.addLegalDialect<mlir::func::FuncDialect>();
   // We use some scalar ops from ArithDialect
   target.addLegalDialect<mlir::arith::ArithDialect>();

   // copy from RelAlg to
   using namespace lingodb::compiler::dialect;
   using namespace mlir;
   target.addLegalDialect<gpu::GPUDialect>();
   target.addLegalDialect<async::AsyncDialect>();
   target.addIllegalDialect<relalg::RelAlgDialect>();
   target.addLegalDialect<subop::SubOperatorDialect>();
   target.addLegalDialect<db::DBDialect>();
   target.addLegalDialect<lingodb::compiler::dialect::arrow::ArrowDialect>();

   target.addLegalDialect<tuples::TupleStreamDialect>();
   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   target.addLegalDialect<arith::ArithDialect>();
   target.addLegalDialect<cf::ControlFlowDialect>();
   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();

   // Only graphalg ops are only allowed if they are part of the 'Core' subset.
   target.addIllegalDialect<graphalg::GraphAlgDialect>();
   target.addDynamicallyLegalDialect<graphalg::GraphAlgDialect>(
      [](mlir::Operation* op) { return op->hasTrait<IsCore>(); });

   mlir::RewritePatternSet patterns(&getContext());
   patterns.add(convertVecMatMul);
   patterns.add(convertElementWise);
   patterns.add(convertCast);
   patterns.add(convertNVals);
   patterns.add(convertNot);
   patterns.add(convertNeg);
   patterns.add(convertMask);
   patterns.add(convertTriu);
   patterns.add(convertLiteral);

   if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                              std::move(patterns)))) {
      signalPassFailure();
   }
}

} // namespace graphalg
