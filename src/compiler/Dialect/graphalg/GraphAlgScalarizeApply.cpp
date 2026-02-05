#include <optional>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGSCALARIZEAPPLY
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

/**
 * Replaces ApplyInlineOp with ApplyOp, changing the body arguments from scalar
 * matrices to plain scalars.
 * The ops in the body are replaced with their scalar equivalents.
 */
class GraphAlgScalarizeApply
   : public impl::GraphAlgScalarizeApplyBase<GraphAlgScalarizeApply> {
   public:
   using impl::GraphAlgScalarizeApplyBase<
      GraphAlgScalarizeApply>::GraphAlgScalarizeApplyBase;

   void runOnOperation() final;
};

template <typename T>
class OpConversion : public mlir::OpConversionPattern<T> {
   using mlir::OpConversionPattern<T>::OpConversionPattern;

   mlir::LogicalResult
   matchAndRewrite(T op,
                   typename mlir::OpConversionPattern<T>::OpAdaptor adaptor,
                   mlir::ConversionPatternRewriter& rewriter) const override;
};

class ScalarMatrixToScalarTypeConverter : public mlir::TypeConverter {
   public:
   ScalarMatrixToScalarTypeConverter();
};

// Special type converter for ApplyInlineOp that allows keeping the inputs and
// results (non-scalar matrices) as-is.
class ApplyInlineTypeConverter : public mlir::TypeConverter {
   public:
   ApplyInlineTypeConverter();
};

} // namespace

template <>
mlir::LogicalResult OpConversion<ApplyInlineOp>::matchAndRewrite(
   ApplyInlineOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   // ApplyInlineOp over scalar matrices should have been inlined in earlier
   // passes. The type converter would mess with their input/output types, so
   // refuse to process them.
   if (op.getType().isScalar()) {
      return op->emitOpError("operates over scalar inputs:")
         << " Should have already been inlined";
   }

   // NOTE: calling the auto-generated builder which does not populate the body
   // region.
   auto newOp = rewriter.create<ApplyOp>(op->getLoc(), mlir::Type(op.getType()),
                                         op.getInputs());
   rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody().begin());
   if (mlir::failed(
          rewriter.convertRegionTypes(&newOp.getBody(), *typeConverter))) {
      return op->emitOpError("has a body whose arguments cannot be converted");
   }

   rewriter.replaceOp(op, newOp);
   return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<ApplyInlineReturnOp>::matchAndRewrite(
   ApplyInlineReturnOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   rewriter.replaceOpWithNewOp<ApplyReturnOp>(op, adaptor.getValue());
   return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<LiteralOp>::matchAndRewrite(
   LiteralOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getValue());
   return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<ApplyOp>::matchAndRewrite(
   ApplyOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   auto& body = op.getBody().front();
   mlir::IRMapping mapping;

   mapping.map(body.getArguments(), adaptor.getInputs());
   mlir::Value returnValue;
   for (mlir::Operation& op : body) {
      if (auto returnOp = llvm::dyn_cast<ApplyReturnOp>(op)) {
         returnValue = mapping.lookup(returnOp.getValue());
         break;
      }

      auto* newOp = rewriter.clone(op, mapping);
      mapping.map(&op, newOp);
   }

   assert(!!returnValue && "Missing return op in ApplyOp body");
   rewriter.replaceOp(op, returnValue);
   return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<CastOp>::matchAndRewrite(
   CastOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   rewriter.replaceOpWithNewOp<CastScalarOp>(op, op.getType().getSemiring(),
                                             adaptor.getInput());
   return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<ConstantMatrixOp>::matchAndRewrite(
   ConstantMatrixOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getValue());
   return mlir::success();
}

// Unrolls a single iteration of the loop body.
static mlir::LogicalResult
unrollLoopBody(mlir::Operation* loopOp, mlir::Block& body,
               const llvm::APInt& iter, mlir::ValueRange iterArgs,
               llvm::SmallVectorImpl<mlir::Value>& results,
               mlir::OpBuilder& builder) {
   mlir::IRMapping mapping;

   // Map iter var to a constant.
   auto zeroOp = builder.create<ConstantOp>(
      loopOp->getLoc(), mlir::IntegerAttr::get(builder.getI64Type(), iter));
   mapping.map(body.getArgument(0), zeroOp);

   // Map the remainder of the body arguments to the init args.
   for (auto [initArg, blockArg] :
        llvm::zip_equal(iterArgs, body.getArguments().drop_front())) {
      mapping.map(blockArg, initArg);
   }

   // Clone the ops in the body until we reach YieldOp.
   for (auto& oldOp : body) {
      if (auto yieldOp = llvm::dyn_cast<YieldOp>(oldOp)) {
         // Collect the result values from the yield.
         results.clear();
         for (auto val : yieldOp.getInputs()) {
            results.emplace_back(mapping.lookup(val));
         }

         return mlir::success();
      }

      auto newOp = builder.clone(oldOp, mapping);
      mapping.map(&oldOp, newOp);
   }

   return loopOp->emitOpError("does not have a terminator ")
      << YieldOp::getOperationName() << ", so cannot be inlined";
}

static mlir::LogicalResult unrollForDimOne(ForDimOp op,
                                           mlir::PatternRewriter& rewriter) {
   // Well-formed graphalg programs do not use dimension symbols inside
   // functions that do not have this dimension symbol as an argument (but this
   // is not currently enforced by verifiers). Inside of an apply all
   // dimensions are one, so this should cover all well-formed programs.
   if (!op.getDim().isOne()) {
      // We need a constant size to be able to do loop unrolling.
      return op->emitOpError("is contained in an ")
         << ApplyInlineOp::getOperationName()
         << ", but contains a non-1 dimension symbol reference";
   }

   // Inline the body
   llvm::APInt iter(64, std::int64_t(0));
   llvm::SmallVector<mlir::Value> resultValues;
   if (mlir::failed(unrollLoopBody(op, op.getBody().front(), iter,
                                   op.getInitArgs(), resultValues, rewriter))) {
      return mlir::failure();
   }

   rewriter.replaceOp(op, resultValues);
   return mlir::success();
}

static std::optional<llvm::APInt> tryGetConstantRangeValue(mlir::Value v) {
   auto litOp = v.getDefiningOp<LiteralOp>();
   if (!litOp) {
      return std::nullopt;
   }

   if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(litOp.getValue())) {
      return intAttr.getValue();
   }

   return std::nullopt;
}

static mlir::LogicalResult unrollForConst(ForConstOp op,
                                          mlir::PatternRewriter& rewriter) {
   auto begin = tryGetConstantRangeValue(op.getRangeBegin());
   auto end = tryGetConstantRangeValue(op.getRangeEnd());
   if (!begin || !end) {
      // TODO: Handle loops where the bound is not yet constant.
      // Ranges should be constant in graphalg, but that constant may be
      // provided by the caller of a function, so we could encounter
      // \c mlir::BlockArgument here in a valid program.
      return op->emitOpError("does not have a constant range")
         << ", so cannot be unrolled";
   }

   llvm::SmallVector<mlir::Value> iterArgs(op.getInitArgs());

   auto one = llvm::APInt(begin->getBitWidth(), 1);
   for (auto i = *begin; i.slt(*end); i = i + one) {
      llvm::SmallVector<mlir::Value> results;
      if (mlir::failed(unrollLoopBody(op, op.getBody().front(), i, iterArgs,
                                      results, rewriter))) {
         return mlir::failure();
      }

      // Results of the last iteration are the inputs for the next iteration.
      iterArgs = results;
   }

   rewriter.replaceOp(op, iterArgs);
   return mlir::success();
}

mlir::FailureOr<mlir::Value> createScalarOpFor(mlir::Location loc, BinaryOp op,
                                               mlir::Value lhs, mlir::Value rhs,
                                               mlir::OpBuilder& builder) {
   assert(lhs.getType() == rhs.getType());
   auto type = lhs.getType();

   switch (op) {
      case BinaryOp::ADD:
         return mlir::Value(builder.create<AddOp>(loc, lhs, rhs));
      case BinaryOp::SUB:
         if (type.isIntOrIndex()) {
            return mlir::Value(builder.create<mlir::arith::SubIOp>(loc, lhs, rhs));
         } else if (type.isF64()) {
            return mlir::Value(builder.create<mlir::arith::SubFOp>(loc, lhs, rhs));
         } else {
            return mlir::emitError(loc)
               << "subtraction is not defined for values of type " << type;
         }
      case BinaryOp::MUL:
         return mlir::Value(builder.create<MulOp>(loc, lhs, rhs));
      case BinaryOp::DIV:
         if (type.isF64()) {
            return mlir::Value(builder.create<mlir::arith::DivFOp>(loc, lhs, rhs));
         } else {
            return mlir::emitError(loc)
               << "division is not defined for values of type " << type;
         }
      case BinaryOp::EQ:
         return mlir::Value(builder.create<EqOp>(loc, lhs, rhs));
      case BinaryOp::NE: {
         auto falseOp = builder.create<ConstantOp>(loc, builder.getBoolAttr(false));
         auto eqOp = builder.create<EqOp>(loc, lhs, rhs);
         // Compare with false to invert the condition.
         return mlir::Value(builder.create<EqOp>(loc, falseOp, eqOp));
      }
      case BinaryOp::LT:
      case BinaryOp::GT:
      case BinaryOp::LE:
      case BinaryOp::GE:
         return mlir::emitError(loc)
            << "operator " << stringifyBinaryOp(op) << " is not yet supported";
   }
}

template <>
mlir::LogicalResult OpConversion<ElementWiseOp>::matchAndRewrite(
   ElementWiseOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   auto scalarOp = createScalarOpFor(op->getLoc(), op.getOp(), adaptor.getLhs(),
                                     adaptor.getRhs(), rewriter);
   if (mlir::failed(scalarOp)) {
      return mlir::failure();
   }

   rewriter.replaceOp(op, *scalarOp);
   return mlir::success();
}

static mlir::Value applyMask(mlir::TypedValue<MatrixType> mask, bool complement,
                             mlir::TypedValue<MatrixType> value,
                             mlir::OpBuilder& builder) {
   if (complement) {
      mask = builder.create<NotOp>(mask.getLoc(), mask);
   }

   auto castOp = builder.create<CastOp>(mask.getLoc(), value.getType(), mask);
   return builder.create<ElementWiseOp>(mask.getLoc(), castOp, BinaryOp::MUL,
                                        value);
}

static mlir::LogicalResult convertMask(MaskOp op,
                                       mlir::PatternRewriter& rewriter) {
   // Apply mask to the two sides.
   auto base =
      applyMask(op.getMask(), !op.getComplement(), op.getBase(), rewriter);
   auto input =
      applyMask(op.getMask(), op.getComplement(), op.getInput(), rewriter);

   rewriter.replaceOpWithNewOp<ElementWiseOp>(op, base, BinaryOp::ADD, input);
   return mlir::success();
}

static mlir::LogicalResult convertMatMul(MatMulOp op,
                                         mlir::PatternRewriter& rewriter) {
   // Scalar matrix multiply is equal to element-wise multiplication.
   if (!op.getType().isScalar()) {
      return mlir::failure();
   }

   rewriter.replaceOpWithNewOp<ElementWiseOp>(op, op.getLhs(), BinaryOp::MUL,
                                              op.getRhs());
   return mlir::success();
}

static mlir::LogicalResult convertNeg(NegOp op,
                                      mlir::PatternRewriter& rewriter) {
   auto sring = llvm::cast<SemiringTypeInterface>(op.getType().getSemiring());
   auto zeroOp = rewriter.create<LiteralOp>(op->getLoc(), sring.addIdentity());
   rewriter.replaceOpWithNewOp<ElementWiseOp>(op, zeroOp, BinaryOp::SUB,
                                              op.getInput());
   return mlir::success();
}

static mlir::LogicalResult convertNot(NotOp op,
                                      mlir::PatternRewriter& rewriter) {
   auto falseOp =
      rewriter.create<LiteralOp>(op->getLoc(), rewriter.getBoolAttr(false));
   rewriter.replaceOpWithNewOp<ElementWiseOp>(op, falseOp, BinaryOp::EQ,
                                              op.getInput());
   return mlir::success();
}

static mlir::LogicalResult convertNVals(NValsOp op,
                                        mlir::PatternRewriter& rewriter) {
   // If the matrix is scalar, there is only one value to consider.
   if (!op.getType().isScalar()) {
      return mlir::failure();
   }

   // 1 if value != 0, or 0 otherwise.
   auto sring =
      llvm::cast<SemiringTypeInterface>(op.getInput().getType().getSemiring());
   auto zeroOp = rewriter.create<LiteralOp>(op->getLoc(), sring.addIdentity());
   auto neOp = rewriter.create<ElementWiseOp>(op->getLoc(), zeroOp, BinaryOp::NE,
                                              op.getInput());
   rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), neOp);
   return mlir::success();
}

ScalarMatrixToScalarTypeConverter::ScalarMatrixToScalarTypeConverter() {
   addConversion([](MatrixType matrix) -> std::optional<mlir::Type> {
      if (matrix.isScalar()) {
         return matrix.getSemiring();
      }

      return std::nullopt;
   });
}

ApplyInlineTypeConverter::ApplyInlineTypeConverter() {
   addConversion([](MatrixType matrix) -> mlir::Type {
      if (matrix.isScalar()) {
         // For block arguments.
         return matrix.getSemiring();
      }

      // For op inputs and results.
      return matrix;
   });

   // The type converter is called after we convert ApplyInlineOp to check that
   // the argument types of the block are valid. Accept any scalar type that is
   // a semiring.
   addConversion([](SemiringTypeInterface ring) { return ring; });
}

static bool isScalarMatrix(mlir::Type t) {
   if (auto matrix = llvm::dyn_cast<MatrixType>(t)) {
      if (matrix.isScalar()) {
         return true;
      }
   }

   return false;
}

static bool doesNotUseScalarMatrices(mlir::Operation* op) {
   for (auto t : op->getOperandTypes()) {
      if (isScalarMatrix(t)) {
         return false;
      }
   }

   for (auto t : op->getResultTypes()) {
      if (isScalarMatrix(t)) {
         return false;
      }
   }

   return true;
}

void GraphAlgScalarizeApply::runOnOperation() {
   // Eliminate all uses of scalar matrices, to be replaced with operations
   // over plain scalars.
   mlir::ConversionTarget target(getContext());
   target.addDynamicallyLegalDialect<GraphAlgDialect>(doesNotUseScalarMatrices);
   target.addIllegalOp<ApplyInlineOp>();
   // We use scalar ops for subtraction and division defined here.
   target.addLegalDialect<mlir::arith::ArithDialect>();

   // Conversions and simplications to remove matrix-based ops.
   mlir::RewritePatternSet conversions(&getContext());
   ScalarMatrixToScalarTypeConverter typeConverter;
   // Conversions from scalar matrix -> scalar
   conversions.add<OpConversion<ApplyInlineReturnOp>, OpConversion<CastOp>,
                   OpConversion<ConstantMatrixOp>, OpConversion<ElementWiseOp>,
                   OpConversion<LiteralOp>, OpConversion<ApplyOp>>(
      typeConverter, &getContext());
   // Simplifications into ops that can be converted using rules above.
   conversions.add(unrollForDimOne);
   conversions.add(unrollForConst);
   conversions.add(convertMask);
   conversions.add(convertMatMul);
   conversions.add(convertNeg);
   conversions.add(convertNot);
   conversions.add(convertNVals);
   // Converts the outer ApplyInlineOp
   ApplyInlineTypeConverter applyTypeConverter;
   conversions.add<OpConversion<ApplyInlineOp>>(applyTypeConverter,
                                                &getContext());

   // Convert only ApplyInlineOp and their bodies.
   llvm::SmallVector<ApplyInlineOp> applyOps;
   getOperation().walk([&](ApplyInlineOp op) { applyOps.emplace_back(op); });
   mlir::FrozenRewritePatternSet frozenConversions(std::move(conversions));
   for (auto op : applyOps) {
      if (mlir::failed(
             mlir::applyFullConversion(op, target, frozenConversions))) {
         op->emitOpError("could not be converted");
         return signalPassFailure();
      }
   }
}

} // namespace graphalg
