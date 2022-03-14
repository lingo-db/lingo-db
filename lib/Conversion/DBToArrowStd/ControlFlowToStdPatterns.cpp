#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

static Value convertBooleanCondition(mlir::Location loc, mlir::OpBuilder& rewriter, Type t, Value v) {
   auto nullableType = t.dyn_cast_or_null<db::NullableType>();
   if (nullableType && isIntegerType(nullableType.getType(), 1)) {
      auto i1Type = rewriter.getI1Type();
      auto unpacked = rewriter.create<util::UnPackOp>(loc, v);
      Value constTrue = rewriter.create<arith::ConstantOp>(loc, i1Type, rewriter.getIntegerAttr(i1Type, 1));
      auto negated = rewriter.create<arith::XOrIOp>(loc, unpacked.getResult(0), constTrue); //negate
      return rewriter.create<arith::AndIOp>(loc, i1Type, negated, unpacked.getResult(1));
   } else {
      return v;
   }
}
class DeriveTruthLowering : public ConversionPattern {
   public:
   explicit DeriveTruthLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DeriveTruth::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      db::DeriveTruthAdaptor adaptor(operands);
      db::DeriveTruth deriveTruth = cast<db::DeriveTruth>(op);
      rewriter.replaceOp(op, convertBooleanCondition(deriveTruth.getLoc(), rewriter, deriveTruth.val().getType(), adaptor.val()));
      return success();
   }
};
class CondSkipTypeConversion : public ConversionPattern {
   public:
   explicit CondSkipTypeConversion(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CondSkipOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto condskip = mlir::cast<mlir::db::CondSkipOp>(op);
      db::CondSkipOpAdaptor adaptor(operands);
      rewriter.replaceOpWithNewOp<mlir::db::CondSkipOp>(op, convertBooleanCondition(op->getLoc(), rewriter, condskip.condition().getType(), adaptor.condition()), adaptor.args());
      return success();
   }
};

} // namespace
void mlir::db::populateControlFlowToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CondSkipTypeConversion>(typeConverter, patterns.getContext());
   patterns.insert<DeriveTruthLowering>(typeConverter, patterns.getContext());
}