#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

static Value convertBooleanCondition(mlir::Location loc, mlir::OpBuilder& rewriter, Type t, Value v) {
   auto nullableType=t.dyn_cast_or_null<db::NullableType>();
   if (nullableType && isIntegerType(nullableType.getType(),1)) {
      auto i1Type = rewriter.getI1Type();
      auto unpacked = rewriter.create<util::UnPackOp>(loc, v);
      Value constTrue = rewriter.create<arith::ConstantOp>(loc, i1Type, rewriter.getIntegerAttr(i1Type, 1));
      auto negated = rewriter.create<arith::XOrIOp>(loc, unpacked.getResult(0), constTrue); //negate
      return rewriter.create<arith::AndIOp>(loc, i1Type, negated, unpacked.getResult(1));
   } else {
      return v;
   }
}
//lower dbexec::If to scf::If
class IfLowering : public ConversionPattern {
   public:
   explicit IfLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::IfOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto ifOp = cast<mlir::db::IfOp>(op);
      mlir::db::IfOpAdaptor ifOpAdaptor(operands);
      auto loc = op->getLoc();
      std::vector<Type> resultTypes;
      for (auto res : ifOp.results()) {
         resultTypes.push_back(typeConverter->convertType(res.getType()));
      }
      Value condition = convertBooleanCondition(loc, rewriter, ifOp.condition().getType(), ifOpAdaptor.condition());

      auto newIfOp = rewriter.create<mlir::scf::IfOp>(loc, TypeRange(resultTypes), condition, !ifOp.elseRegion().empty());
      {
         scf::IfOp::ensureTerminator(newIfOp.getThenRegion(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newIfOp.getThenRegion().front());
         Block* originalThenBlock = &ifOp.thenRegion().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalThenBlock, terminator, {});
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      if (!ifOp.elseRegion().empty()) {
         scf::IfOp::ensureTerminator(newIfOp.getElseRegion(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newIfOp.getElseRegion().front());
         Block* originalElseBlock = &ifOp.elseRegion().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalElseBlock, terminator, {});
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }

      rewriter.replaceOp(ifOp, newIfOp.getResults());

      return success();
   }
};
class YieldLowering : public ConversionPattern {
   public:
   explicit YieldLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::YieldOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
      return success();
   }
};
class DeriveTruthLowering : public ConversionPattern {
   public:
   explicit DeriveTruthLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DeriveTruth::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      db::DeriveTruthAdaptor adaptor(operands);
      db::DeriveTruth deriveTruth = cast<db::DeriveTruth>(op);
      rewriter.replaceOp(op,convertBooleanCondition(deriveTruth.getLoc(), rewriter, deriveTruth.val().getType(), adaptor.val()));
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
   patterns.insert<IfLowering>(typeConverter, patterns.getContext());
   patterns.insert<YieldLowering>(typeConverter, patterns.getContext());
   patterns.insert<CondSkipTypeConversion>(typeConverter, patterns.getContext());
   patterns.insert<DeriveTruthLowering>(typeConverter, patterns.getContext());
}