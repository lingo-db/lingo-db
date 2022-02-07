#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

static Value convertBooleanCondition(mlir::Location loc, mlir::OpBuilder& rewriter, Type t, Value v) {
   auto boolType = t.dyn_cast_or_null<db::BoolType>();
   if (boolType && boolType.isNullable()) {
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
class WhileLowering : public ConversionPattern {
   public:
   explicit WhileLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::WhileOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto whileOp = cast<mlir::db::WhileOp>(op);
      mlir::db::WhileOpAdaptor adaptor(operands);
      auto loc = op->getLoc();
      std::vector<Type> resultTypes;
      for (auto res : whileOp.results()) {
         resultTypes.push_back(typeConverter->convertType(res.getType()));
      }
      auto newWhileOp = rewriter.create<mlir::scf::WhileOp>(loc, TypeRange(resultTypes), adaptor.inits());
      Block* before = new Block;
      Block* after = new Block;
      newWhileOp.getBefore().push_back(before);
      newWhileOp.getAfter().push_back(after);
      for (auto t : resultTypes) {
         before->addArgument(t,loc);
         after->addArgument(t,loc);
      }

      {
         scf::IfOp::ensureTerminator(newWhileOp.getBefore(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newWhileOp.getBefore().front());
         Block* originalThenBlock = &whileOp.before().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalThenBlock, terminator, newWhileOp.getBefore().front().getArguments());
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      {
         scf::IfOp::ensureTerminator(newWhileOp.getAfter(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newWhileOp.getAfter().front());
         Block* originalElseBlock = &whileOp.after().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalElseBlock, terminator, newWhileOp.getAfter().front().getArguments());
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         whileOp.before().push_back(new Block());
         for (auto t : resultTypes) {
            whileOp.before().addArgument(t,loc);
         }
         rewriter.setInsertionPointToStart(&whileOp.before().front());
         rewriter.create<mlir::db::YieldOp>(whileOp.getLoc());
      }
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         whileOp.after().push_back(new Block());
         for (auto t : resultTypes) {
            whileOp.after().addArgument(t,loc);
         }
         rewriter.setInsertionPointToStart(&whileOp.after().front());
         rewriter.create<mlir::db::YieldOp>(whileOp.getLoc());
      }
      rewriter.replaceOp(whileOp, newWhileOp.getResults());

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
class ConditionLowering : public ConversionPattern {
   public:
   explicit ConditionLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ConditionOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      db::ConditionOpAdaptor adaptor(operands);
      db::ConditionOp conditionOp = cast<db::ConditionOp>(op);
      rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(op, convertBooleanCondition(conditionOp.getLoc(), rewriter, conditionOp.condition().getType(), adaptor.condition()), adaptor.args());
      return success();
   }
};
class SelectLowering : public ConversionPattern {
   public:
   explicit SelectLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::SelectOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      db::SelectOpAdaptor adaptor(operands);
      db::SelectOp selectOp = cast<db::SelectOp>(op);
      rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, convertBooleanCondition(selectOp.getLoc(), rewriter, selectOp.getCondition().getType(), adaptor.condition()), adaptor.true_value(), adaptor.false_value());
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
      condskip.dump();
      rewriter.replaceOpWithNewOp<mlir::db::CondSkipOp>(op, convertBooleanCondition(op->getLoc(), rewriter, condskip.condition().getType(), adaptor.condition()), adaptor.args());
      return success();
   }
};

} // namespace
void mlir::db::populateControlFlowToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<IfLowering>(typeConverter, patterns.getContext());
   patterns.insert<WhileLowering>(typeConverter, patterns.getContext());
   patterns.insert<ConditionLowering>(typeConverter, patterns.getContext());
   patterns.insert<YieldLowering>(typeConverter, patterns.getContext());
   patterns.insert<SelectLowering>(typeConverter, patterns.getContext());
   patterns.insert<CondSkipTypeConversion>(typeConverter, patterns.getContext());
}