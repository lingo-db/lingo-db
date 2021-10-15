#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {
//lower dbexec::If to scf::If
class IfLowering : public ConversionPattern {
   public:
   explicit IfLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::IfOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto ifOp = cast<mlir::db::IfOp>(op);
      auto loc = op->getLoc();
      std::vector<Type> resultTypes;
      for (auto res : ifOp.results()) {
         resultTypes.push_back(typeConverter->convertType(res.getType()));
      }
      Value condition;
      auto boolType = ifOp.condition().getType().dyn_cast_or_null<db::BoolType>();
      if (boolType && boolType.isNullable()) {
         auto i1Type = rewriter.getI1Type();
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({i1Type, i1Type}), ifOp.condition());
         Value constTrue = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), i1Type, rewriter.getIntegerAttr(i1Type, 1));
         auto negated = rewriter.create<arith::XOrIOp>(rewriter.getUnknownLoc(), unpacked.getResult(0), constTrue); //negate
         auto anded = rewriter.create<arith::AndIOp>(rewriter.getUnknownLoc(), i1Type, negated, unpacked.getResult(1));
         condition = anded;
      } else {
         condition = ifOp.condition();
      }
      auto newIfOp = rewriter.create<mlir::scf::IfOp>(loc, TypeRange(resultTypes), condition, !ifOp.elseRegion().empty());
      {
         scf::IfOp::ensureTerminator(newIfOp.thenRegion(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newIfOp.thenRegion().front());
         Block* originalThenBlock = &ifOp.thenRegion().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalThenBlock, terminator, {});
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      if (!ifOp.elseRegion().empty()) {
         scf::IfOp::ensureTerminator(newIfOp.elseRegion(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newIfOp.elseRegion().front());
         Block* originalElseBlock = &ifOp.elseRegion().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalElseBlock, terminator, {});
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }

      rewriter.replaceOp(ifOp, newIfOp.results());

      return success();
   }
};
class WhileLowering : public ConversionPattern {
   public:
   explicit WhileLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::WhileOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto whileOp = cast<mlir::db::WhileOp>(op);
      auto loc = op->getLoc();
      std::vector<Type> resultTypes;
      for (auto res : whileOp.results()) {
         resultTypes.push_back(typeConverter->convertType(res.getType()));
      }
      auto newWhileOp = rewriter.create<mlir::scf::WhileOp>(loc, TypeRange(resultTypes), whileOp.inits());
      Block* before = new Block;
      Block* after = new Block;
      newWhileOp.before().push_back(before);
      newWhileOp.after().push_back(after);
      for (auto t : resultTypes) {
         before->addArgument(t);
         after->addArgument(t);
      }

      {
         scf::IfOp::ensureTerminator(newWhileOp.before(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newWhileOp.before().front());
         Block* originalThenBlock = &whileOp.before().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalThenBlock, terminator, newWhileOp.before().front().getArguments());
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      {
         scf::IfOp::ensureTerminator(newWhileOp.after(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newWhileOp.after().front());
         Block* originalElseBlock = &whileOp.after().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalElseBlock, terminator, newWhileOp.after().front().getArguments());
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         whileOp.before().push_back(new Block());
         for (auto t : resultTypes) {
            whileOp.before().addArgument(t);
         }
         rewriter.setInsertionPointToStart(&whileOp.before().front());
         rewriter.create<mlir::db::YieldOp>(whileOp.getLoc());
      }
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         whileOp.after().push_back(new Block());
         for (auto t : resultTypes) {
            whileOp.after().addArgument(t);
         }
         rewriter.setInsertionPointToStart(&whileOp.after().front());
         rewriter.create<mlir::db::YieldOp>(whileOp.getLoc());
      }
      rewriter.replaceOp(whileOp, newWhileOp.results());

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
      auto boolType = conditionOp.condition().getType().dyn_cast_or_null<db::BoolType>();
      if (boolType && boolType.isNullable()) {
         auto i1Type = rewriter.getI1Type();
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({i1Type, i1Type}), adaptor.condition());
         Value constTrue = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), i1Type, rewriter.getIntegerAttr(i1Type, 1));
         auto negated = rewriter.create<arith::XOrIOp>(rewriter.getUnknownLoc(), unpacked.getResult(0), constTrue); //negate
         auto anded = rewriter.create<arith::AndIOp>(rewriter.getUnknownLoc(), i1Type, negated, unpacked.getResult(1));
         rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, anded, adaptor.args());
      } else {
         rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, adaptor.condition(), adaptor.args());
      }
      return success();
   }
};
class SelectLowering : public ConversionPattern {
   public:
   explicit SelectLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::SelectOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      db::SelectOpAdaptor adaptor(operands);
      db::SelectOp conditionOp = cast<db::SelectOp>(op);
      auto boolType = conditionOp.condition().getType().dyn_cast_or_null<db::BoolType>();
      if (boolType && boolType.isNullable()) {
         auto i1Type = rewriter.getI1Type();
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({i1Type, i1Type}), adaptor.condition());
         Value constTrue = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), i1Type, rewriter.getIntegerAttr(i1Type, 1));
         auto negated = rewriter.create<arith::XOrIOp>(rewriter.getUnknownLoc(), unpacked.getResult(0), constTrue); //negate
         auto anded = rewriter.create<arith::AndIOp>(rewriter.getUnknownLoc(), i1Type, negated, unpacked.getResult(1));
         rewriter.replaceOpWithNewOp<mlir::SelectOp>(op, anded,adaptor.true_value(),adaptor.false_value());
      } else {
         rewriter.replaceOpWithNewOp<mlir::SelectOp>(op, adaptor.condition(), adaptor.true_value(),adaptor.false_value());
      }
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
}