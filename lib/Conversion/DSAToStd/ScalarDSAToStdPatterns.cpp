#include "mlir-support/parsing.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/SCF/SCF.h>

using namespace mlir;
namespace {

class CreateFlagLowering : public ConversionPattern {
   public:
   explicit CreateFlagLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::CreateFlag::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto boolType = rewriter.getI1Type();
      Type memrefType = util::RefType::get(rewriter.getContext(), boolType);
      Value alloca;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         auto func = op->getParentOfType<mlir::FuncOp>();
         rewriter.setInsertionPointToStart(&func.getBody().front());
         alloca = rewriter.create<mlir::util::AllocaOp>(op->getLoc(), memrefType, Value());
      }
      Value falseVal = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), boolType, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.create<util::StoreOp>(op->getLoc(), falseVal, alloca, Value());
      rewriter.replaceOp(op, alloca);
      return success();
   }
};
class SetFlagLowering : public ConversionPattern {
   public:
   explicit SetFlagLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::SetFlag::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::dsa::SetFlagAdaptor adaptor(operands);
      rewriter.create<util::StoreOp>(op->getLoc(), adaptor.val(), adaptor.flag(), Value());
      rewriter.eraseOp(op);
      return success();
   }
};
class GetFlagLowering : public ConversionPattern {
   public:
   explicit GetFlagLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::GetFlag::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::dsa::GetFlagAdaptor adaptor(operands);
      auto boolType = rewriter.getI1Type();

      Value flagValue = rewriter.create<util::LoadOp>(op->getLoc(), boolType, adaptor.flag(), Value());
      rewriter.replaceOp(op, flagValue);
      return success();
   }
};


} // namespace
void mlir::dsa::populateScalarToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](mlir::dsa::FlagType type) {
      Type memrefType = util::RefType::get(patterns.getContext(), IntegerType::get(type.getContext(), 1));
      return memrefType;
   });

   patterns.insert<CreateFlagLowering>(typeConverter, patterns.getContext());
   patterns.insert<SetFlagLowering>(typeConverter, patterns.getContext());
   patterns.insert<GetFlagLowering>(typeConverter, patterns.getContext());
}