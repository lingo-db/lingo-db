#include "mlir-support/parsing.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include <mlir/Dialect/util/UtilOps.h>

#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Conversion/DSAToStd/FunctionRegistry.h"
#include "mlir/Transforms/DialectConversion.h" //
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
namespace {

class FreeOpLowering : public ConversionPattern {
   public:
   explicit FreeOpLowering(dsa::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::FreeOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      /*      auto freeOp = cast<mlir::dsa::FreeOp>(op);
      mlir::dsa::FreeOpAdaptor adaptor(operands);
      auto val = adaptor.val();
      auto rewritten = ::llvm::TypeSwitch<::mlir::Type, bool>(freeOp.val().getType())
                          .Case<::mlir::dsa::AggregationHashtableType>([&](::mlir::dsa::AggregationHashtableType type) {
                             if (!type.getKeyType().getTypes().empty()) {
                                //todo free aggregation hashtable
                                //functionRegistry.call(rewriter, loc, FunctionId::AggrHtFree, val);
                             }
                             return true;
                          })
                          .Case<::mlir::dsa::VectorType>([&](::mlir::dsa::VectorType) {
                             //todo: free vector
                             //functionRegistry.call(rewriter, loc, FunctionId::VectorFree, val);
                             return true;
                          })
                          .Case<::mlir::dsa::JoinHashtableType>([&](::mlir::dsa::JoinHashtableType) {
                             //todo: free join hashtable
                             //functionRegistry.call(rewriter, loc, FunctionId::JoinHtFree, val);
                             return true;
                          })
                          .Default([&](::mlir::Type) { return false; });
      if (rewritten) {
         rewriter.eraseOp(op);
         return success();
      } else {
         return failure();
      }
      */
      rewriter.eraseOp(op);
      return success();
   }
};
/*class DateAddOpLowering : public ConversionPattern {
   dsa::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit DateAddOpLowering(dsa::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::DateAddOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = dsa::codegen::FunctionRegistry::FunctionId;
      auto dateAddOp = mlir::cast<mlir::dsa::DateAddOp>(op);
      mlir::dsa::DateAddOpAdaptor adaptor(operands);
      auto dateVal = adaptor.left();
      auto invervalVal = adaptor.right();
      auto loc = op->getLoc();
      if (dateAddOp.right().getType().cast<mlir::dsa::IntervalType>().getUnit() == mlir::dsa::IntervalUnitAttr::daytime) {
         dateVal = rewriter.create<mlir::arith::AddIOp>(op->getLoc(), dateVal, invervalVal);
      } else {
         dateVal = functionRegistry.call(rewriter, loc, FunctionId::TimestampAddMonth, ValueRange({invervalVal, dateVal}))[0];
      }
      rewriter.replaceOp(op, dateVal);
      return success();
   }
};*/
} // end namespace
void mlir::dsa::populateRuntimeSpecificScalarToStdPatterns(mlir::dsa::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<FreeOpLowering>(functionRegistry, typeConverter, patterns.getContext());
}