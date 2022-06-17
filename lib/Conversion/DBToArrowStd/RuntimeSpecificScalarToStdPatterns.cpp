#include "mlir-support/parsing.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "runtime-defs/StringRuntime.h"

#include <mlir/Dialect/util/FunctionHelper.h>

#include "mlir/Transforms/DialectConversion.h" //
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
namespace {

class StringCastOpLowering : public OpConversionPattern<mlir::db::CastOp> {
   public:
   using OpConversionPattern<mlir::db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::CastOp castOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = castOp->getLoc();
      auto scalarSourceType = castOp.val().getType();
      auto scalarTargetType = castOp.getType();
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (!scalarSourceType.isa<mlir::db::StringType>() && !scalarTargetType.isa<mlir::db::StringType>()) return failure();

      Value valueToCast = adaptor.val();
      Value result;
      if (scalarSourceType == scalarTargetType) {
         //nothing to do here
      } else if (auto stringType = scalarSourceType.dyn_cast_or_null<db::StringType>()) {
         if (auto intWidth = getIntegerWidth(scalarTargetType, false)) {
            result = rt::StringRuntime::toInt(rewriter, loc)({valueToCast})[0];
            if (intWidth < 64) {
               result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, result);
            }
         } else if (auto floatType = scalarTargetType.dyn_cast_or_null<FloatType>()) {
            result = floatType.getWidth() == 32 ? rt::StringRuntime::toFloat32(rewriter, loc)({valueToCast})[0] : rt::StringRuntime::toFloat64(rewriter, loc)({valueToCast})[0];
         } else if (auto decimalType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalType.getS()));
            result = rt::StringRuntime::toDecimal(rewriter, loc)({valueToCast, scale})[0];
            if (typeConverter->convertType(decimalType).cast<mlir::IntegerType>().getWidth() < 128) {
               auto converted = rewriter.create<arith::TruncIOp>(loc, typeConverter->convertType(decimalType), result);
               result = converted;
            }
         }
      } else if (auto intWidth = getIntegerWidth(scalarSourceType, false)) {
         result = rt::StringRuntime::fromInt(rewriter, loc)({valueToCast})[0];
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<FloatType>()) {
         result = floatType.getWidth() == 32 ? rt::StringRuntime::fromFloat32(rewriter, loc)({valueToCast})[0] : rt::StringRuntime::fromFloat64(rewriter, loc)({valueToCast})[0];
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalSourceType.getS()));
            result = rt::StringRuntime::fromDecimal(rewriter, loc)({valueToCast, scale})[0];
      } else if (auto charType = scalarSourceType.dyn_cast_or_null<db::CharType>()) {
            auto bytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(charType.getBytes()));
            result = rt::StringRuntime::fromChar(rewriter, loc)({valueToCast, bytes})[0];

      }
      if (result) {
         rewriter.replaceOp(castOp, result);
         return success();
      } else {
         return failure();
      }
   }
};
class StringCmpOpLowering : public OpConversionPattern<mlir::db::CmpOp> {
   public:
   using OpConversionPattern<mlir::db::CmpOp>::OpConversionPattern;

   bool stringIsOk(std::string str) const {
      for (auto x : str) {
         if (!std::isalnum(x)) return false;
      }
      return true;
   }
   LogicalResult matchAndRewrite(mlir::db::CmpOp cmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = cmpOp.left().getType();
      if (!type.isa<db::StringType>()) {
         return failure();
      }
      Value res;
      Value left = adaptor.left();
      Value right = adaptor.right();
      switch (cmpOp.predicate()) {
         case db::DBCmpPredicate::eq:
            res = rt::StringRuntime::compareEq(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::neq:
            res = rt::StringRuntime::compareNEq(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lt:
            res = rt::StringRuntime::compareLt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gt:
            res = rt::StringRuntime::compareGt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lte:
            res = rt::StringRuntime::compareLte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gte:
            res = rt::StringRuntime::compareGte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
      }
      rewriter.replaceOp(cmpOp, res);
      return success();
   }
};

class RuntimeCallLowering : public OpConversionPattern<mlir::db::RuntimeCall> {
   public:
   using OpConversionPattern<mlir::db::RuntimeCall>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::RuntimeCall runtimeCallOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
      auto* fn = reg->lookup(runtimeCallOp.fn().str());
      if (!fn) return failure();
      Value result;
      mlir::Type resType = runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type();
      if (std::holds_alternative<mlir::util::FunctionSpec>(fn->implementation)) {
         auto& implFn = std::get<mlir::util::FunctionSpec>(fn->implementation);
         auto resRange = implFn(rewriter, rewriter.getUnknownLoc())(adaptor.args());
         assert((resRange.size() == 1 && resType) || (resRange.empty() && !resType));
         result = resRange.size() == 1 ? resRange[0] : mlir::Value();
      } else if (std::holds_alternative<mlir::db::RuntimeFunction::loweringFnT>(fn->implementation)) {
         auto& implFn = std::get<mlir::db::RuntimeFunction::loweringFnT>(fn->implementation);
         result = implFn(rewriter, adaptor.args(), runtimeCallOp.args().getTypes(), runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type(), typeConverter,runtimeCallOp->getLoc());
      }

      if (runtimeCallOp->getNumResults() == 0) {
         rewriter.eraseOp(runtimeCallOp);
      } else {
         rewriter.replaceOp(runtimeCallOp, result);
      }
      return success();
   }
};
} // namespace

void mlir::db::populateRuntimeSpecificScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<StringCmpOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<StringCastOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<RuntimeCallLowering>(typeConverter, patterns.getContext());
}