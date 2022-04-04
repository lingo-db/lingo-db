#include "mlir-support/parsing.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <mlir/Dialect/util/UtilOps.h>

#include "runtime-defs/DateRuntime.h"
#include "runtime-defs/DumpRuntime.h"
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
            result = runtime::StringRuntime::toInt(rewriter, loc)({valueToCast})[0];
            if (intWidth < 64) {
               result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, result);
            }
         } else if (auto floatType = scalarTargetType.dyn_cast_or_null<FloatType>()) {
            result = floatType.getWidth() == 32 ? runtime::StringRuntime::toFloat32(rewriter, loc)({valueToCast})[0] : runtime::StringRuntime::toFloat64(rewriter, loc)({valueToCast})[0];
         } else if (auto decimalType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalType.getS()));
            result = runtime::StringRuntime::toDecimal(rewriter, loc)({valueToCast, scale})[0];
            if (typeConverter->convertType(decimalType).cast<mlir::IntegerType>().getWidth() < 128) {
               auto converted = rewriter.create<arith::TruncIOp>(loc, typeConverter->convertType(decimalType), result);
               result = converted;
            }
         }
      } else if (auto intWidth = getIntegerWidth(scalarSourceType, false)) {
         if (scalarTargetType.isa<db::StringType>()) {
            if (intWidth < 64) {
               valueToCast = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), valueToCast);
            }
            result = runtime::StringRuntime::fromInt(rewriter, loc)({valueToCast})[0];
         }
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<FloatType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            result = floatType.getWidth() == 32 ? runtime::StringRuntime::fromFloat32(rewriter, loc)({valueToCast})[0] : runtime::StringRuntime::fromFloat64(rewriter, loc)({valueToCast})[0];
         }
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalSourceType.getS()));
            if (typeConverter->convertType(decimalSourceType).cast<mlir::IntegerType>().getWidth() < 128) {
               valueToCast = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(128), valueToCast);
            }
            result = runtime::StringRuntime::fromDecimal(rewriter, loc)({valueToCast, scale})[0];
         }
      } else if (auto charType = scalarSourceType.dyn_cast_or_null<db::CharType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            if (charType.getBytes() < 8) {
               valueToCast = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), valueToCast);
            }
            auto bytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(charType.getBytes()));
            result = runtime::StringRuntime::fromChar(rewriter, loc)({valueToCast, bytes})[0];
         }
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
      if (cmpOp.predicate() == db::DBCmpPredicate::like) {
         if (auto* defOp = cmpOp.right().getDefiningOp()) {
            if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
               std::string likeCond = constOp.getValue().cast<mlir::StringAttr>().str();
               if (likeCond.ends_with('%') && stringIsOk(likeCond.substr(0, likeCond.size() - 1))) {
                  auto newConst = rewriter.create<mlir::db::ConstantOp>(cmpOp->getLoc(), mlir::db::StringType::get(getContext()), rewriter.getStringAttr(likeCond.substr(0, likeCond.size() - 1)));
                  Value res = runtime::StringRuntime::startsWith(rewriter, cmpOp->getLoc())({adaptor.left(), rewriter.getRemappedValue(newConst)})[0];
                  rewriter.replaceOp(cmpOp, res);
                  return success();
               } else if (likeCond.starts_with('%') && stringIsOk(likeCond.substr(1, likeCond.size() - 1))) {
                  auto newConst = rewriter.create<mlir::db::ConstantOp>(cmpOp->getLoc(), mlir::db::StringType::get(getContext()), rewriter.getStringAttr(likeCond.substr(1, likeCond.size() - 1)));
                  Value res = runtime::StringRuntime::endsWith(rewriter, cmpOp->getLoc())({adaptor.left(), rewriter.getRemappedValue(newConst)})[0];
                  rewriter.replaceOp(cmpOp, res);
                  return success();
               }
            }
         }
      }
      Value res;
      Value left = adaptor.left();
      Value right = adaptor.right();
      switch (cmpOp.predicate()) {
         case db::DBCmpPredicate::eq:
            res = runtime::StringRuntime::compareEq(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::neq:
            res = runtime::StringRuntime::compareNEq(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lt:
            res = runtime::StringRuntime::compareLt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gt:
            res = runtime::StringRuntime::compareGt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lte:
            res = runtime::StringRuntime::compareLte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gte:
            res = runtime::StringRuntime::compareGte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::like:
            res = runtime::StringRuntime::like(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
      }
      rewriter.replaceOp(cmpOp, res);
      return success();
   }
};

class DBToArrowRFLowering : public mlir::db::RuntimeFunction::LoweringImpl {
   protected:
   mlir::TypeConverter* typeConverter;

   public:
   DBToArrowRFLowering() {
      this->loweringType = 0;
   }
   virtual ~DBToArrowRFLowering() {}
   void setTypeConverter(TypeConverter* typeConverter) {
      DBToArrowRFLowering::typeConverter = typeConverter;
   }
   static bool classof(const mlir::db::RuntimeFunction::LoweringImpl* impl) {
      return impl->loweringType == 0;
   }
};
class OpRFLowering : public DBToArrowRFLowering {
   using loweringFn_t = std::function<mlir::Value(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter*)>;
   loweringFn_t fn;

   public:
   OpRFLowering(const loweringFn_t& fn) : fn(fn) {}
   mlir::Value lower(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType) override {
      return fn(builder, loweredArguments, originalArgumentTypes, resType, typeConverter);
   }
};
class FnRFLowering : public DBToArrowRFLowering {
   mlir::util::FunctionSpec& func;

   public:
   FnRFLowering(mlir::util::FunctionSpec& func) : func(func) {}
   mlir::Value lower(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType) override {
      auto res = func(builder, builder.getUnknownLoc())(loweredArguments);
      assert((res.size() == 1 && resType) || (res.empty() && !resType));
      return res.size() == 1 ? res[0] : mlir::Value();
   }
};

class RuntimeCallLowering  : public OpConversionPattern<mlir::db::RuntimeCall> {
   public:
   using OpConversionPattern<mlir::db::RuntimeCall>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::RuntimeCall runtimeCallOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
      auto* fn = reg->lookup(runtimeCallOp.fn().str());
      if (!fn) return failure();
      if (!fn->lowering) return failure();
      if (auto* toArrowLowering = llvm::dyn_cast<DBToArrowRFLowering>(fn->lowering.get())) {
         toArrowLowering->setTypeConverter(typeConverter);
      }

      mlir::Value res = fn->lowering->lower(rewriter, adaptor.args(), runtimeCallOp.args().getTypes(), runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type());
      if (runtimeCallOp->getNumResults() == 0) {
         rewriter.eraseOp(runtimeCallOp);
      } else {
         rewriter.replaceOp(runtimeCallOp, res);
      }
      return success();
   }
};
} // namespace

void mlir::db::populateRuntimeSpecificScalarToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   auto reg = patterns.getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   reg->lookup("Substring")->lowering = std::make_unique<FnRFLowering>(runtime::StringRuntime::substr);
   reg->lookup("ExtractDayFromDate")->lowering = std::make_unique<FnRFLowering>(runtime::DateRuntime::extractDay);
   reg->lookup("ExtractMonthFromDate")->lowering = std::make_unique<FnRFLowering>(runtime::DateRuntime::extractMonth);
   reg->lookup("ExtractYearFromDate")->lowering = std::make_unique<FnRFLowering>(runtime::DateRuntime::extractYear);
   reg->lookup("DateAdd")->lowering = std::make_unique<OpRFLowering>([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter) -> Value {
      if (originalArgumentTypes[1].cast<mlir::db::IntervalType>().getUnit() == mlir::db::IntervalUnitAttr::daytime) {
         return rewriter.create<mlir::arith::AddIOp>(rewriter.getUnknownLoc(), loweredArguments);
      } else {
         return runtime::DateRuntime::addMonths(rewriter, rewriter.getUnknownLoc())(loweredArguments)[0];
      }
   });
   reg->lookup("DateSubtract")->lowering = std::make_unique<OpRFLowering>([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter) -> Value {
      if (originalArgumentTypes[1].cast<mlir::db::IntervalType>().getUnit() == mlir::db::IntervalUnitAttr::daytime) {
         return rewriter.create<mlir::arith::SubIOp>(rewriter.getUnknownLoc(), loweredArguments);
      } else {
         return runtime::DateRuntime::subtractMonths(rewriter, rewriter.getUnknownLoc())(loweredArguments)[0];
      }
   });
   reg->lookup("DumpValue")->lowering = std::make_unique<OpRFLowering>([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter) -> Value {
      auto loc = rewriter.getUnknownLoc();
      auto i128Type = IntegerType::get(rewriter.getContext(), 128);
      auto i64Type = IntegerType::get(rewriter.getContext(), 64);
      auto nullableType = originalArgumentTypes[0].dyn_cast_or_null<mlir::db::NullableType>();
      auto baseType = getBaseType(originalArgumentTypes[0]);

      auto f64Type = FloatType::getF64(rewriter.getContext());
      Value isNull;
      Value val;
      if (nullableType) {
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, loweredArguments[0]);
         isNull = unPackOp.vals()[0];
         val = unPackOp.vals()[1];
      } else {
         isNull = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         val = loweredArguments[0];
      }
      if (baseType.isa<mlir::IndexType>()) {
         runtime::DumpRuntime::dumpIndex(rewriter, loc)(loweredArguments[0]);
      } else if (isIntegerType(baseType, 1)) {
         runtime::DumpRuntime::dumpBool(rewriter, loc)({isNull, val});
      } else if (auto intWidth = getIntegerWidth(baseType, false)) {
         if (intWidth < 64) {
            val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
         }
         runtime::DumpRuntime::dumpInt(rewriter, loc)({isNull, val});
      } else if (auto uIntWidth = getIntegerWidth(baseType, true)) {
         if (uIntWidth < 64) {
            val = rewriter.create<arith::ExtUIOp>(loc, i64Type, val);
         }
         runtime::DumpRuntime::dumpUInt(rewriter, loc)({isNull, val});
      } else if (auto decType = baseType.dyn_cast_or_null<mlir::db::DecimalType>()) {
         if (typeConverter->convertType(decType).cast<mlir::IntegerType>().getWidth() < 128) {
            auto converted = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(128), val);
            val = converted;
         }
         Value low = rewriter.create<arith::TruncIOp>(loc, i64Type, val);
         Value shift = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
         Value scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
         Value high = rewriter.create<arith::ShRUIOp>(loc, i128Type, val, shift);
         high = rewriter.create<arith::TruncIOp>(loc, i64Type, high);
         runtime::DumpRuntime::dumpDecimal(rewriter, loc)({isNull, low, high, scale});
      } else if (auto dateType = baseType.dyn_cast_or_null<mlir::db::DateType>()) {
         runtime::DumpRuntime::dumpDate(rewriter, loc)({isNull, val});
      } else if (auto timestampType = baseType.dyn_cast_or_null<mlir::db::TimestampType>()) {
         switch (timestampType.getUnit()) {
            case mlir::db::TimeUnitAttr::second: runtime::DumpRuntime::dumpTimestampSecond(rewriter, loc)({isNull, val}); break;
            case mlir::db::TimeUnitAttr::millisecond: runtime::DumpRuntime::dumpTimestampMilliSecond(rewriter, loc)({isNull, val}); break;
            case mlir::db::TimeUnitAttr::microsecond: runtime::DumpRuntime::dumpTimestampMicroSecond(rewriter, loc)({isNull, val}); break;
            case mlir::db::TimeUnitAttr::nanosecond: runtime::DumpRuntime::dumpTimestampNanoSecond(rewriter, loc)({isNull, val}); break;
         }
      } else if (auto intervalType = baseType.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            runtime::DumpRuntime::dumpIntervalMonths(rewriter, loc)({isNull, val});
         } else {
            runtime::DumpRuntime::dumpIntervalDaytime(rewriter, loc)({isNull, val});
         }
      } else if (auto floatType = baseType.dyn_cast_or_null<mlir::FloatType>()) {
         if (floatType.getWidth() < 64) {
            val = rewriter.create<arith::ExtFOp>(loc, f64Type, val);
         }
         runtime::DumpRuntime::dumpFloat(rewriter, loc)({isNull, val});
      } else if (baseType.isa<mlir::db::StringType>()) {
         runtime::DumpRuntime::dumpString(rewriter, loc)({isNull, val});
      } else if (auto charType = baseType.dyn_cast_or_null<mlir::db::CharType>()) {
         Value numBytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(charType.getBytes()));
         if (charType.getBytes() < 8) {
            val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
         }
         runtime::DumpRuntime::dumpChar(rewriter, loc)({isNull, val, numBytes});
      }
      return mlir::Value();
   });
   patterns.insert<StringCmpOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<StringCastOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<RuntimeCallLowering>(typeConverter, patterns.getContext());
}