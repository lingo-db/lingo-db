#include "mlir-support/parsing.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include <mlir/Dialect/util/UtilOps.h>

#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Transforms/DialectConversion.h" //
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
namespace {

class StringCastOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit StringCastOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CastOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;

      auto castOp = cast<mlir::db::CastOp>(op);
      auto loc = op->getLoc();
      auto scalarSourceType = castOp.val().getType();
      auto scalarTargetType = castOp.getType();
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (!scalarSourceType.isa<mlir::db::StringType>() && !scalarTargetType.isa<mlir::db::StringType>()) return failure();

      Value valueToCast = operands[0];
      Value result;
      if (scalarSourceType == scalarTargetType) {
         //nothing to do here
      } else if (auto stringType = scalarSourceType.dyn_cast_or_null<db::StringType>()) {
         if (auto intWidth = getIntegerWidth(scalarTargetType, false)) {
            result = functionRegistry.call(rewriter, loc, FunctionId::CastStringToInt64, ValueRange({valueToCast}))[0];
            if (intWidth < 64) {
               result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, result);
            }
         } else if (auto floatType = scalarTargetType.dyn_cast_or_null<FloatType>()) {
            FunctionId castFn = floatType.getWidth() == 32 ? FunctionId ::CastStringToFloat32 : FunctionId ::CastStringToFloat64;
            result = functionRegistry.call(rewriter, loc, castFn, ValueRange({valueToCast}))[0];
         } else if (auto decimalType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalType.getS()));
            result = functionRegistry.call(rewriter, loc, FunctionId ::CastStringToDecimal, ValueRange({valueToCast, scale}))[0];
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
            result = functionRegistry.call(rewriter, loc, FunctionId ::CastInt64ToString, ValueRange({valueToCast}))[0];
         }
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<FloatType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            FunctionId castFn = floatType.getWidth() == 32 ? FunctionId ::CastFloat32ToString : FunctionId ::CastFloat64ToString;
            result = functionRegistry.call(rewriter, loc, castFn, ValueRange({valueToCast}))[0];
         }
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalSourceType.getS()));
            if (typeConverter->convertType(decimalSourceType).cast<mlir::IntegerType>().getWidth() < 128) {
               valueToCast = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(128), valueToCast);
            }
            result = functionRegistry.call(rewriter, loc, FunctionId ::CastDecimalToString, ValueRange({valueToCast, scale}))[0];
         }
      } else if (auto charType = scalarSourceType.dyn_cast_or_null<db::CharType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            if (charType.getBytes() < 8) {
               valueToCast = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), valueToCast);
            }
            auto bytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(charType.getBytes()));
            result = functionRegistry.call(rewriter, loc, FunctionId ::CastCharToString, ValueRange({valueToCast, bytes}))[0];
         }
      }
      if (result) {
         rewriter.replaceOp(op, result);
         return success();
      } else {
         return failure();
      }
   }
};
class StringCmpOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;
   using FuncId = mlir::db::codegen::FunctionRegistry::FunctionId;

   public:
   explicit StringCmpOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CmpOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}
   mlir::db::codegen::FunctionRegistry::FunctionId funcForStrCompare(db::DBCmpPredicate pred) const {
      switch (pred) {
         case db::DBCmpPredicate::eq:
            return FuncId::CmpStringEQ;
         case db::DBCmpPredicate::neq:
            return FuncId::CmpStringNEQ;
         case db::DBCmpPredicate::lt:
            return FuncId::CmpStringLT;
         case db::DBCmpPredicate::gt:
            return FuncId::CmpStringGT;
         case db::DBCmpPredicate::lte:
            return FuncId::CmpStringLTE;
         case db::DBCmpPredicate::gte:
            return FuncId::CmpStringGTE;
         case db::DBCmpPredicate::like:
            return FuncId::CmpStringLike;
      }
      assert(false && "unexpected case");
      return FuncId::CmpStringEQ;
   }
   bool stringIsOk(std::string str) const {
      for (auto x : str) {
         if (!std::isalnum(x)) return false;
      }
      return true;
   }
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto cmpOp = cast<db::CmpOp>(op);
      db::CmpOpAdaptor adaptor(operands);
      auto type = cmpOp.left().getType();
      if (!type.isa<db::StringType>()) {
         return failure();
      }
      using FuncId = mlir::db::codegen::FunctionRegistry::FunctionId;
      if (cmpOp.predicate() == db::DBCmpPredicate::like) {
         if (auto* defOp = cmpOp.right().getDefiningOp()) {
            if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
               std::string likeCond = constOp.getValue().cast<mlir::StringAttr>().str();
               if (likeCond.ends_with('%') && stringIsOk(likeCond.substr(0, likeCond.size() - 1))) {
                  auto newConst = rewriter.create<mlir::db::ConstantOp>(op->getLoc(), mlir::db::StringType::get(getContext()), rewriter.getStringAttr(likeCond.substr(0, likeCond.size() - 1)));
                  Value res = functionRegistry.call(rewriter, op->getLoc(), FuncId ::CmpStringStartsWith, ValueRange({adaptor.left(), rewriter.getRemappedValue(newConst)}))[0];
                  rewriter.replaceOp(op, res);
                  return success();
               } else if (likeCond.starts_with('%') && stringIsOk(likeCond.substr(1, likeCond.size() - 1))) {
                  auto newConst = rewriter.create<mlir::db::ConstantOp>(op->getLoc(), mlir::db::StringType::get(getContext()), rewriter.getStringAttr(likeCond.substr(1, likeCond.size() - 1)));
                  Value res = functionRegistry.call(rewriter, op->getLoc(), FuncId ::CmpStringEndsWith, ValueRange({adaptor.left(), rewriter.getRemappedValue(newConst)}))[0];
                  rewriter.replaceOp(op, res);
                  return success();
               }
            }
         }
      }
      FuncId cmpFunc = funcForStrCompare(cmpOp.predicate());
      Value res = functionRegistry.call(rewriter, op->getLoc(), cmpFunc, ValueRange({adaptor.left(), adaptor.right()}))[0];
      rewriter.replaceOp(op, res);
      return success();
   }
};

class FreeOpLowering : public ConversionPattern {
   public:
   explicit FreeOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::FreeOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      /*      auto freeOp = cast<mlir::db::FreeOp>(op);
      mlir::db::FreeOpAdaptor adaptor(operands);
      auto val = adaptor.val();
      auto rewritten = ::llvm::TypeSwitch<::mlir::Type, bool>(freeOp.val().getType())
                          .Case<::mlir::db::AggregationHashtableType>([&](::mlir::db::AggregationHashtableType type) {
                             if (!type.getKeyType().getTypes().empty()) {
                                //todo free aggregation hashtable
                                //functionRegistry.call(rewriter, loc, FunctionId::AggrHtFree, val);
                             }
                             return true;
                          })
                          .Case<::mlir::db::VectorType>([&](::mlir::db::VectorType) {
                             //todo: free vector
                             //functionRegistry.call(rewriter, loc, FunctionId::VectorFree, val);
                             return true;
                          })
                          .Case<::mlir::db::JoinHashtableType>([&](::mlir::db::JoinHashtableType) {
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
class DateAddOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit DateAddOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DateAddOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto dateAddOp = mlir::cast<mlir::db::DateAddOp>(op);
      mlir::db::DateAddOpAdaptor adaptor(operands);
      auto dateVal = adaptor.left();
      auto invervalVal = adaptor.right();
      auto loc = op->getLoc();
      auto dateType = dateAddOp.left().getType().cast<mlir::db::DateType>().getUnit();
      if (dateType == db::DateUnitAttr::day) {
         dateVal = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), dateVal);
         Value multiplier = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 24 * 60 * 60 * 1000));
         dateVal = rewriter.create<arith::MulIOp>(loc, dateVal, multiplier);
      }
      if (dateAddOp.right().getType().cast<mlir::db::IntervalType>().getUnit() == mlir::db::IntervalUnitAttr::daytime) {
         dateVal = rewriter.create<mlir::arith::AddIOp>(op->getLoc(), dateVal, invervalVal);
      } else {
         dateVal = functionRegistry.call(rewriter, loc, FunctionId::TimestampAddMonth, ValueRange({invervalVal, dateVal}))[0];
      }
      if (dateType == db::DateUnitAttr::day) {
         Value multiplier = rewriter.template create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 24 * 60 * 60 * 1000));
         dateVal = rewriter.template create<arith::DivUIOp>(loc, dateVal, multiplier);
         dateVal = rewriter.template create<arith::TruncIOp>(loc, rewriter.getI32Type(), dateVal);
      }
      rewriter.replaceOp(op, dateVal);
      return success();
   }
};
class DateExtractOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit DateExtractOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DateExtractOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto dateExtractOp = mlir::cast<mlir::db::DateExtractOp>(op);
      mlir::db::DateExtractOpAdaptor adaptor(operands);
      auto v = adaptor.val();
      auto loc = op->getLoc();
      auto dateType = dateExtractOp.val().getType().cast<mlir::db::DateType>();
      if (dateType.getUnit() == db::DateUnitAttr::day) {
         auto i64Type = IntegerType::get(rewriter.getContext(), 64);
         v = rewriter.template create<arith::ExtUIOp>(loc, i64Type, v);
         Value multiplier = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i64Type, 24 * 60 * 60 * 1000));
         v = rewriter.template create<arith::MulIOp>(loc, v, multiplier);
      }
      FunctionId functionId;
      switch (dateExtractOp.unit()) {
         case mlir::db::ExtractableTimeUnitAttr::second: functionId = FunctionId::DateExtractSecond; break;
         case mlir::db::ExtractableTimeUnitAttr::minute: functionId = FunctionId::DateExtractMinute; break;
         case mlir::db::ExtractableTimeUnitAttr::hour: functionId = FunctionId::DateExtractHour; break;
         case mlir::db::ExtractableTimeUnitAttr::dow: functionId = FunctionId::DateExtractDow; break;
         //case mlir::db::ExtractableTimeUnitAttr::week: functionId = FunctionId::DateExtractWeek; break;
         case mlir::db::ExtractableTimeUnitAttr::day: functionId = FunctionId::DateExtractDay; break;
         case mlir::db::ExtractableTimeUnitAttr::month: functionId = FunctionId::DateExtractMonth; break;
         case mlir::db::ExtractableTimeUnitAttr::doy: functionId = FunctionId::DateExtractDoy; break;
         //case mlir::db::ExtractableTimeUnitAttr::quarter: functionId = FunctionId::DateExtractQuarter; break;
         case mlir::db::ExtractableTimeUnitAttr::year: functionId = FunctionId::DateExtractYear; break;
         //case mlir::db::ExtractableTimeUnitAttr::decade: functionId = FunctionId::DateExtractDecade; break;
         //case mlir::db::ExtractableTimeUnitAttr::century: functionId = FunctionId::DateExtractCentury; break;
         //case mlir::db::ExtractableTimeUnitAttr::millennium: functionId = FunctionId::DateExtractMillenium; break;
         default:
            assert(false && "not implemented yet");
      }
      rewriter.replaceOp(op, functionRegistry.call(rewriter, op->getLoc(), functionId, v)[0]);
      return success();
   }
};
class DBToArrowRFLowering : public mlir::db::RuntimeFunction::LoweringImpl {
   protected:
   mlir::db::codegen::FunctionRegistry* nativeFunctionRegistry;
   mlir::TypeConverter* typeConverter;

   public:
   DBToArrowRFLowering() {
      this->loweringType = 0;
   }
   virtual ~DBToArrowRFLowering() {}
   void setNativeFunctionRegistry(db::codegen::FunctionRegistry* nativeFunctionRegistry) {
      DBToArrowRFLowering::nativeFunctionRegistry = nativeFunctionRegistry;
   }
   void setTypeConverter(TypeConverter* typeConverter) {
      DBToArrowRFLowering::typeConverter = typeConverter;
   }
   static bool classof(const mlir::db::RuntimeFunction::LoweringImpl* impl) {
      return impl->loweringType == 0;
   }
};
class OpRFLowering : public DBToArrowRFLowering {
   using loweringFn_t = std::function<mlir::Value(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter*, mlir::db::codegen::FunctionRegistry*)>;
   loweringFn_t fn;

   public:
   OpRFLowering(const loweringFn_t& fn) : fn(fn) {}
   mlir::Value lower(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType) override {
      return fn(builder, loweredArguments, originalArgumentTypes, resType, typeConverter, nativeFunctionRegistry);
   }
};
class FnRFLowering : public DBToArrowRFLowering {
   mlir::db::codegen::FunctionRegistry::FunctionId functionId;

   public:
   FnRFLowering(mlir::db::codegen::FunctionRegistry::FunctionId functionId) : functionId(functionId) {}
   mlir::Value convert(mlir::OpBuilder& builder, mlir::Value v, mlir::Type target) {
      if (v.getType() == target) return v;
      if (target.isIndex()) {
         return builder.create<mlir::arith::IndexCastOp>(builder.getUnknownLoc(), target, v);
      }
      assert(false && "should not happen");
      return v;
   }
   mlir::Value lower(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType) override {
      auto registeredFunction = nativeFunctionRegistry->getRegisteredFunction(functionId);
      std::vector<mlir::Value> args;
      assert(registeredFunction.operands.size() == loweredArguments.size());
      for (size_t i = 0; i < loweredArguments.size(); i++) {
         args.push_back(convert(builder, loweredArguments[i], registeredFunction.operands[i]));
      }
      auto res = nativeFunctionRegistry->call(builder, builder.getUnknownLoc(), functionId, args);
      assert((res.size() == 1 && resType) || (res.empty() && !resType));
      return res.size() == 1 ? res[0] : mlir::Value();
   }
};

class RuntimeCallLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit RuntimeCallLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::RuntimeCall::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();

      auto runtimeCallOp = mlir::cast<mlir::db::RuntimeCall>(op);
      mlir::db::RuntimeCallAdaptor adaptor(operands);
      auto* fn = reg->lookup(runtimeCallOp.fn().str());
      if (!fn) return failure();
      if (!fn->lowering) return failure();
      if (auto* toArrowLowering = llvm::dyn_cast<DBToArrowRFLowering>(fn->lowering.get())) {
         toArrowLowering->setNativeFunctionRegistry(&functionRegistry);
         toArrowLowering->setTypeConverter(typeConverter);
      }

      mlir::Value res = fn->lowering->lower(rewriter, adaptor.args(), runtimeCallOp.args().getTypes(), runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type());
      if (runtimeCallOp->getNumResults() == 0) {
         rewriter.eraseOp(op);
      } else {
         rewriter.replaceOp(op, res);
      }
      return success();
   }
};
} // namespace

void mlir::db::populateRuntimeSpecificScalarToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   auto reg = patterns.getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   reg->lookup("Substring")->lowering = std::make_unique<FnRFLowering>(mlir::db::codegen::FunctionRegistry::FunctionId::Substring);
   reg->lookup("DumpValue")->lowering = std::make_unique<OpRFLowering>([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter, mlir::db::codegen::FunctionRegistry* functionRegistry) -> Value {
      using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;
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
         functionRegistry->call(rewriter, loc, FunctionId::DumpIndex, loweredArguments[0]);
      } else if (isIntegerType(baseType, 1)) {
         functionRegistry->call(rewriter, loc, FunctionId::DumpBool, ValueRange({isNull, val}));
      } else if (auto intWidth = getIntegerWidth(baseType, false)) {
         if (intWidth < 64) {
            val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
         }
         functionRegistry->call(rewriter, loc, FunctionId::DumpInt, ValueRange({isNull, val}));
      } else if (auto uIntWidth = getIntegerWidth(baseType, true)) {
         if (uIntWidth < 64) {
            val = rewriter.create<arith::ExtUIOp>(loc, i64Type, val);
         }
         functionRegistry->call(rewriter, loc, FunctionId::DumpUInt, ValueRange({isNull, val}));
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
         functionRegistry->call(rewriter, loc, FunctionId::DumpDecimal, ValueRange({isNull, low, high, scale}));
      } else if (auto dateType = baseType.dyn_cast_or_null<mlir::db::DateType>()) {
         if (dateType.getUnit() == mlir::db::DateUnitAttr::millisecond) {
            functionRegistry->call(rewriter, loc, FunctionId::DumpDateMillisecond, ValueRange({isNull, val}));
         } else {
            functionRegistry->call(rewriter, loc, FunctionId::DumpDateDay, ValueRange({isNull, val}));
         }
      } else if (auto timestampType = baseType.dyn_cast_or_null<mlir::db::TimestampType>()) {
         FunctionId functionId;
         switch (timestampType.getUnit()) {
            case mlir::db::TimeUnitAttr::second: functionId = FunctionId::DumpTimestampSecond; break;
            case mlir::db::TimeUnitAttr::millisecond: functionId = FunctionId::DumpTimestampMillisecond; break;
            case mlir::db::TimeUnitAttr::microsecond: functionId = FunctionId::DumpTimestampMicrosecond; break;
            case mlir::db::TimeUnitAttr::nanosecond: functionId = FunctionId::DumpTimestampNanosecond; break;
         }
         functionRegistry->call(rewriter, loc, functionId, ValueRange({isNull, val}));
      } else if (auto intervalType = baseType.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            functionRegistry->call(rewriter, loc, FunctionId::DumpIntervalMonths, ValueRange({isNull, val}));
         } else {
            functionRegistry->call(rewriter, loc, FunctionId::DumpIntervalDayTime, ValueRange({isNull, val}));
         }

      } else if (auto floatType = baseType.dyn_cast_or_null<mlir::FloatType>()) {
         if (floatType.getWidth() < 64) {
            val = rewriter.create<arith::ExtFOp>(loc, f64Type, val);
         }
         functionRegistry->call(rewriter, loc, FunctionId::DumpFloat, ValueRange({isNull, val}));
      } else if (baseType.isa<mlir::db::StringType>()) {
         functionRegistry->call(rewriter, loc, FunctionId::DumpString, ValueRange({isNull, val}));
      } else if (auto charType = baseType.dyn_cast_or_null<mlir::db::CharType>()) {
         Value numBytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(charType.getBytes()));
         if (charType.getBytes() < 8) {
            val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
         }
         functionRegistry->call(rewriter, loc, FunctionId::DumpChar, ValueRange({isNull, val, numBytes}));
      }
      return mlir::Value();
   });
   patterns.insert<DateExtractOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<DateAddOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<StringCmpOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<StringCastOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<FreeOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<RuntimeCallLowering>(functionRegistry, typeConverter, patterns.getContext());
}