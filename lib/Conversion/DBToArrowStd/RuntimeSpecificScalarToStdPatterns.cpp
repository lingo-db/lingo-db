#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include <mlir/Conversion/DBToArrowStd/NullHandler.h>
#include <mlir/Dialect/util/UtilOps.h>

#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Transforms/DialectConversion.h" //
using namespace mlir;
namespace {

template <class OpClass, class LeftT, class RightT, class ResT>
class SimpleBinOpToFuncLowering : public ConversionPattern {
   std::function<Value(LeftT, Value, ConversionPatternRewriter&)> processLeft;
   std::function<Value(RightT, Value, ConversionPatternRewriter&)> processRight;
   std::function<std::vector<Value>(Value, Value)> combine;
   std::function<FuncOp(OpClass, LeftT, RightT, ConversionPatternRewriter& rewriter)> provideFunc;
   std::function<Value(ResT, Value, ConversionPatternRewriter&)> processResult;

   public:
   explicit SimpleBinOpToFuncLowering(MLIRContext* context,
                                      std::function<Value(LeftT, Value, ConversionPatternRewriter&)>
                                         processLeft,
                                      std::function<Value(RightT, Value, ConversionPatternRewriter&)>
                                         processRight,
                                      std::function<std::vector<Value>(Value, Value)>
                                         combine,
                                      std::function<FuncOp(OpClass, LeftT, RightT, ConversionPatternRewriter& rewriter)>
                                         provideFunc,
                                      std::function<Value(ResT, Value, ConversionPatternRewriter&)>
                                         processResult)
      : ConversionPattern(OpClass::getOperationName(), 1, context), processLeft(processLeft), processRight(processRight), combine(combine), provideFunc(provideFunc), processResult(processResult) {}
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      typename OpClass::Adaptor opAdaptor(operands);
      db::NullHandler nullHandler(*typeConverter, rewriter);
      auto casted = cast<OpClass>(op);
      LeftT leftType = casted.left().getType().template dyn_cast_or_null<LeftT>();
      RightT rightType = casted.right().getType().template dyn_cast_or_null<RightT>();
      ResT resType = casted.getResult().getType().template dyn_cast_or_null<ResT>();
      if (!(leftType && rightType && resType)) {
         return failure();
      }

      Value left = nullHandler.getValue(casted.left(), opAdaptor.left());
      Value right = nullHandler.getValue(casted.right(), opAdaptor.right());

      left = processLeft(leftType, left, rewriter);
      right = processRight(rightType, right, rewriter);
      FuncOp func = provideFunc(casted, leftType, rightType, rewriter);
      auto call = rewriter.create<CallOp>(rewriter.getUnknownLoc(), func, combine(left, right));
      Value res = call.getResult(0);
      res = processResult(resType, res, rewriter);
      rewriter.replaceOp(op, nullHandler.combineResult(res));
      return success();
   }
};
template <class OpClass, class ValT, class ResT>
class SimpleUnOpToFuncLowering : public ConversionPattern {
   std::function<Value(ValT, Value, ConversionPatternRewriter&)> processVal;
   std::function<FuncOp(OpClass, ValT, ConversionPatternRewriter& rewriter)> provideFunc;
   std::function<Value(ResT, Value, ConversionPatternRewriter&)> processResult;

   public:
   explicit SimpleUnOpToFuncLowering(MLIRContext* context,
                                     std::function<Value(ValT, Value, ConversionPatternRewriter&)>
                                        processIn,
                                     std::function<FuncOp(OpClass, ValT, ConversionPatternRewriter& rewriter)>
                                        provideFunc,
                                     std::function<Value(ResT, Value, ConversionPatternRewriter&)>
                                        processResult)
      : ConversionPattern(OpClass::getOperationName(), 1, context), processVal(processIn), provideFunc(provideFunc), processResult(processResult) {}
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      typename OpClass::Adaptor opAdaptor(operands);
      db::NullHandler nullHandler(*typeConverter, rewriter);
      auto casted = cast<OpClass>(op);
      ValT valType = casted.val().getType().template dyn_cast_or_null<ValT>();
      ResT resType = casted.getResult().getType().template dyn_cast_or_null<ResT>();
      if (!(valType && resType)) {
         return failure();
      }
      Value val = nullHandler.getValue(casted.val(), opAdaptor.val());
      val = processVal(valType, val, rewriter);
      FuncOp func = provideFunc(casted, valType, rewriter);
      auto call = rewriter.create<CallOp>(rewriter.getUnknownLoc(), func, val);
      Value res = call.getResult(0);
      res = processResult(resType, res, rewriter);
      rewriter.replaceOp(op, nullHandler.combineResult(res));
      return success();
   }
};

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
      auto loc = rewriter.getUnknownLoc();
      auto sourceType = castOp.val().getType().cast<db::DBType>();
      auto targetType = castOp.getType().cast<db::DBType>();
      auto scalarSourceType = sourceType.getBaseType();
      auto scalarTargetType = targetType.getBaseType();
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (!scalarSourceType.isa<mlir::db::StringType>() && !scalarTargetType.isa<mlir::db::StringType>()) return failure();

      Value isNull;
      Value value;
      if (sourceType.isNullable()) {
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, typeConverter->convertType(sourceType).dyn_cast_or_null<TupleType>().getTypes(), operands[0]);
         isNull = unPackOp.vals()[0];
         value = unPackOp.vals()[1];
      } else {
         isNull = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         value = operands[0];
      }
      if (scalarSourceType == scalarTargetType) {
         //nothing to do here
      } else if (auto stringType = scalarSourceType.dyn_cast_or_null<db::StringType>()) {
         if (auto intType = scalarTargetType.dyn_cast_or_null<db::IntType>()) {
            value = functionRegistry.call(rewriter, FunctionId::CastStringToInt64, ValueRange({isNull, value}))[0];
            if (intType.getWidth() < 64) {
               value = rewriter.create<TruncateIOp>(loc, value, convertedTargetType);
            }
         } else if (auto floatType = scalarTargetType.dyn_cast_or_null<db::FloatType>()) {
            FunctionId castFn = floatType.getWidth() == 32 ? FunctionId ::CastStringToFloat32 : FunctionId ::CastStringToFloat64;
            value = functionRegistry.call(rewriter, castFn, ValueRange({isNull, value}))[0];
         } else if (auto decimalType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto scale = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalType.getS()));
            value = functionRegistry.call(rewriter, FunctionId ::CastStringToDecimal, ValueRange({isNull, value, scale}))[0];
         } else {
            return failure();
         }
      } else if (auto intType = scalarSourceType.dyn_cast_or_null<db::IntType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            if (intType.getWidth() < 64) {
               value = rewriter.create<SignExtendIOp>(loc, value, rewriter.getI64Type());
            }
            value = functionRegistry.call(rewriter, FunctionId ::CastInt64ToString, ValueRange({isNull, value}))[0];
         } else {
            return failure();
         }
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<db::FloatType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            FunctionId castFn = floatType.getWidth() == 32 ? FunctionId ::CastFloat32ToString : FunctionId ::CastFloat64ToString;
            value = functionRegistry.call(rewriter, castFn, ValueRange({isNull, value}))[0];
         } else {
            return failure();
         }
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
         if (scalarTargetType.isa<db::StringType>()) {
            auto scale = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalSourceType.getS()));

            value = functionRegistry.call(rewriter, FunctionId ::CastDecimalToString, ValueRange({isNull, value, scale}))[0];
         } else {
            return failure();
         }
      } else {
         return failure();
      }
      //todo convert types
      if (targetType.isNullable()) {
         Value combined = rewriter.create<mlir::util::PackOp>(loc, typeConverter->convertType(targetType), ValueRange({isNull, value}));
         rewriter.replaceOp(op, combined);
      } else {
         rewriter.replaceOp(op, value);
      }
      return success();
   }
};
class StringCmpOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit StringCmpOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CmpOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}
   mlir::db::codegen::FunctionRegistry::FunctionId funcForStrCompare(db::DBCmpPredicate pred) const {
      using FuncId = mlir::db::codegen::FunctionRegistry::FunctionId;
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
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto cmpOp = cast<db::CmpOp>(op);

      auto type = cmpOp.left().getType().cast<db::DBType>().getBaseType();
      if (!type.isa<db::StringType>()) {
         return failure();
      }
      db::NullHandler nullHandler(*typeConverter, rewriter);
      using FuncId = mlir::db::codegen::FunctionRegistry::FunctionId;
      FuncId cmpFunc = funcForStrCompare(cmpOp.predicate());
      Value res = functionRegistry.call(rewriter, cmpFunc, ValueRange({nullHandler.isNull(), cmpOp.left(), cmpOp.right()}))[0];
      rewriter.replaceOp(op, nullHandler.combineResult(res));
      return success();
   }
};
class DumpOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit DumpOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DumpOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;
      mlir::db::DumpOp::Adaptor dumpOpAdaptor(operands);
      auto loc = op->getLoc();
      auto printOp = cast<mlir::db::DumpOp>(op);
      Value val = printOp.val();
      auto i128Type = IntegerType::get(rewriter.getContext(), 128);
      auto i64Type = IntegerType::get(rewriter.getContext(), 64);
      auto type = val.getType().dyn_cast_or_null<mlir::db::DBType>().getBaseType();

      auto f64Type = FloatType::getF64(rewriter.getContext());
      Value isNull;
      if (val.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
         TupleType tupleType = typeConverter->convertType(val.getType()).dyn_cast_or_null<TupleType>();
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), tupleType.getTypes(), dumpOpAdaptor.val());
         isNull = unPackOp.vals()[0];
         val = unPackOp.vals()[1];
      } else {
         isNull = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         val = dumpOpAdaptor.val();
      }

      if (auto dbIntType = type.dyn_cast_or_null<mlir::db::IntType>()) {
         if (dbIntType.getWidth() < 64) {
            val = rewriter.create<SignExtendIOp>(loc, val, i64Type);
         }
         functionRegistry.call(rewriter, FunctionId::DumpInt, ValueRange({isNull, val}));
      } else if (auto dbUIntType = type.dyn_cast_or_null<mlir::db::UIntType>()) {
         if (dbUIntType.getWidth() < 64) {
            val = rewriter.create<ZeroExtendIOp>(loc, val, i64Type);
         }
         functionRegistry.call(rewriter, FunctionId::DumpUInt, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::BoolType>()) {
         functionRegistry.call(rewriter, FunctionId::DumpBool, ValueRange({isNull, val}));
      } else if (auto decType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         Value low = rewriter.create<TruncateIOp>(loc, val, i64Type);
         Value shift = rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
         Value scale = rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
         Value high = rewriter.create<UnsignedShiftRightOp>(loc, i128Type, val, shift);
         high = rewriter.create<TruncateIOp>(loc, high, i64Type);
         functionRegistry.call(rewriter, FunctionId::DumpDecimal, ValueRange({isNull, low, high, scale}));
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         if (dateType.getUnit() == mlir::db::DateUnitAttr::millisecond) {
            functionRegistry.call(rewriter, FunctionId::DumpDateMillisecond, ValueRange({isNull, val}));
         } else {
            functionRegistry.call(rewriter, FunctionId::DumpDateDay, ValueRange({isNull, val}));
         }
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         FunctionId functionId;
         switch (timestampType.getUnit()) {
            case mlir::db::TimeUnitAttr::second: functionId = FunctionId::DumpTimestampSecond; break;
            case mlir::db::TimeUnitAttr::millisecond: functionId = FunctionId::DumpTimestampMillisecond; break;
            case mlir::db::TimeUnitAttr::microsecond: functionId = FunctionId::DumpTimestampMicrosecond; break;
            case mlir::db::TimeUnitAttr::nanosecond: functionId = FunctionId::DumpTimestampNanosecond; break;
         }
         functionRegistry.call(rewriter, functionId, ValueRange({isNull, val}));
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            functionRegistry.call(rewriter, FunctionId::DumpIntervalMonths, ValueRange({isNull, val}));
         } else {
            functionRegistry.call(rewriter, FunctionId::DumpIntervalDayTime, ValueRange({isNull, val}));
         }

      } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
         if (floatType.getWidth() < 64) {
            val = rewriter.create<FPExtOp>(loc, val, f64Type);
         }
         functionRegistry.call(rewriter, FunctionId::DumpFloat, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::StringType>()) {
         functionRegistry.call(rewriter, FunctionId::DumpString, ValueRange({isNull, val}));
      }
      rewriter.eraseOp(op);

      return success();
   }
};
class DumpIndexOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit DumpIndexOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DumpIndexOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      functionRegistry.call(rewriter, FunctionId::DumpIndex, operands[0]);

      rewriter.eraseOp(op);

      return success();
   }
};

class HashLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;
   Value xorImpl(OpBuilder& builder, Value v, Value totalHash) const {
      return builder.create<mlir::XOrOp>(builder.getUnknownLoc(), v, totalHash);
   }
   Value hashImpl(OpBuilder& builder, Value v, Value totalHash, Type originalType) const {
      //todo: more checks:
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      if (auto intType = v.getType().dyn_cast_or_null<mlir::IntegerType>()) {
         switch (intType.getWidth()) {
            case 1: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashBool, v)[0]);
            case 8: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt8, v)[0]);
            case 16: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt16, v)[0]);
            case 32: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt32, v)[0]);
            case 64: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt64, v)[0]);
            case 128: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt128, v)[0]);
         }
      } else if (auto floatType = v.getType().dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 32: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashFloat32, v)[0]);
            case 64: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashFloat64, v)[0]);
         }
      } else if (auto memrefType = v.getType().dyn_cast_or_null<mlir::MemRefType>()) {
         return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashBinary, v)[0]);
      } else if (auto tupleType = v.getType().dyn_cast_or_null<mlir::TupleType>()) {
         if (auto originalTupleType = originalType.dyn_cast_or_null<mlir::TupleType>()) {
            auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), v);
            size_t i = 0;
            for (auto v : unpacked->getResults()) {
               totalHash = hashImpl(builder, v, totalHash, originalTupleType.getType(i++));
            }
            return totalHash;
         } else if (auto dbType = originalType.dyn_cast_or_null<mlir::db::DBType>()) {
            assert(dbType.isNullable());
            auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), v);
            mlir::Value hashedIfNotNull = hashImpl(builder, unpacked.getResult(1), totalHash, dbType.getBaseType());
            return builder.create<mlir::SelectOp>(builder.getUnknownLoc(), unpacked.getResult(0),totalHash,hashedIfNotNull);
         }
         assert(false && "should not happen");
         return Value();
      }
      assert(false && "should not happen");
      return Value();
   }

   public:
   explicit HashLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::Hash::getOperationName(), 1, context), functionRegistry(functionRegistry) {}
   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::HashAdaptor hashAdaptor(operands);
      auto hashOp = mlir::cast<mlir::db::Hash>(op);
      hashAdaptor.val().getType().dump();
      hashOp.val().getType().dump();
      Value const0 = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
      rewriter.replaceOp(op, hashImpl(rewriter, hashAdaptor.val(),const0, hashOp.val().getType()));
      return success();
   }
};
} // namespace

void mlir::db::populateRuntimeSpecificScalarToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;

   auto ensureDate64 = [](mlir::db::DateType dateType, Value v, ConversionPatternRewriter& rewriter) {
      if (dateType.getUnit() == db::DateUnitAttr::day) {
         auto i64Type = IntegerType::get(rewriter.getContext(), 64);
         v = rewriter.template create<ZeroExtendIOp>(rewriter.getUnknownLoc(), v, i64Type);
         Value multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(i64Type, 24 * 60 * 60 * 1000));
         v = rewriter.template create<MulIOp>(rewriter.getUnknownLoc(), v, multiplier);
         return v;
      } else {
         return v;
      }
   };
   auto negateInterval = [](mlir::db::IntervalType dateType, Value v, ConversionPatternRewriter& rewriter) {
      Value multiplier = rewriter.template create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(v.getType(), -1));
      return rewriter.template create<MulIOp>(rewriter.getUnknownLoc(), v, multiplier);
   };
   auto transformDateBack = [](mlir::db::DateType dateType, Value v, ConversionPatternRewriter& rewriter) {
      if (dateType.getUnit() == db::DateUnitAttr::day) {
         auto i64Type = IntegerType::get(rewriter.getContext(), 64);
         auto i32Type = IntegerType::get(rewriter.getContext(), 32);
         Value multiplier = rewriter.template create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(i64Type, 24 * 60 * 60 * 1000));
         v = rewriter.template create<UnsignedDivIOp>(rewriter.getUnknownLoc(), v, multiplier);
         v = rewriter.template create<TruncateIOp>(rewriter.getUnknownLoc(), v, i32Type);
         return v;
      }
      return v;
   };
   auto identity = [](auto, Value v, auto&) { return v; };
   auto rightleft = [](Value left, Value right) { return std::vector<Value>({right, left}); };
   auto dateAddFunction = [&](Operation* op, mlir::db::DateType dateType, mlir::db::IntervalType intervalType, ConversionPatternRewriter& rewriter) {
      if(intervalType.getUnit()==mlir::db::IntervalUnitAttr::daytime) {
         return functionRegistry.getFunction(rewriter, FunctionId::TimestampAddMillis);
      }else{
         return functionRegistry.getFunction(rewriter, FunctionId::TimestampAddMonth);
      }
   };
   auto dateExtractFunction = [&](mlir::db::DateExtractOp dateExtractOp, mlir::db::DateType dateType, ConversionPatternRewriter& rewriter) {
      FunctionId functionId;
      switch (dateExtractOp.unit()) {
         case mlir::db::ExtractableTimeUnitAttr::second: functionId = FunctionId::DateExtractSecond; break;
         case mlir::db::ExtractableTimeUnitAttr::minute: functionId = FunctionId::DateExtractMinute; break;
         case mlir::db::ExtractableTimeUnitAttr::hour: functionId = FunctionId::DateExtractHour; break;
         case mlir::db::ExtractableTimeUnitAttr::dow: functionId = FunctionId::DateExtractDow; break;
         case mlir::db::ExtractableTimeUnitAttr::week: functionId = FunctionId::DateExtractWeek; break;
         case mlir::db::ExtractableTimeUnitAttr::day: functionId = FunctionId::DateExtractDay; break;
         case mlir::db::ExtractableTimeUnitAttr::month: functionId = FunctionId::DateExtractMonth; break;
         case mlir::db::ExtractableTimeUnitAttr::doy: functionId = FunctionId::DateExtractDoy; break;
         case mlir::db::ExtractableTimeUnitAttr::quarter: functionId = FunctionId::DateExtractQuarter; break;
         case mlir::db::ExtractableTimeUnitAttr::year: functionId = FunctionId::DateExtractYear; break;
         case mlir::db::ExtractableTimeUnitAttr::decade: functionId = FunctionId::DateExtractDecade; break;
         case mlir::db::ExtractableTimeUnitAttr::century: functionId = FunctionId::DateExtractCentury; break;
         case mlir::db::ExtractableTimeUnitAttr::millennium: functionId = FunctionId::DateExtractMillenium; break;
      }
      return functionRegistry.getFunction(rewriter, functionId);
   };

   patterns.insert<SimpleBinOpToFuncLowering<mlir::db::DateAddOp, mlir::db::DateType, mlir::db::IntervalType, mlir::db::DateType>>(
      patterns.getContext(), ensureDate64, identity, rightleft, dateAddFunction, transformDateBack);
   patterns.insert<SimpleBinOpToFuncLowering<mlir::db::DateSubOp, mlir::db::DateType, mlir::db::IntervalType, mlir::db::DateType>>(
      patterns.getContext(), ensureDate64, negateInterval, rightleft, dateAddFunction, transformDateBack);
   patterns.insert<SimpleUnOpToFuncLowering<mlir::db::DateExtractOp, mlir::db::DateType, mlir::db::IntType>>(
      patterns.getContext(), ensureDate64, dateExtractFunction, identity);
   patterns.insert<StringCmpOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<StringCastOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<DumpOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<DumpIndexOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<HashLowering>(functionRegistry, typeConverter, patterns.getContext());
}