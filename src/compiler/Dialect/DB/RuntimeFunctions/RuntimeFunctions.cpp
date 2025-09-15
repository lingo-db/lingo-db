#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"

#include "lingodb/compiler/Dialect/DB/Passes.h"
#include "lingodb/compiler/runtime/DateRuntime.h"
#include "lingodb/compiler/runtime/DecimalRuntime.h"
#include "lingodb/compiler/runtime/DumpRuntime.h"
#include "lingodb/compiler/runtime/FloatRuntime.h"
#include "lingodb/compiler/runtime/IntegerRuntime.h"
#include "lingodb/compiler/runtime/StringRuntime.h"
#include "lingodb/compiler/runtime/Timing.h"
#include "lingodb/runtime/DateRuntime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

lingodb::compiler::dialect::db::RuntimeFunction* lingodb::compiler::dialect::db::RuntimeFunctionRegistry::lookup(std::string name) {
   return registeredFunctions[name].get();
}
namespace {
using namespace lingodb::compiler::runtime;
using namespace lingodb::compiler::dialect;
mlir::Value dateAddImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   if (mlir::cast<db::IntervalType>(originalArgumentTypes[1]).getUnit() == db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::AddIOp>(loc, loweredArguments);
   } else {
      return DateRuntime::addMonths(rewriter, loc)(loweredArguments)[0];
   }
}
mlir::Value absImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   mlir::Value val = loweredArguments[0];
   mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, typeConverter->convertType(resType), rewriter.getIntegerAttr(typeConverter->convertType(resType), 0));
   mlir::Value negated = rewriter.create<mlir::arith::SubIOp>(loc, zero, val);
   mlir::Value ltZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, val, zero);
   return rewriter.create<mlir::arith::SelectOp>(loc, ltZero, negated, val);
}
mlir::Value sqrtImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;

   mlir::Value val = loweredArguments[0];
   if (mlir::isa<mlir::IntegerType>(val.getType())) {
      mlir::Value res = IntegerRuntime::sqrt(rewriter, loc)(val)[0]; //todo: for decimals
      if (res.getType() != val.getType()) {
         res = rewriter.create<mlir::arith::TruncIOp>(loc, val.getType(), res);
      }
      return res;

   } else {
      return FloatRuntime::sqrt(rewriter, loc)(val)[0];
   }
}
mlir::Value dateSubImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   if (mlir::cast<db::IntervalType>(originalArgumentTypes[1]).getUnit() == db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::SubIOp>(loc, loweredArguments);
   } else {
      return DateRuntime::subtractMonths(rewriter, loc)(loweredArguments)[0];
   }
}
mlir::Value matchPart(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lastMatchEnd, std::string pattern, mlir::Value str, mlir::Value end, bool flexibleStart) {
   if (pattern.empty()) {
      if (!lastMatchEnd) {
         lastMatchEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      }
      return lastMatchEnd;
   }
   mlir::Value needleValue = builder.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(builder.getContext()), pattern);
   mlir::Value patternLen = builder.create<mlir::arith::ConstantIndexOp>(loc, pattern.size());
   if (lastMatchEnd) {
      mlir::Value matchEnd;
      if (flexibleStart) {
         // match first occurrence of pattern beginning at position lastMatchEnd
         matchEnd = StringRuntime::findMatch(builder, loc)(mlir::ValueRange{str, needleValue, lastMatchEnd, end})[0];
      } else {
         // pattern must start at position lastMatchEnd
         mlir::Value patternEnd = builder.create<mlir::arith::AddIOp>(loc, lastMatchEnd, patternLen);
         matchEnd = StringRuntime::findMatch(builder, loc)(mlir::ValueRange{str, needleValue, lastMatchEnd, patternEnd})[0];
      }
      return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), matchEnd);
   } else {
      // match the start of the string
      mlir::Value startsWithPattern = StringRuntime::startsWith(builder, loc)(mlir::ValueRange{str, needleValue})[0];
      mlir::Value invalidPos = builder.create<mlir::arith::ConstantIndexOp>(loc, 0x8000000000000000);
      return builder.create<mlir::arith::SelectOp>(loc, startsWithPattern, patternLen, invalidPos);
   }
}
mlir::Value constLikeImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   mlir::Value str = loweredArguments[0];
   mlir::Value patternValue = loweredArguments[1];
   if (auto constStrOp = mlir::dyn_cast_or_null<util::CreateConstVarLen>(patternValue.getDefiningOp())) {
      auto pattern = constStrOp.getStr();
      size_t pos = 0;
      std::string currentSubPattern;
      mlir::Value lastMatchEnd;
      mlir::Value end = rewriter.create<util::VarLenGetLen>(loc, rewriter.getIndexType(), str);
      bool flexibleStart = false; // start of the next sub pattern is flexible (sub pattern was preceded by '%')

      // find beginning of suffix pattern (sub pattern after last '%' -> must exactly match the end of the string)
      size_t suffixBegin = pattern.size();
      size_t suffixLen = 0;
      for (size_t p = 0; p < pattern.size(); ++p) {
         ++suffixLen;
         if (pattern[p] == '\\') {
            ++p;
         } else if (pattern[p] == '%') {
            suffixBegin = p + 1;
            suffixLen = 0;
         }
      }
      while (pos < pattern.size()) {
         if (pos == suffixBegin) { //remaining sub patterns match the suffix of the string
            mlir::Value remainingPatternLen = rewriter.create<mlir::arith::ConstantIndexOp>(loc, suffixLen);
            mlir::Value suffixBeginPos = rewriter.create<mlir::arith::SubIOp>(loc, end, remainingPatternLen);
            mlir::Value rightShiftSuffixBegin = rewriter.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::ule, lastMatchEnd, suffixBeginPos);
            // move lastMatchEnd to position where suffix must start (only move position to the right to prevent overlapping with already matched part)
            lastMatchEnd = rewriter.create<mlir::arith::SelectOp>(loc, rightShiftSuffixBegin, suffixBeginPos, lastMatchEnd);
            flexibleStart = false;
         }
         if (pattern[pos] == '\\') {
            currentSubPattern += pattern[pos + 1];
            pos += 2;
         } else if (pattern[pos] == '%') {
            // in case of a simple 'starts with' pattern (i.e. 'abc%'), we can simply return the result of the startsWith call
            if (!lastMatchEnd && pos == pattern.size() - 1) {
               // we only need to check that the prefix matches
               mlir::Value needleValue = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()), currentSubPattern);
               return StringRuntime::startsWith(rewriter, loc)(mlir::ValueRange{str, needleValue})[0];
            }
            lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end, flexibleStart);
            currentSubPattern = "";
            flexibleStart = true;
            pos += 1;
         } else if (pattern[pos] == '_') {
            //match current pattern
            lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end, flexibleStart);

            auto nextChar = StringRuntime::nextChar(rewriter, loc)(mlir::ValueRange{str, lastMatchEnd})[0];
            lastMatchEnd = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), nextChar);

            currentSubPattern = "";
            pos += 1;
         } else {
            currentSubPattern += pattern[pos];
            pos += 1;
         }
      }

      if (!currentSubPattern.empty()) {
         lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end, flexibleStart);
      }
      return rewriter.create<mlir::arith::CmpIOp>(loc, flexibleStart ? arith::CmpIPredicate::ule : arith::CmpIPredicate::eq, lastMatchEnd, end);
   }

   return Value();
}
mlir::Value dumpValuesImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   const auto i128Type = IntegerType::get(rewriter.getContext(), 128);
   const auto i64Type = IntegerType::get(rewriter.getContext(), 64);
   const auto i32Type = IntegerType::get(rewriter.getContext(), 32);
   const auto f64Type = Float64Type::get(rewriter.getContext());
   const auto nullableType = mlir::dyn_cast_or_null<db::NullableType>(originalArgumentTypes[0]);
   const auto baseType = getBaseType(originalArgumentTypes[0]);

   Value isNull;
   Value val;
   if (nullableType) {
      auto unPackOp = rewriter.create<util::UnPackOp>(loc, loweredArguments[0]);
      isNull = unPackOp.getVals()[0];
      val = unPackOp.getVals()[1];
   } else {
      isNull = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      val = loweredArguments[0];
   }
   if (mlir::isa<mlir::IndexType>(baseType)) {
      DumpRuntime::dumpIndex(rewriter, loc)(loweredArguments[0]);
   } else if (isIntegerType(baseType, 1)) {
      DumpRuntime::dumpBool(rewriter, loc)({isNull, val});
   } else if (auto intWidth = getIntegerWidth(baseType, false)) {
      if (intWidth < 64) {
         val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
      }
      DumpRuntime::dumpInt(rewriter, loc)({isNull, val});
   } else if (auto uIntWidth = getIntegerWidth(baseType, true)) {
      if (uIntWidth < 64) {
         val = rewriter.create<arith::ExtUIOp>(loc, i64Type, val);
      }
      DumpRuntime::dumpUInt(rewriter, loc)({isNull, val});
   } else if (auto decType = mlir::dyn_cast_or_null<db::DecimalType>(baseType)) {
      if (mlir::cast<mlir::IntegerType>(typeConverter->convertType(decType)).getWidth() < 128) {
         auto converted = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(128), val);
         val = converted;
      }
      Value low = rewriter.create<arith::TruncIOp>(loc, i64Type, val);
      Value shift = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
      Value scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
      Value high = rewriter.create<arith::ShRUIOp>(loc, i128Type, val, shift);
      high = rewriter.create<arith::TruncIOp>(loc, i64Type, high);
      DumpRuntime::dumpDecimal(rewriter, loc)({isNull, low, high, scale});
   } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(baseType)) {
      DumpRuntime::dumpDate(rewriter, loc)({isNull, val});
   } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(baseType)) {
      switch (timestampType.getUnit()) {
         case db::TimeUnitAttr::second: DumpRuntime::dumpTimestampSecond(rewriter, loc)({isNull, val}); break;
         case db::TimeUnitAttr::millisecond: DumpRuntime::dumpTimestampMilliSecond(rewriter, loc)({isNull, val}); break;
         case db::TimeUnitAttr::microsecond: DumpRuntime::dumpTimestampMicroSecond(rewriter, loc)({isNull, val}); break;
         case db::TimeUnitAttr::nanosecond: DumpRuntime::dumpTimestampNanoSecond(rewriter, loc)({isNull, val}); break;
      }
   } else if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(baseType)) {
      if (intervalType.getUnit() == db::IntervalUnitAttr::months) {
         DumpRuntime::dumpIntervalMonths(rewriter, loc)({isNull, val});
      } else {
         DumpRuntime::dumpIntervalDaytime(rewriter, loc)({isNull, val});
      }
   } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(baseType)) {
      if (floatType.getWidth() < 64) {
         val = rewriter.create<arith::ExtFOp>(loc, f64Type, val);
      }
      DumpRuntime::dumpFloat(rewriter, loc)({isNull, val});
   } else if (mlir::isa<db::StringType>(baseType)) {
      DumpRuntime::dumpString(rewriter, loc)({isNull, val});
   } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(baseType)) {
      Value numBytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(charType.getLen()));
      if (charType.getLen() <= 1 && val.getType() != i32Type) {
         val = rewriter.create<arith::ExtSIOp>(loc, i32Type, val);
      }
      DumpRuntime::dumpChar(rewriter, loc)({isNull, val, numBytes});
   }
   return mlir::Value();
}
mlir::LogicalResult dateAddFoldFn(mlir::TypeRange types, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
   if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(types[0])) {
      if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(types[1])) {
         auto leftIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[0]);
         auto rightIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[1]);
         if (leftIntAttr && rightIntAttr) {
            if (intervalType.getUnit() == db::IntervalUnitAttr::daytime) {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), leftIntAttr.getValue() + rightIntAttr.getValue()));
               return mlir::success();
            } else {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), lingodb::runtime::DateRuntime::addMonths(leftIntAttr.getInt(), rightIntAttr.getInt())));
               return mlir::success();
            }
         }
      }
   }
   return mlir::failure();
}
mlir::LogicalResult dateSubtractFoldFn(mlir::TypeRange types, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
   if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(types[0])) {
      if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(types[1])) {
         auto leftIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[0]);
         auto rightIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[1]);
         if (leftIntAttr && rightIntAttr) {
            if (intervalType.getUnit() == db::IntervalUnitAttr::daytime) {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), leftIntAttr.getValue() - rightIntAttr.getValue()));
               return mlir::success();
            } else {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), lingodb::runtime::DateRuntime::subtractMonths(leftIntAttr.getInt(), rightIntAttr.getInt())));
               return mlir::success();
            }
         }
      }
   }
   return mlir::failure();
}
} // namespace
std::shared_ptr<db::RuntimeFunctionRegistry> db::RuntimeFunctionRegistry::getBuiltinRegistry(mlir::MLIRContext* context) {
   auto builtinRegistry = std::make_shared<RuntimeFunctionRegistry>(context);
   builtinRegistry->add("DumpValue").handlesNulls().matchesTypes({RuntimeFunction::anyType}, RuntimeFunction::noReturnType).implementedAs(dumpValuesImpl);
   auto resTypeIsI64 = [](mlir::Type t, mlir::TypeRange) { return t.isInteger(64); };
   auto resTypeIsF64 = [](mlir::Type t, mlir::TypeRange) { return t.isF64(); };
   auto resTypeIsBool = [](mlir::Type t, mlir::TypeRange) { return t.isInteger(1); };
   auto resTypeIsIndex = [](mlir::Type t, mlir::TypeRange) { return t.isIndex(); };
   builtinRegistry->add("Substring").implementedAs(StringRuntime::substr).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("StringFind").implementedAs(StringRuntime::findNext).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::intLike}, resTypeIsI64);
   builtinRegistry->add("StringLength").implementedAs(StringRuntime::len).matchesTypes({RuntimeFunction::stringLike}, resTypeIsI64);

   builtinRegistry->add( "ToUpper").implementedAs(StringRuntime::toUpper).matchesTypes({RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("Concatenate").implementedAs(StringRuntime::concat).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());

   builtinRegistry->add("RegexpReplace").implementedAs(StringRuntime::regexpReplace).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("Like").implementedAs(StringRuntime::like).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool);
   builtinRegistry->add("ConstLike").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool).implementedAs(constLikeImpl).needsWrapping();
   builtinRegistry->add("RoundDecimal").matchesTypes({RuntimeFunction::anyDecimal, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument()).needsWrapping().implementedAs([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) -> mlir::Value {
      mlir::Value s = rewriter.create<mlir::arith::ConstantIndexOp>(loc, mlir::cast<lingodb::compiler::dialect::db::DecimalType>(originalArgumentTypes[0]).getS());
      mlir::Value res = DecimalRuntime::round(rewriter, loc)(mlir::ValueRange({loweredArguments[0], loweredArguments[1], s}))[0];
      auto loweredResType = typeConverter->convertType(resType);
      if (!loweredResType.isInteger(128)) {
         res = rewriter.create<mlir::arith::TruncIOp>(loc, loweredResType, res);
      }
      return res;
   });
   builtinRegistry->add("RoundInt64").implementedAs(IntegerRuntime::round64).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("startTiming").implementedAs(Timing::start).matchesTypes({}, resTypeIsI64);
   builtinRegistry->add("startPerf").implementedAs(Timing::startPerf).matchesTypes({}, RuntimeFunction::noReturnType);
   builtinRegistry->add("stopPerf").implementedAs(Timing::stopPerf).matchesTypes({}, RuntimeFunction::noReturnType);
   builtinRegistry->add("stopTiming").implementedAs(Timing::stop).matchesTypes({RuntimeFunction::intLike}, RuntimeFunction::noReturnType);
   builtinRegistry->add("RoundInt32").implementedAs(IntegerRuntime::round32).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("RoundInt16").implementedAs(IntegerRuntime::round16).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("RoundInt8").implementedAs(IntegerRuntime::round8).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("RandomInRange").implementedAs(IntegerRuntime::randomInRange).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());

   /*
    * DateDiff and ExtractFromDate are special, as no implementation is provided through .implementedAs()
    * Instead during the pass OptimizeRuntimeFunctions they are optimized to call the function with the correct unit (e.g. Extract*Year*FromDate)
    */
   builtinRegistry->add("DateTrunc").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateTrunc);
   builtinRegistry->add("DateDiff").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).needsWrapping();
   builtinRegistry->add("DateDiffDay").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffDay);
   builtinRegistry->add("DateDiffHour").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffHour);
   builtinRegistry->add("DateDiffMinute").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffMinute);
   builtinRegistry->add("DateDiffSecond").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffSecond);
   builtinRegistry->add("ExtractFromDate").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, resTypeIsI64).needsWrapping();
   builtinRegistry->add("ExtractYearFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractYear);
   builtinRegistry->add("ExtractMonthFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractMonth);
   builtinRegistry->add("ExtractDayFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractDay);
   builtinRegistry->add("ExtractHourFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractHour);
   builtinRegistry->add("ExtractMinuteFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractMinute);
   builtinRegistry->add("ExtractSecondFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractSecond);
   builtinRegistry->add("DateAdd").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateAddImpl).folds(dateAddFoldFn);
   builtinRegistry->add("DateSubtract").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateSubImpl).folds(dateSubtractFoldFn);

   builtinRegistry->add("AbsInt").handlesInvalid().matchesTypes({RuntimeFunction::intLike}, RuntimeFunction::matchesArgument()).implementedAs(absImpl);
   builtinRegistry->add("AbsDecimal").handlesInvalid().matchesTypes({RuntimeFunction::anyDecimal}, RuntimeFunction::matchesArgument()).implementedAs(absImpl);
   builtinRegistry->add("Sqrt").needsWrapping().matchesTypes({RuntimeFunction::anyNumber}, RuntimeFunction::matchesArgument()).implementedAs(sqrtImpl);
   builtinRegistry->add("Sin").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::sin);
   builtinRegistry->add("Log").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::log);
   builtinRegistry->add("Exp").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::exp);
   builtinRegistry->add("Erf").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::erf);
   builtinRegistry->add("Cos").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::cos);
   builtinRegistry->add("ASin").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::arcsin);
   builtinRegistry->add("Hash").matchesTypes({RuntimeFunction::anyType}, resTypeIsIndex).needsWrapping().implementedAs([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) -> mlir::Value {
      return rewriter.create<lingodb::compiler::dialect::db::Hash>(loc, loweredArguments[0]);
   });
   builtinRegistry->add("CombineHashes").matchesTypes({RuntimeFunction::onlyIndex, RuntimeFunction::onlyIndex}, resTypeIsIndex).needsWrapping().implementedAs([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) -> mlir::Value {
      return rewriter.create<util::HashCombine>(loc, rewriter.getIndexType(), loweredArguments[0], loweredArguments[1]);
   });
   return builtinRegistry;
}
