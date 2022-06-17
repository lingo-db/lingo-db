#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/DB/Passes.h"
#include "runtime-defs/DateRuntime.h"
#include "runtime-defs/DumpRuntime.h"
#include "runtime-defs/StringRuntime.h"

mlir::db::RuntimeFunction* mlir::db::RuntimeFunctionRegistry::lookup(std::string name) {
   return registeredFunctions[name].get();
}
static mlir::Value dateAddImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   if (originalArgumentTypes[1].cast<mlir::db::IntervalType>().getUnit() == mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::AddIOp>(loc, loweredArguments);
   } else {
      return rt::DateRuntime::addMonths(rewriter, loc)(loweredArguments)[0];
   }
}
static mlir::Value absIntImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   mlir::Value val = loweredArguments[0];
   mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, resType, rewriter.getIntegerAttr(resType, 0));
   mlir::Value negated = rewriter.create<mlir::arith::SubIOp>(loc, zero, val);
   mlir::Value ltZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, val, zero);
   return rewriter.create<mlir::arith::SelectOp>(loc, ltZero, negated, val);
}
static mlir::Value dateSubImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   if (originalArgumentTypes[1].cast<mlir::db::IntervalType>().getUnit() == mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::SubIOp>(loc, loweredArguments);
   } else {
      return rt::DateRuntime::subtractMonths(rewriter, loc)(loweredArguments)[0];
   }
}
static mlir::Value matchPart(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lastMatchEnd, std::string pattern, mlir::Value str, mlir::Value end) {
   if (pattern.empty()) {
      if (!lastMatchEnd) {
         lastMatchEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      }
      return lastMatchEnd;
   }
   mlir::Value needleValue = builder.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(builder.getContext()), pattern);
   if (lastMatchEnd) {
      mlir::Value matchEnd = rt::StringRuntime::findMatch(builder, loc)(mlir::ValueRange{str, needleValue, lastMatchEnd, end})[0];
      return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), matchEnd);
   } else {
      mlir::Value startsWithPattern = rt::StringRuntime::startsWith(builder, loc)(mlir::ValueRange{str, needleValue})[0];
      mlir::Value patternLen = builder.create<mlir::arith::ConstantIndexOp>(loc, pattern.size());
      mlir::Value invalidPos = builder.create<mlir::arith::ConstantIndexOp>(loc, 0x8000000000000000);

      mlir::Value matchEnd = builder.create<mlir::arith::SelectOp>(loc, startsWithPattern, patternLen, invalidPos);

      return matchEnd;
   }
}
static mlir::Value constLikeImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   mlir::Value str = loweredArguments[0];
   mlir::Value patternValue = loweredArguments[1];
   if (auto constStrOp = mlir::dyn_cast_or_null<mlir::util::CreateConstVarLen>(patternValue.getDefiningOp())) {
      auto pattern = constStrOp.str().str();
      size_t pos = 0;
      std::string currentSubPattern;
      mlir::Value lastMatchEnd;
      mlir::Value end = rewriter.create<util::VarLenGetLen>(loc, rewriter.getIndexType(), str);
      bool flexible=false;
      while (pos < pattern.size()) {
         if (pattern[pos] == '\\') {
            currentSubPattern += pattern[pos + 1];
            pos += 2;
         } else if (pattern[pos] == '.') {
            //match current pattern
            lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
            mlir::Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

            lastMatchEnd = rewriter.create<arith::AddIOp>(loc, lastMatchEnd, one);
            currentSubPattern = "";
            //lastMatchEnd+=1
            pos += 1;
         } else if (pattern[pos] == '%') {
            flexible=true;
            lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
            currentSubPattern = "";
            pos += 1;
         } else {
            currentSubPattern += pattern[pos];
            pos += 1;
         }
      }
      if (!currentSubPattern.empty()) {
         mlir::Value needleValue = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(rewriter.getContext()), currentSubPattern);
         mlir::Value endsWith = rt::StringRuntime::endsWith(rewriter, loc)({str, needleValue})[0];
         if (lastMatchEnd) {
            mlir::Value patternLength = rewriter.create<mlir::arith::ConstantIndexOp>(loc, currentSubPattern.size());
            lastMatchEnd = rewriter.create<mlir::arith::AddIOp>(loc, lastMatchEnd, patternLength);
            mlir::Value previousMatchesEnd = rewriter.create<mlir::arith::CmpIOp>(loc, flexible?arith::CmpIPredicate::ule:arith::CmpIPredicate::eq, lastMatchEnd, end);
            return rewriter.create<mlir::arith::AndIOp>(loc, previousMatchesEnd, endsWith);
         } else {
            return endsWith;
         }
         lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
      }

      return rewriter.create<mlir::arith::CmpIOp>(loc, flexible?arith::CmpIPredicate::ule:arith::CmpIPredicate::eq, lastMatchEnd, end);
   }

   return Value();
}
static mlir::Value dumpValuesImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
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
      rt::DumpRuntime::dumpIndex(rewriter, loc)(loweredArguments[0]);
   } else if (isIntegerType(baseType, 1)) {
      rt::DumpRuntime::dumpBool(rewriter, loc)({isNull, val});
   } else if (auto intWidth = getIntegerWidth(baseType, false)) {
      if (intWidth < 64) {
         val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
      }
      rt::DumpRuntime::dumpInt(rewriter, loc)({isNull, val});
   } else if (auto uIntWidth = getIntegerWidth(baseType, true)) {
      if (uIntWidth < 64) {
         val = rewriter.create<arith::ExtUIOp>(loc, i64Type, val);
      }
      rt::DumpRuntime::dumpUInt(rewriter, loc)({isNull, val});
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
      rt::DumpRuntime::dumpDecimal(rewriter, loc)({isNull, low, high, scale});
   } else if (auto dateType = baseType.dyn_cast_or_null<mlir::db::DateType>()) {
      rt::DumpRuntime::dumpDate(rewriter, loc)({isNull, val});
   } else if (auto timestampType = baseType.dyn_cast_or_null<mlir::db::TimestampType>()) {
      switch (timestampType.getUnit()) {
         case mlir::db::TimeUnitAttr::second: rt::DumpRuntime::dumpTimestampSecond(rewriter, loc)({isNull, val}); break;
         case mlir::db::TimeUnitAttr::millisecond: rt::DumpRuntime::dumpTimestampMilliSecond(rewriter, loc)({isNull, val}); break;
         case mlir::db::TimeUnitAttr::microsecond: rt::DumpRuntime::dumpTimestampMicroSecond(rewriter, loc)({isNull, val}); break;
         case mlir::db::TimeUnitAttr::nanosecond: rt::DumpRuntime::dumpTimestampNanoSecond(rewriter, loc)({isNull, val}); break;
      }
   } else if (auto intervalType = baseType.dyn_cast_or_null<mlir::db::IntervalType>()) {
      if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
         rt::DumpRuntime::dumpIntervalMonths(rewriter, loc)({isNull, val});
      } else {
         rt::DumpRuntime::dumpIntervalDaytime(rewriter, loc)({isNull, val});
      }
   } else if (auto floatType = baseType.dyn_cast_or_null<mlir::FloatType>()) {
      if (floatType.getWidth() < 64) {
         val = rewriter.create<arith::ExtFOp>(loc, f64Type, val);
      }
      rt::DumpRuntime::dumpFloat(rewriter, loc)({isNull, val});
   } else if (baseType.isa<mlir::db::StringType>()) {
      rt::DumpRuntime::dumpString(rewriter, loc)({isNull, val});
   } else if (auto charType = baseType.dyn_cast_or_null<mlir::db::CharType>()) {
      Value numBytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(charType.getBytes()));
      if (charType.getBytes() < 8) {
         val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
      }
      rt::DumpRuntime::dumpChar(rewriter, loc)({isNull, val, numBytes});
   }
   return mlir::Value();
}
std::shared_ptr<mlir::db::RuntimeFunctionRegistry> mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(mlir::MLIRContext* context) {
   auto builtinRegistry = std::make_shared<RuntimeFunctionRegistry>(context);
   builtinRegistry->add("DumpValue").handlesNulls().matchesTypes({RuntimeFunction::anyType}, RuntimeFunction::noReturnType).implementedAs(dumpValuesImpl);
   auto resTypeIsI64 = [](mlir::Type t, mlir::TypeRange) { return t.isInteger(64); };
   auto resTypeIsBool = [](mlir::Type t, mlir::TypeRange) { return t.isInteger(1); };
   builtinRegistry->add("Substring").implementedAs(rt::StringRuntime::substr).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("Like").implementedAs(rt::StringRuntime::like).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool);
   builtinRegistry->add("ConstLike").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool).implementedAs(constLikeImpl).needsWrapping();

   builtinRegistry->add("ExtractFromDate").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, resTypeIsI64);
   builtinRegistry->add("ExtractYearFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(rt::DateRuntime::extractYear);
   builtinRegistry->add("ExtractMonthFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(rt::DateRuntime::extractMonth);
   builtinRegistry->add("ExtractDayFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(rt::DateRuntime::extractDay);
   builtinRegistry->add("DateAdd").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateAddImpl);
   builtinRegistry->add("AbsInt").handlesInvalid().matchesTypes({RuntimeFunction::intLike}, RuntimeFunction::matchesArgument()).implementedAs(absIntImpl);
   builtinRegistry->add("DateSubtract").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateSubImpl);
   return builtinRegistry;
}