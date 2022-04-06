#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/DB/Passes.h"
#include "runtime-defs/DateRuntime.h"
#include "runtime-defs/DumpRuntime.h"
#include "runtime-defs/StringRuntime.h"

mlir::db::RuntimeFunction* mlir::db::RuntimeFunctionRegistry::lookup(std::string name) {
   return registeredFunctions[name].get();
}
static mlir::Value dateAddImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter) {
   using namespace mlir;
   if (originalArgumentTypes[1].cast<mlir::db::IntervalType>().getUnit() == mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::AddIOp>(rewriter.getUnknownLoc(), loweredArguments);
   } else {
      return runtime::DateRuntime::addMonths(rewriter, rewriter.getUnknownLoc())(loweredArguments)[0];
   }
}
static mlir::Value dateSubImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter) {
   using namespace mlir;
   if (originalArgumentTypes[1].cast<mlir::db::IntervalType>().getUnit() == mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::SubIOp>(rewriter.getUnknownLoc(), loweredArguments);
   } else {
      return runtime::DateRuntime::subtractMonths(rewriter, rewriter.getUnknownLoc())(loweredArguments)[0];
   }
}
static mlir::Value dumpValuesImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter) {
   using namespace mlir;
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
}
std::shared_ptr<mlir::db::RuntimeFunctionRegistry> mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(mlir::MLIRContext* context) {
   auto builtinRegistry = std::make_shared<RuntimeFunctionRegistry>(context);
   builtinRegistry->add("DumpValue").handlesNulls().matchesTypes({RuntimeFunction::anyType}, RuntimeFunction::noReturnType).implementedAs(dumpValuesImpl);
   auto resTypeIsI64 = [](mlir::Type t, mlir::TypeRange) { return t.isInteger(64); };
   builtinRegistry->add("Substring").implementedAs(runtime::StringRuntime::substr).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("ExtractFromDate").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, resTypeIsI64);
   builtinRegistry->add("ExtractYearFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(runtime::DateRuntime::extractYear);
   builtinRegistry->add("ExtractMonthFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(runtime::DateRuntime::extractMonth);
   builtinRegistry->add("ExtractDayFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(runtime::DateRuntime::extractDay);
   builtinRegistry->add("DateAdd").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateAddImpl);
   builtinRegistry->add("DateSubtract").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateSubImpl);
   return builtinRegistry;
}