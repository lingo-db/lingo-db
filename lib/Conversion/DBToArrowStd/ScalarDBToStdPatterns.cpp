#include "mlir-support/mlir-support.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/DBToArrowStd/NullHandler.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
namespace {
support::TimeUnit toTimeUnit(mlir::db::TimeUnitAttr attr) {
   switch (attr) {
      case mlir::db::TimeUnitAttr::second: return support::TimeUnit::SECOND;

      case mlir::db::TimeUnitAttr::millisecond: return support::TimeUnit::MILLI;
      case mlir::db::TimeUnitAttr::microsecond: return support::TimeUnit::MICRO;
      case mlir::db::TimeUnitAttr::nanosecond: return support::TimeUnit::NANO;
   }
   return support::TimeUnit::SECOND;
}
class NotOpLowering : public ConversionPattern {
   public:
   explicit NotOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::NotOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto notOp = cast<mlir::db::NotOp>(op);
      Value trueValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      mlir::db::DBType valType = notOp.val().getType().cast<mlir::db::DBType>();
      if (valType.isNullable()) {
         auto tupleType = typeConverter->convertType(notOp.val().getType());
         Value val = rewriter.create<util::GetTupleOp>(rewriter.getUnknownLoc(), rewriter.getI1Type(), operands[0], 1);
         val = rewriter.create<XOrOp>(rewriter.getUnknownLoc(), val, trueValue);
         rewriter.replaceOpWithNewOp<util::SetTupleOp>(op, tupleType, operands[0], val, 1);
         return success();
      } else {
         rewriter.replaceOpWithNewOp<XOrOp>(op, operands[0], trueValue);
         return success();
      }
   }
};
class AndOpLowering : public ConversionPattern {
   public:
   explicit AndOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::AndOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto andOp = cast<mlir::db::AndOp>(op);

      Value result;
      Value isNull;
      auto loc = rewriter.getUnknownLoc();
      Value falseValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      Value trueValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

      for (size_t i = 0; i < operands.size(); i++) {
         auto currType = andOp.vals()[i].getType();
         bool currNullable = currType.dyn_cast_or_null<mlir::db::DBType>().isNullable();
         Value currNull;
         Value currVal;
         if (currNullable) {
            TupleType tupleType = typeConverter->convertType(currType).dyn_cast_or_null<TupleType>();
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, tupleType.getTypes(), operands[i]);
            currNull = unPackOp.vals()[0];
            currVal = unPackOp.vals()[1];
         } else {
            currVal = operands[i];
         }
         if (i == 0) {
            if (currNullable) {
               result = rewriter.create<SelectOp>(loc, currNull, trueValue, currVal);
            } else {
               result = currVal;
            }
            isNull = currNull;
         } else {
            if (currNullable) {
               if (isNull) {
                  isNull = rewriter.create<OrOp>(loc, isNull, currNull);
               } else {
                  isNull = currNull;
               }
            }
            if (currNullable) {
               result = rewriter.create<SelectOp>(loc, currNull, result, rewriter.create<SelectOp>(loc, currVal, result, falseValue));
            } else {
               result = rewriter.create<SelectOp>(loc, currVal, result, falseValue);
            }
         }
      }
      if (andOp.getResult().getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
         isNull = rewriter.create<SelectOp>(loc, result, isNull, falseValue);
         Value combined = rewriter.create<mlir::util::PackOp>(loc, typeConverter->convertType(andOp.getResult().getType()), ValueRange({isNull, result}));
         rewriter.replaceOp(op, combined);
      } else {
         rewriter.replaceOp(op, result);
      }
      return success();
   }
};
class OrOpLowering : public ConversionPattern {
   public:
   explicit OrOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::OrOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto orOp = cast<mlir::db::OrOp>(op);

      Value result;
      Value isNull;
      auto loc = rewriter.getUnknownLoc();
      Value falseValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      Value trueValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

      for (size_t i = 0; i < operands.size(); i++) {
         auto currType = orOp.vals()[i].getType();
         bool currNullable = currType.dyn_cast_or_null<mlir::db::DBType>().isNullable();
         Value currNull;
         Value currVal;
         if (currNullable) {
            TupleType tupleType = typeConverter->convertType(currType).dyn_cast_or_null<TupleType>();
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, tupleType.getTypes(), operands[i]);
            currNull = unPackOp.vals()[0];
            currVal = unPackOp.vals()[1];
         } else {
            currVal = operands[i];
         }
         if (i == 0) {
            if (currNullable) {
               result = rewriter.create<SelectOp>(loc, currNull, falseValue, currVal);
            } else {
               result = currVal;
            }
            isNull = currNull;
         } else {
            if (currNullable) {
               if (isNull) {
                  isNull = rewriter.create<OrOp>(loc, isNull, currNull);
               } else {
                  isNull = currNull;
               }
            }
            if (currNullable) {
               result = rewriter.create<SelectOp>(loc, currNull, result, rewriter.create<SelectOp>(loc, currVal, trueValue, result));
            } else {
               result = rewriter.create<SelectOp>(loc, currVal, trueValue, result);
            }
         }
      }
      if (orOp.getResult().getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
         isNull = rewriter.create<SelectOp>(loc, result, falseValue, isNull);
         Value combined = rewriter.create<mlir::util::PackOp>(loc, typeConverter->convertType(orOp.getResult().getType()), ValueRange({isNull, result}));
         rewriter.replaceOp(op, combined);
      } else {
         rewriter.replaceOp(op, result);
      }
      return success();
   }
};

template <class OpClass, class OperandType, class StdOpClass>
class BinOpLowering : public ConversionPattern {
   public:
   explicit BinOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, OpClass::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto addOp = cast<OpClass>(op);
      db::NullHandler nullHandler(*typeConverter, rewriter);
      auto type = addOp.left().getType();
      Type resType = addOp.result().getType().template cast<db::DBType>().getBaseType();
      Value left = nullHandler.getValue(addOp.left());
      Value right = nullHandler.getValue(addOp.right());
      if (type.template isa<OperandType>()) {
         Value replacement = rewriter.create<StdOpClass>(rewriter.getUnknownLoc(), typeConverter->convertType(resType), left, right);
         rewriter.replaceOp(op, nullHandler.combineResult(replacement));
         return success();
      }
      return failure();
   }
};
template <class DBOp, class Op>
class DecimalOpScaledLowering : public ConversionPattern {
   public:
   explicit DecimalOpScaledLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, DBOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto addOp = cast<DBOp>(op);
      Value left = addOp.left();
      Value right = addOp.right();
      if (left.getType() != right.getType()) {
         return failure();
      }
      auto type = left.getType();
      if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         auto [low, high] = support::getDecimalScaleMultiplier(decimalType.getS());
         std::vector<uint64_t> parts = {low, high};
         auto stdType = typeConverter->convertType(type);
         auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), stdType, rewriter.getIntegerAttr(stdType, APInt(128, parts)));
         left = rewriter.create<mlir::MulIOp>(rewriter.getUnknownLoc(), stdType, left, multiplier);
         rewriter.replaceOpWithNewOp<Op>(op, stdType, left, right);
         return success();
      }
      return failure();
   }
};
class IsNullOpLowering : public ConversionPattern {
   public:
   explicit IsNullOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::IsNullOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto nullOp = cast<mlir::db::IsNullOp>(op);
      auto tupleType = typeConverter->convertType(nullOp.val().getType()).dyn_cast_or_null<TupleType>();
      auto unPackOp = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), tupleType.getTypes(), nullOp.val());
      rewriter.replaceOp(op, unPackOp.vals()[0]);
      return success();
   }
};
class CombineNullOpLowering : public ConversionPattern {
   public:
   explicit CombineNullOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CombineNullOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::CombineNullOpAdaptor adaptor(operands);
      auto combineNullOp = cast<mlir::db::CombineNullOp>(op);
      auto packOp = rewriter.create<mlir::util::PackOp>(rewriter.getUnknownLoc(), typeConverter->convertType(combineNullOp.getType()), ValueRange({adaptor.null(), adaptor.val()}));
      rewriter.replaceOp(op, packOp.tuple());
      return success();
   }
};
class NullOpLowering : public ConversionPattern {
   public:
   explicit NullOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::NullOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto nullOp = cast<mlir::db::NullOp>(op);
      auto tupleType = typeConverter->convertType(nullOp.getType());
      auto undefTuple = rewriter.create<mlir::util::UndefTupleOp>(rewriter.getUnknownLoc(), tupleType);
      auto trueValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

      rewriter.replaceOpWithNewOp<mlir::util::SetTupleOp>(op, tupleType, undefTuple, trueValue, 0);
      return success();
   }
};
class ConstantLowering : public ConversionPattern {
   public:
   explicit ConstantLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ConstantOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto constantOp = cast<mlir::db::ConstantOp>(op);
      auto type = constantOp.getType();
      auto stdType = typeConverter->convertType(type);
      if (constantOp.getType().isa<mlir::db::UIntType>()) {
         if (!constantOp.value().isa<IntegerAttr>()) {
            return failure();
         }
         auto integerVal = constantOp.value().dyn_cast_or_null<IntegerAttr>().getInt();
         rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
         return success();
      } else if (constantOp.getType().isa<mlir::db::IntType>() || constantOp.getType().isa<mlir::db::BoolType>()) {
         if (!constantOp.value().isa<IntegerAttr>()) {
            return failure();
         }
         auto integerVal = constantOp.value().dyn_cast_or_null<IntegerAttr>().getInt();
         rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
         return success();
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         if (auto strAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            auto [low, high] = support::parseDecimal(strAttr.getValue().str(), decimalType.getS());
            std::vector<uint64_t> parts = {low, high};
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, APInt(128, parts)));
            return success();
         }
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         if (auto strAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            if (dateType.getUnit() == db::DateUnitAttr::day) {
               int32_t integerVal = support::parseDate32(strAttr.getValue().str());
               rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
            } else {
               int64_t integerVal = support::parseDate32(strAttr.getValue().str());
               integerVal *= 24 * 60 * 60 * 1000;
               rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
            }
            return success();
         }
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         if (auto strAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            int64_t integerVal = support::parseTimestamp(strAttr.getValue().str(), toTimeUnit(timestampType.getUnit()));
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
            return success();
         }
      } else if (type.isa<mlir::db::IntervalType>()) {
         if (auto intAttr = constantOp.value().dyn_cast_or_null<IntegerAttr>()) {
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, intAttr.getInt()));
            return success();
         }
      } else if (type.isa<mlir::db::FloatType>()) {
         if (auto floatAttr = constantOp.value().dyn_cast_or_null<FloatAttr>()) {
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getFloatAttr(stdType, floatAttr.getValueAsDouble()));
            return success();
         }
      } else if (type.isa<mlir::db::StringType>()) {
         if (auto stringAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            const std::string& str = stringAttr.getValue().str();
            Value result;
            ModuleOp parentModule = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();
            auto loc = rewriter.getUnknownLoc();
            auto* context = rewriter.getContext();
            auto i8Type = IntegerType::get(context, 8);
            auto insertionPoint = rewriter.saveInsertionPoint();
            int64_t strLen = str.size();
            std::vector<uint8_t> vec;
            for (auto c : str) {
               vec.push_back(c);
            }
            auto strStaticType = MemRefType::get({strLen}, i8Type);
            auto strDynamicType = MemRefType::get({-1}, IntegerType::get(context, 8));
            rewriter.setInsertionPointToStart(parentModule.getBody());
            auto initialValue = DenseIntElementsAttr::get(
               RankedTensorType::get({strLen}, i8Type), vec);
            static int id = 0;
            auto globalop = rewriter.create<mlir::memref::GlobalOp>(loc, "db_constant_string" + std::to_string(id++), rewriter.getStringAttr("private"), strStaticType, initialValue, true);
            rewriter.restoreInsertionPoint(insertionPoint);
            Value conststr = rewriter.create<mlir::memref::GetGlobalOp>(loc, strStaticType, globalop.sym_name());
            result = rewriter.create<memref::CastOp>(loc, conststr, strDynamicType);
            rewriter.replaceOp(op, result);
            return success();
         }
      }

      return failure();
   }
};
class CmpOpLowering : public ConversionPattern {
   public:
   explicit CmpOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CmpOp::getOperationName(), 1, context) {}
   CmpIPredicate translateIPredicate(db::DBCmpPredicate pred) const {
      switch (pred) {
         case db::DBCmpPredicate::eq:
            return CmpIPredicate::eq;
         case db::DBCmpPredicate::neq:
            return CmpIPredicate::ne;
         case db::DBCmpPredicate::lt:
            return CmpIPredicate::slt;
         case db::DBCmpPredicate::gt:
            return CmpIPredicate::sgt;
         case db::DBCmpPredicate::lte:
            return CmpIPredicate::sle;
         case db::DBCmpPredicate::gte:
            return CmpIPredicate::sge;
         case db::DBCmpPredicate::like:
            assert(false && "can not evaluate like on integers");
            return CmpIPredicate::ne;
      }
      assert(false && "unexpected case");
      return CmpIPredicate::eq;
   }
   CmpFPredicate translateFPredicate(db::DBCmpPredicate pred) const {
      switch (pred) {
         case db::DBCmpPredicate::eq:
            return CmpFPredicate::OEQ;
         case db::DBCmpPredicate::neq:
            return CmpFPredicate::ONE;
         case db::DBCmpPredicate::lt:
            return CmpFPredicate::OLT;
         case db::DBCmpPredicate::gt:
            return CmpFPredicate::OGT;
         case db::DBCmpPredicate::lte:
            return CmpFPredicate::OLE;
         case db::DBCmpPredicate::gte:
            return CmpFPredicate::OGE;
         case db::DBCmpPredicate::like:
            assert(false && "can not evaluate like on integers");
            return CmpFPredicate::OEQ;
      }
      assert(false && "unexpected case");
      return CmpFPredicate::OEQ;
   }
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto loc = rewriter.getUnknownLoc();
      auto cmpOp = cast<db::CmpOp>(op);
      auto type = cmpOp.left().getType().cast<db::DBType>().getBaseType();
      if (type.isa<db::StringType>()) {
         return failure();
      }
      db::NullHandler nullHandler(*typeConverter, rewriter);
      Value left = nullHandler.getValue(cmpOp.left());
      Value right = nullHandler.getValue(cmpOp.right());
      if (type.isa<db::BoolType>() || type.isa<db::IntType>() || type.isa<db::DecimalType>() || type.isa<db::DateType>() || type.isa<db::TimestampType>() || type.isa<db::IntervalType>()) {
         Value res = rewriter.create<CmpIOp>(loc, translateIPredicate(cmpOp.predicate()), left, right);
         rewriter.replaceOp(op, nullHandler.combineResult(res));
         return success();
      } else if (type.isa<db::FloatType>()) {
         Value res = rewriter.create<CmpFOp>(loc, translateFPredicate(cmpOp.predicate()), left, right);
         rewriter.replaceOp(op, nullHandler.combineResult(res));
         return success();
      }
      return failure();
   }
};

class CastOpLowering : public ConversionPattern {
   public:
   explicit CastOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CastOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto castOp = cast<mlir::db::CastOp>(op);
      auto loc = rewriter.getUnknownLoc();
      auto sourceType = castOp.val().getType().cast<db::DBType>();
      auto targetType = castOp.getType().cast<db::DBType>();
      auto scalarSourceType = sourceType.getBaseType();
      auto scalarTargetType = targetType.getBaseType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (scalarSourceType.isa<mlir::db::StringType>() || scalarTargetType.isa<mlir::db::StringType>()) return failure();
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
      } else if (auto intType = scalarSourceType.dyn_cast_or_null<db::IntType>()) {
         if (scalarTargetType.isa<db::FloatType>()) {
            value = rewriter.create<mlir::SIToFPOp>(loc, value, convertedTargetType);
         } else if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto sourceScale = decimalTargetType.getS();
            auto [low, high] = support::getDecimalScaleMultiplier(sourceScale);
            std::vector<uint64_t> parts = {low, high};
            auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), convertedTargetType, rewriter.getIntegerAttr(convertedTargetType, APInt(128, parts)));
            if (intType.getWidth() < 128) {
               value = rewriter.create<SignExtendIOp>(loc, value, convertedTargetType);
            }
            value = rewriter.create<mlir::MulIOp>(rewriter.getUnknownLoc(), convertedTargetType, value, multiplier);
         } else {
            return failure();
         }
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<db::FloatType>()) {
         if (scalarTargetType.isa<db::IntType>()) {
            value = rewriter.create<mlir::FPToSIOp>(loc, value, convertedTargetType);
         } else if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), convertedSourceType, FloatAttr::get(convertedSourceType, powf(10, decimalTargetType.getS())));
            value = rewriter.create<mlir::MulFOp>(rewriter.getUnknownLoc(), convertedSourceType, value, multiplier);
            value = rewriter.create<mlir::FPToSIOp>(loc, value, convertedTargetType);
         } else {
            return failure();
         }
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
         if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto sourceScale = decimalSourceType.getS();
            auto targetScale = decimalTargetType.getS();
            auto [low, high] = support::getDecimalScaleMultiplier(std::max(sourceScale, targetScale) - std::min(sourceScale, targetScale));
            std::vector<uint64_t> parts = {low, high};
            auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), convertedTargetType, rewriter.getIntegerAttr(convertedTargetType, APInt(128, parts)));
            if (sourceScale < targetScale) {
               value = rewriter.create<mlir::MulIOp>(rewriter.getUnknownLoc(), convertedTargetType, value, multiplier);
            } else {
               value = rewriter.create<mlir::SignedDivIOp>(rewriter.getUnknownLoc(), convertedTargetType, value, multiplier);
            }
         } else if (scalarTargetType.isa<db::FloatType>()) {
            auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), convertedTargetType, FloatAttr::get(convertedTargetType, powf(10, decimalSourceType.getS())));
            value = rewriter.create<mlir::SIToFPOp>(loc, value, convertedTargetType);
            value = rewriter.create<mlir::DivFOp>(rewriter.getUnknownLoc(), convertedTargetType, value, multiplier);
         } else if (auto intType = scalarTargetType.dyn_cast_or_null<db::IntType>()) {
            auto sourceScale = decimalSourceType.getS();
            auto [low, high] = support::getDecimalScaleMultiplier(sourceScale);
            std::vector<uint64_t> parts = {low, high};
            auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), convertedSourceType, rewriter.getIntegerAttr(convertedSourceType, APInt(128, parts)));
            value = rewriter.create<mlir::SignedDivIOp>(rewriter.getUnknownLoc(), convertedSourceType, value, multiplier);
            if (intType.getWidth() < 128) {
               value = rewriter.create<TruncateIOp>(loc, value, convertedTargetType);
            }
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
class CreateFlagLowering : public ConversionPattern {
   public:
   explicit CreateFlagLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateFlag::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto boolType= mlir::db::BoolType::get(rewriter.getContext());
      Type memrefType = util::GenericMemrefType::get(rewriter.getContext(), boolType, llvm::Optional<int64_t>());

      Value alloca = rewriter.create<mlir::util::AllocaOp>(op->getLoc(), memrefType, Value());
      Value falseVal = rewriter.create<mlir::db::ConstantOp>(op->getLoc(),boolType, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.create<util::StoreOp>(op->getLoc(), falseVal, alloca, Value());
      rewriter.replaceOp(op,alloca);
      return success();
   }
};
class SetFlagLowering : public ConversionPattern {
   public:
   explicit SetFlagLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::SetFlag::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::SetFlagAdaptor adaptor(operands);
         auto boolType= mlir::db::BoolType::get(rewriter.getContext());

      Value trueVal = rewriter.create<mlir::db::ConstantOp>(op->getLoc(),boolType, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      rewriter.create<util::StoreOp>(op->getLoc(), trueVal, adaptor.flag(), Value());
      rewriter.eraseOp(op);
      return success();
   }
};
class GetFlagLowering : public ConversionPattern {
   public:
   explicit GetFlagLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::GetFlag::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::GetFlagAdaptor adaptor(operands);
      auto boolType= mlir::db::BoolType::get(rewriter.getContext());

      Value flagValue=rewriter.create<util::LoadOp>(op->getLoc(), boolType, adaptor.flag(), Value());
      rewriter.replaceOp(op,flagValue);
      return success();
   }
};
} // namespace
void mlir::db::populateScalarToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](mlir::db::DBType type) {
      Type rawType = ::llvm::TypeSwitch<::mlir::db::DBType, mlir::Type>(type)
                        .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
                           return mlir::IntegerType::get(patterns.getContext(), 1);
                        })
                        .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
                           if (t.getUnit() == mlir::db::DateUnitAttr::day) {
                              return mlir::IntegerType::get(patterns.getContext(), 32);
                           } else {
                              return mlir::IntegerType::get(patterns.getContext(), 64);
                           }
                        })
                        .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
                           if (t.getUnit() == mlir::db::TimeUnitAttr::second && t.getUnit() == mlir::db::TimeUnitAttr::millisecond) {
                              return mlir::IntegerType::get(patterns.getContext(), 32);
                           } else {
                              return mlir::IntegerType::get(patterns.getContext(), 64);
                           }
                        })
                        .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                           return mlir::IntegerType::get(patterns.getContext(), 128);
                        })
                        .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
                           return mlir::IntegerType::get(patterns.getContext(), t.getWidth());
                        })
                        .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
                           return mlir::IntegerType::get(patterns.getContext(), t.getWidth());
                        })
                        .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
                           mlir::Type res;
                           if (t.getWidth() == 32) {
                              res = mlir::FloatType::getF32(patterns.getContext());
                           } else if (t.getWidth() == 64) {
                              res = mlir::FloatType::getF64(patterns.getContext());
                           }
                           return res;
                        })
                        .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
                           return mlir::MemRefType::get({-1}, IntegerType::get(patterns.getContext(), 8));
                        })
                        .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
                           return mlir::IntegerType::get(patterns.getContext(), 64);
                        })
                        .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
                           return mlir::IntegerType::get(patterns.getContext(), 64);
                        })
                        .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
                           if (t.getUnit() == mlir::db::IntervalUnitAttr::daytime) {
                              return mlir::IntegerType::get(patterns.getContext(), 64);
                           } else {
                              return mlir::IntegerType::get(patterns.getContext(), 32);
                           }
                        })
                        .Default([](::mlir::Type) { return Type(); });
      if (type.isNullable()) {
         return (Type) TupleType::get(patterns.getContext(), {IntegerType::get(patterns.getContext(), 1), rawType});
      } else {
         return rawType;
      }
   });
   typeConverter.addConversion([&](mlir::db::FlagType type) {
     auto boolType= mlir::db::BoolType::get(patterns.getContext());
     Type memrefType = util::GenericMemrefType::get(patterns.getContext(), boolType, llvm::Optional<int64_t>());
      return memrefType;
   });

     patterns.insert<CmpOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<NotOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<AndOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<OrOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::IntType, mlir::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::IntType, mlir::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::IntType, mlir::MulIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::IntType, mlir::SignedDivIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::IntType, mlir::SignedRemIOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::UIntType, mlir::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::UIntType, mlir::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::UIntType, mlir::MulIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::UIntType, mlir::UnsignedDivIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::UIntType, mlir::UnsignedRemIOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::FloatType, mlir::AddFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::FloatType, mlir::SubFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::FloatType, mlir::MulFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::FloatType, mlir::MulFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::FloatType, mlir::RemFOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::DecimalType, mlir::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::DecimalType, mlir::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::DecimalType, mlir::MulIOp>>(typeConverter, patterns.getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::DivOp, mlir::SignedDivIOp>>(typeConverter, patterns.getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::ModOp, mlir::SignedRemIOp>>(typeConverter, patterns.getContext());

   patterns.insert<NullOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<IsNullOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<CombineNullOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<ConstantLowering>(typeConverter, patterns.getContext());
   patterns.insert<CastOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<CreateFlagLowering>(typeConverter, patterns.getContext());
   patterns.insert<SetFlagLowering>(typeConverter, patterns.getContext());
   patterns.insert<GetFlagLowering>(typeConverter, patterns.getContext());

}