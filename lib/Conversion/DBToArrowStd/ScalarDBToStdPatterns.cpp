#include "mlir-support/parsing.h"
#include "mlir/Conversion/DBToArrowStd/ArrowTypes.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/SCF/SCF.h>

using namespace mlir;
namespace {

class NotOpLowering : public ConversionPattern {
   public:
   explicit NotOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::NotOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      Value trueValue = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, operands[0], trueValue);
      return success();
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
      auto loc = op->getLoc();

      for (size_t i = 0; i < operands.size(); i++) {
         auto currType = andOp.vals()[i].getType();
         bool currNullable = currType.isa<mlir::db::NullableType>();
         Value currNull;
         Value currVal;
         if (currNullable) {
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, operands[i]);
            currNull = unPackOp.vals()[0];
            currVal = unPackOp.vals()[1];
         } else {
            currVal = operands[i];
         }
         if (i == 0) {
            if (currNullable) {
               result = rewriter.create<arith::OrIOp>(loc, currNull, currVal);
            } else {
               result = currVal;
            }
            isNull = currNull;
         } else {
            if (currNullable) {
               if (isNull) {
                  isNull = rewriter.create<arith::OrIOp>(loc, isNull, currNull);
               } else {
                  isNull = currNull;
               }
            }
            if (currNullable) {
               result = rewriter.create<arith::SelectOp>(loc, currNull, result, rewriter.create<arith::AndIOp>(loc, currVal, result));
            } else {
               result = rewriter.create<arith::AndIOp>(loc, currVal, result);
            }
         }
      }
      if (andOp.getResult().getType().isa<mlir::db::NullableType>()) {
         isNull = rewriter.create<arith::AndIOp>(loc, result, isNull);
         Value combined = rewriter.create<mlir::util::PackOp>(loc, ValueRange({isNull, result}));
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
      auto loc = op->getLoc();
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      for (size_t i = 0; i < operands.size(); i++) {
         auto currType = orOp.vals()[i].getType();
         bool currNullable = currType.isa<mlir::db::NullableType>();
         Value currNull;
         Value currVal;
         if (currNullable) {
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, operands[i]);
            currNull = unPackOp.vals()[0];
            currVal = unPackOp.vals()[1];
         } else {
            currVal = operands[i];
         }
         if (i == 0) {
            if (currNullable) {
               result = rewriter.create<arith::SelectOp>(loc, currNull, falseValue, currVal);
            } else {
               result = currVal;
            }
            isNull = currNull;
         } else {
            if (currNullable) {
               if (isNull) {
                  isNull = rewriter.create<arith::OrIOp>(loc, isNull, currNull);
               } else {
                  isNull = currNull;
               }
            }
            if (currNullable) {
               result = rewriter.create<arith::SelectOp>(loc, currNull, result, rewriter.create<arith::OrIOp>(loc, currVal, result));
            } else {
               result = rewriter.create<arith::OrIOp>(loc, currVal, result);
            }
         }
      }
      if (orOp.getResult().getType().isa<mlir::db::NullableType>()) {
         isNull = rewriter.create<arith::SelectOp>(loc, result, falseValue, isNull);
         Value combined = rewriter.create<mlir::util::PackOp>(loc, ValueRange({isNull, result}));
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
      using AT = typename OpClass::Adaptor;
      auto adaptor = AT(operands);
      auto type = getBaseType(addOp.left().getType());
      if (type.template isa<OperandType>()) {
         rewriter.template replaceOpWithNewOp<StdOpClass>(op, typeConverter->convertType(addOp.result().getType()), adaptor.left(), adaptor.right());
         return success();
      }
      return failure();
   }
};
mlir::Value getDecimalScaleMultiplierConstant(mlir::OpBuilder& builder, unsigned s, mlir::Type stdType, mlir::Location loc) {
   auto [low, high] = support::getDecimalScaleMultiplier(s);
   std::vector<uint64_t> parts = {low, high};
   auto multiplier = builder.create<arith::ConstantOp>(loc, stdType, builder.getIntegerAttr(stdType, APInt(stdType.template cast<mlir::IntegerType>().getWidth(), parts)));
   return multiplier;
}
template <class DBOp, class Op>
class DecimalOpScaledLowering : public ConversionPattern {
   public:
   explicit DecimalOpScaledLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, DBOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto addOp = cast<DBOp>(op);
      typename DBOp::Adaptor adaptor(operands);
      auto type = getBaseType(addOp.getType());
      if (auto decimalType = type.template dyn_cast_or_null<mlir::db::DecimalType>()) {
         auto stdType = typeConverter->convertType(decimalType);
         auto scaled = rewriter.create<arith::MulIOp>(op->getLoc(), stdType, adaptor.left(), getDecimalScaleMultiplierConstant(rewriter, decimalType.getS(), stdType, op->getLoc()));
         rewriter.template replaceOpWithNewOp<Op>(op, stdType, scaled, adaptor.right());
         return success();
      }
      return failure();
   }
};
class DecimalMulLowering : public ConversionPattern {
   public:
   explicit DecimalMulLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::MulOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto mulOp = cast<mlir::db::MulOp>(op);
      typename mlir::db::MulOpAdaptor adaptor(operands);
      if (auto decimalType = mulOp.getType().template dyn_cast_or_null<mlir::db::DecimalType>()) {
         auto stdType = typeConverter->convertType(decimalType);
         auto multiplied = rewriter.create<mlir::arith::MulIOp>(op->getLoc(), stdType, adaptor.left(), adaptor.right());
         rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, stdType, multiplied, getDecimalScaleMultiplierConstant(rewriter, decimalType.getS(), stdType, op->getLoc()));
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
      auto isNullOp = mlir::dyn_cast_or_null<mlir::db::IsNullOp>(op);
      mlir::db::IsNullOpAdaptor isNullOpAdaptor(operands);

      if (isNullOp.val().getType().isa<mlir::db::NullableType>()) {
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(op->getLoc(), isNullOpAdaptor.val());
         rewriter.replaceOp(op, unPackOp.vals()[0]);
      } else {
         rewriter.replaceOp(op, isNullOpAdaptor.val());
      }
      return success();
   }
};
class NullableGetValOpLowering : public ConversionPattern {
   public:
   explicit NullableGetValOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::NullableGetVal::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::NullableGetValAdaptor nullableGetValAdaptor(operands);
      auto unPackOp = rewriter.create<mlir::util::UnPackOp>(op->getLoc(), nullableGetValAdaptor.val());
      rewriter.replaceOp(op, unPackOp.vals()[1]);
      return success();
   }
};
class AsNullableOpLowering : public ConversionPattern {
   public:
   explicit AsNullableOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::AsNullableOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::AsNullableOpAdaptor adaptor(operands);
      mlir::Value isNull = adaptor.null();
      if (!isNull) {
         isNull = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      }
      auto packOp = rewriter.create<mlir::util::PackOp>(op->getLoc(), ValueRange({isNull, adaptor.val()}));
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
      auto undefTuple = rewriter.create<mlir::util::UndefTupleOp>(op->getLoc(), tupleType);
      auto trueValue = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

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
      auto [arrowType, param1, param2] = mlir::db::codegen::convertTypeToArrow(type);
      std::variant<int64_t, double, std::string> parseArg;
      if (auto integerAttr = constantOp.value().dyn_cast_or_null<IntegerAttr>()) {
         parseArg = integerAttr.getInt();
      } else if (auto floatAttr = constantOp.value().dyn_cast_or_null<FloatAttr>()) {
         parseArg = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
         parseArg = stringAttr.str();
      } else {
         return failure();
      }
      auto parseResult = support::parse(parseArg, arrowType, param1, param2);
      if (auto intType = stdType.dyn_cast_or_null<IntegerType>()) {
         if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
            auto [low, high] = support::parseDecimal(std::get<std::string>(parseResult), decimalType.getS());
            std::vector<uint64_t> parts = {low, high};
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, APInt(stdType.cast<mlir::IntegerType>().getWidth(), parts)));
            return success();
         } else {
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, std::get<int64_t>(parseResult)));
            return success();
         }
      } else if (auto floatType = stdType.dyn_cast_or_null<FloatType>()) {
         rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, stdType, rewriter.getFloatAttr(stdType, std::get<double>(parseResult)));
         return success();
      } else if (type.isa<mlir::db::StringType>()) {
         std::string str = std::get<std::string>(parseResult);

         rewriter.replaceOpWithNewOp<mlir::util::CreateConstVarLen>(op, mlir::util::VarLen32Type::get(rewriter.getContext()), rewriter.getStringAttr(str));
         return success();
      } else {
         return failure();
      }
      return failure();
   }
};
class CmpOpLowering : public ConversionPattern {
   public:
   explicit CmpOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CmpOp::getOperationName(), 1, context) {}
   arith::CmpIPredicate translateIPredicate(db::DBCmpPredicate pred) const {
      switch (pred) {
         case db::DBCmpPredicate::eq:
            return arith::CmpIPredicate::eq;
         case db::DBCmpPredicate::neq:
            return arith::CmpIPredicate::ne;
         case db::DBCmpPredicate::lt:
            return arith::CmpIPredicate::slt;
         case db::DBCmpPredicate::gt:
            return arith::CmpIPredicate::sgt;
         case db::DBCmpPredicate::lte:
            return arith::CmpIPredicate::sle;
         case db::DBCmpPredicate::gte:
            return arith::CmpIPredicate::sge;
         case db::DBCmpPredicate::like:
            assert(false && "can not evaluate like on integers");
            return arith::CmpIPredicate::ne;
      }
      assert(false && "unexpected case");
      return arith::CmpIPredicate::eq;
   }
   arith::CmpFPredicate translateFPredicate(db::DBCmpPredicate pred) const {
      switch (pred) {
         case db::DBCmpPredicate::eq:
            return arith::CmpFPredicate::OEQ;
         case db::DBCmpPredicate::neq:
            return arith::CmpFPredicate::ONE;
         case db::DBCmpPredicate::lt:
            return arith::CmpFPredicate::OLT;
         case db::DBCmpPredicate::gt:
            return arith::CmpFPredicate::OGT;
         case db::DBCmpPredicate::lte:
            return arith::CmpFPredicate::OLE;
         case db::DBCmpPredicate::gte:
            return arith::CmpFPredicate::OGE;
         case db::DBCmpPredicate::like:
            assert(false && "can not evaluate like on integers");
            return arith::CmpFPredicate::OEQ;
      }
      assert(false && "unexpected case");
      return arith::CmpFPredicate::OEQ;
   }
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto cmpOp = cast<db::CmpOp>(op);
      mlir::db::CmpOpAdaptor adaptor(operands);
      if (!adaptor.left().getType().isIntOrIndexOrFloat()) {
         return failure();
      }
      if (adaptor.left().getType().isIntOrIndex()) {
         rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, translateIPredicate(cmpOp.predicate()), adaptor.left(), adaptor.right());
      } else {
         rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, translateFPredicate(cmpOp.predicate()), adaptor.left(), adaptor.right());
      }
      return success();
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
      auto loc = op->getLoc();
      auto scalarSourceType = castOp.val().getType();
      auto scalarTargetType = castOp.getType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (scalarSourceType.isa<mlir::db::StringType>() || scalarTargetType.isa<mlir::db::StringType>()) return failure();
      Value value = operands[0];
      if (scalarSourceType == scalarTargetType) {
         rewriter.replaceOp(op, value);
         return success();
      }
      if (auto sourceIntWidth = getIntegerWidth(scalarSourceType, false)) {
         if (scalarTargetType.isa<FloatType>()) {
            value = rewriter.create<arith::SIToFPOp>(loc, convertedTargetType, value);
            rewriter.replaceOp(op, value);
            return success();
         } else if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            int decimalWidth = typeConverter->convertType(decimalTargetType).cast<mlir::IntegerType>().getWidth();
            if (sourceIntWidth < decimalWidth) {
               value = rewriter.create<arith::ExtSIOp>(loc, convertedTargetType, value);
            }
            rewriter.replaceOpWithNewOp<arith::MulIOp>(op, convertedTargetType, value, getDecimalScaleMultiplierConstant(rewriter, decimalTargetType.getS(), convertedTargetType, op->getLoc()));
            return success();
         } else if (auto targetIntWidth = getIntegerWidth(scalarTargetType, false)) {
            if (targetIntWidth < sourceIntWidth) {
               rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, convertedTargetType, value);
            } else {
               rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, convertedTargetType, value);
            }
            return success();
         }
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<FloatType>()) {
         if (getIntegerWidth(scalarTargetType, false)) {
            value = rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, convertedTargetType, value);
            return success();
         } else if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedSourceType, FloatAttr::get(convertedSourceType, powf(10, decimalTargetType.getS())));
            value = rewriter.create<arith::MulFOp>(loc, convertedSourceType, value, multiplier);
            rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, convertedTargetType, value);
            return success();
         }
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
         if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto sourceScale = decimalSourceType.getS();
            auto targetScale = decimalTargetType.getS();
            size_t decimalWidth = convertedSourceType.cast<mlir::IntegerType>().getWidth(); //TODO
            auto [low, high] = support::getDecimalScaleMultiplier(std::max(sourceScale, targetScale) - std::min(sourceScale, targetScale));
            std::vector<uint64_t> parts = {low, high};
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedTargetType, rewriter.getIntegerAttr(convertedTargetType, APInt(decimalWidth, parts)));
            if (sourceScale < targetScale) {
               rewriter.replaceOpWithNewOp<arith::MulIOp>(op, convertedTargetType, value, multiplier);
            } else {
               rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, convertedTargetType, value, multiplier);
            }
            return success();
         } else if (scalarTargetType.isa<FloatType>()) {
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedTargetType, FloatAttr::get(convertedTargetType, powf(10, decimalSourceType.getS())));
            value = rewriter.create<arith::SIToFPOp>(loc, convertedTargetType, value);
            rewriter.replaceOpWithNewOp<arith::DivFOp>(op, convertedTargetType, value, multiplier);
            return success();
         } else if (auto targetIntWidth = getIntegerWidth(scalarTargetType, false)) {
            int decimalWidth = convertedSourceType.cast<mlir::IntegerType>().getWidth();
            auto multiplier = getDecimalScaleMultiplierConstant(rewriter, decimalSourceType.getS(), convertedSourceType, op->getLoc());
            value = rewriter.create<arith::DivSIOp>(loc, convertedSourceType, value, multiplier);
            if (targetIntWidth < decimalWidth) {
               value = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, value);
            }
            rewriter.replaceOp(op, value);
            return success();
         }
      }
      return failure();
   }
};
class BetweenLowering : public ConversionPattern {
   public:
   explicit BetweenLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BetweenOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto betweenOp = mlir::cast<mlir::db::BetweenOp>(op);
      auto isGteLower = rewriter.create<mlir::db::CmpOp>(op->getLoc(), betweenOp.lowerInclusive() ? mlir::db::DBCmpPredicate::gte : mlir::db::DBCmpPredicate::gt, betweenOp.val(), betweenOp.lower());
      auto isLteUpper = rewriter.create<mlir::db::CmpOp>(op->getLoc(), betweenOp.upperInclusive() ? mlir::db::DBCmpPredicate::lte : mlir::db::DBCmpPredicate::lt, betweenOp.val(), betweenOp.upper());
      auto isInRange = rewriter.create<mlir::db::AndOp>(op->getLoc(), ValueRange({isGteLower, isLteUpper}));
      rewriter.replaceOp(op, isInRange.res());
      return success();
   }
};
class OneOfLowering : public ConversionPattern {
   public:
   explicit OneOfLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::OneOfOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto oneOfOp = mlir::cast<mlir::db::OneOfOp>(op);
      std::vector<Value> compared;
      for (auto ele : oneOfOp.vals()) {
         compared.push_back(rewriter.create<mlir::db::CmpOp>(op->getLoc(), mlir::db::DBCmpPredicate::eq, oneOfOp.val(), ele));
      }
      auto isInRange = rewriter.create<mlir::db::OrOp>(op->getLoc(), compared);
      rewriter.replaceOp(op, isInRange.res());
      return success();
   }
};
class HashLowering : public ConversionPattern {
   Value combineHashes(OpBuilder& builder, Location loc, Value hash1, Value totalHash) const {
      if (!totalHash) {
         return hash1;
      } else {
         return builder.create<mlir::util::HashCombine>(loc, builder.getIndexType(), hash1, totalHash);
      }
   }
   Value hashInteger(OpBuilder& builder, Location loc, Value integer) const {
      Value asIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), integer);
      return builder.create<mlir::util::Hash64>(loc, builder.getIndexType(), asIndex);
   }
   Value hashImpl(OpBuilder& builder, Location loc, Value v, Value totalHash, Type originalType) const {
      if (auto intType = v.getType().dyn_cast_or_null<mlir::IntegerType>()) {
         if (intType.getWidth() == 128) {
            auto i64Type = IntegerType::get(builder.getContext(), 64);
            auto i128Type = IntegerType::get(builder.getContext(), 128);

            Value low = builder.create<arith::TruncIOp>(loc, i64Type, v);
            Value shift = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(i128Type, 64));
            Value high = builder.create<arith::ShRUIOp>(loc, i128Type, v, shift);
            Value first = hashInteger(builder, loc, high);
            Value second = hashInteger(builder, loc, low);
            Value combined1 = combineHashes(builder, loc, first, totalHash);
            Value combined2 = combineHashes(builder, loc, second, combined1);
            return combined2;
         } else {
            return combineHashes(builder, loc, hashInteger(builder, loc, v), totalHash);
         }

      } else if (auto floatType = v.getType().dyn_cast_or_null<mlir::FloatType>()) {
         assert(false && "can not hash float values");
      } else if (auto varLenType = v.getType().dyn_cast_or_null<mlir::util::VarLen32Type>()) {
         auto hash = builder.create<mlir::util::HashVarLen>(loc, builder.getIndexType(), v);
         return combineHashes(builder, loc, hash, totalHash);
      } else if (auto tupleType = v.getType().dyn_cast_or_null<mlir::TupleType>()) {
         if (auto originalTupleType = originalType.dyn_cast_or_null<mlir::TupleType>()) {
            auto unpacked = builder.create<util::UnPackOp>(loc, v);
            size_t i = 0;
            for (auto v : unpacked->getResults()) {
               totalHash = hashImpl(builder, loc, v, totalHash, originalTupleType.getType(i++));
            }
            return totalHash;
         } else if (originalType.isa<mlir::db::NullableType>()) {
            auto unpacked = builder.create<util::UnPackOp>(loc, v);
            mlir::Value hashedIfNotNull = hashImpl(builder, loc, unpacked.getResult(1), totalHash, getBaseType(originalType));
            if (!totalHash) {
               totalHash = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
            }
            return builder.create<mlir::arith::SelectOp>(loc, unpacked.getResult(0), totalHash, hashedIfNotNull);
         }
         assert(false && "should not happen");
         return Value();
      }
      assert(false && "should not happen");
      return Value();
   }

   public:
   explicit HashLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::Hash::getOperationName(), 1, context) {}
   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::HashAdaptor hashAdaptor(operands);
      auto hashOp = mlir::cast<mlir::db::Hash>(op);

      rewriter.replaceOp(op, hashImpl(rewriter, op->getLoc(), hashAdaptor.val(), Value(), hashOp.val().getType()));
      return success();
   }
};

} // namespace
void mlir::db::populateScalarToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](::mlir::db::DateType t) {
      return mlir::IntegerType::get(patterns.getContext(), 64);
   });
   typeConverter.addConversion([&](::mlir::db::DecimalType t) {
      if (t.getP() < 19) {
         return mlir::IntegerType::get(patterns.getContext(), 64);
      }
      return mlir::IntegerType::get(patterns.getContext(), 128);
   });
   typeConverter.addConversion([&](::mlir::db::CharType t) {
      if (t.getBytes() > 8) return mlir::Type();
      return (Type) mlir::IntegerType::get(patterns.getContext(), t.getBytes() * 8);
   });
   typeConverter.addConversion([&](::mlir::db::StringType t) {
      return mlir::util::VarLen32Type::get(patterns.getContext());
   });
   typeConverter.addConversion([&](::mlir::db::TimestampType t) {
      return mlir::IntegerType::get(patterns.getContext(), 64);
   });
   typeConverter.addConversion([&](::mlir::db::IntervalType t) {
      if (t.getUnit() == mlir::db::IntervalUnitAttr::daytime) {
         return mlir::IntegerType::get(patterns.getContext(), 64);
      } else {
         return mlir::IntegerType::get(patterns.getContext(), 32);
      }
   });

   typeConverter.addConversion([&](mlir::db::NullableType type) {
      return (Type) TupleType::get(patterns.getContext(), {IntegerType::get(patterns.getContext(), 1), typeConverter.convertType(type.getType())});
   });


   patterns.insert<CmpOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<BetweenLowering>(typeConverter, patterns.getContext());
   patterns.insert<OneOfLowering>(typeConverter, patterns.getContext());

   patterns.insert<NotOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<AndOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<OrOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::IntegerType, arith::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::IntegerType, arith::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::IntegerType, arith::MulIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::IntegerType, arith::DivSIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::IntegerType, arith::RemSIOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::IntegerType, arith::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::IntegerType, arith::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::IntegerType, arith::MulIOp>>(typeConverter, patterns.getContext());
   //patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::UIntType, arith::DivUIOp>>(typeConverter, patterns.getContext());//todo: fix unsigned division
   //patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::UIntType, arith::RemUIOp>>(typeConverter, patterns.getContext());//todo: fix unsigned division

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::FloatType, arith::AddFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::FloatType, arith::SubFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::FloatType, arith::MulFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::FloatType, arith::DivFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::FloatType, arith::RemFOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::DecimalType, arith::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::DecimalType, arith::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::DivOp, arith::DivSIOp>>(typeConverter, patterns.getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::ModOp, arith::RemSIOp>>(typeConverter, patterns.getContext());
   patterns.insert<DecimalMulLowering>(typeConverter, patterns.getContext());

   patterns.insert<NullOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<IsNullOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<AsNullableOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<NullableGetValOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<ConstantLowering>(typeConverter, patterns.getContext());
   patterns.insert<CastOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<HashLowering>(typeConverter, patterns.getContext());
}