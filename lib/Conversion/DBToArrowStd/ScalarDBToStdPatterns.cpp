#include "mlir-support/parsing.h"
#include "mlir/Conversion/DBToArrowStd/ArrowTypes.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/DBToArrowStd/NullHandler.h"
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
      auto notOp = cast<mlir::db::NotOp>(op);
      Value trueValue = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      mlir::db::DBType valType = notOp.val().getType().cast<mlir::db::DBType>();
      if (valType.isNullable()) {
         auto tupleType = typeConverter->convertType(notOp.val().getType());
         Value val = rewriter.create<util::GetTupleOp>(op->getLoc(), rewriter.getI1Type(), operands[0], 1);
         val = rewriter.create<arith::XOrIOp>(op->getLoc(), val, trueValue);
         rewriter.replaceOpWithNewOp<util::SetTupleOp>(op, tupleType, operands[0], val, 1);
         return success();
      } else {
         rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, operands[0], trueValue);
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
      auto loc = op->getLoc();
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

      for (size_t i = 0; i < operands.size(); i++) {
         auto currType = andOp.vals()[i].getType();
         bool currNullable = currType.dyn_cast_or_null<mlir::db::DBType>().isNullable();
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
               result = rewriter.create<SelectOp>(loc, currNull, trueValue, currVal);
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
               result = rewriter.create<SelectOp>(loc, currNull, result, rewriter.create<SelectOp>(loc, currVal, result, falseValue));
            } else {
               result = rewriter.create<SelectOp>(loc, currVal, result, falseValue);
            }
         }
      }
      if (andOp.getResult().getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
         isNull = rewriter.create<SelectOp>(loc, result, isNull, falseValue);
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
      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

      for (size_t i = 0; i < operands.size(); i++) {
         auto currType = orOp.vals()[i].getType();
         bool currNullable = currType.dyn_cast_or_null<mlir::db::DBType>().isNullable();
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
               result = rewriter.create<SelectOp>(loc, currNull, falseValue, currVal);
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
               result = rewriter.create<SelectOp>(loc, currNull, result, rewriter.create<SelectOp>(loc, currVal, trueValue, result));
            } else {
               result = rewriter.create<SelectOp>(loc, currVal, trueValue, result);
            }
         }
      }
      if (orOp.getResult().getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
         isNull = rewriter.create<SelectOp>(loc, result, falseValue, isNull);
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
      db::NullHandler nullHandler(*typeConverter, rewriter, op->getLoc());
      auto type = addOp.left().getType();
      Type resType = addOp.result().getType().template cast<db::DBType>().getBaseType();
      Value left = nullHandler.getValue(addOp.left(), adaptor.left());
      Value right = nullHandler.getValue(addOp.right(), adaptor.right());
      if (type.template isa<OperandType>()) {
         Value replacement = rewriter.create<StdOpClass>(op->getLoc(), typeConverter->convertType(resType), left, right);
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
      typename DBOp::Adaptor adaptor(operands);
      db::NullHandler nullHandler(*typeConverter, rewriter, addOp->getLoc());
      Value left = nullHandler.getValue(addOp.left(), adaptor.left());
      Value right = nullHandler.getValue(addOp.right(), adaptor.right());
      if (left.getType() != right.getType()) {
         return failure();
      }
      auto type = addOp.getType();
      if (auto decimalType = type.template dyn_cast_or_null<mlir::db::DecimalType>()) {
         auto [low, high] = support::getDecimalScaleMultiplier(decimalType.getS());
         std::vector<uint64_t> parts = {low, high};
         auto stdType = typeConverter->convertType(decimalType.getBaseType());
         auto multiplier = rewriter.create<arith::ConstantOp>(op->getLoc(), stdType, rewriter.getIntegerAttr(stdType, APInt(stdType.template cast<mlir::IntegerType>().getWidth(), parts)));
         left = rewriter.create<arith::MulIOp>(op->getLoc(), stdType, left, multiplier);
         auto replacement = rewriter.create<Op>(op->getLoc(), stdType, left, right);
         rewriter.replaceOp(op, nullHandler.combineResult(replacement));

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
      mlir::db::IsNullOpAdaptor isNullOpAdaptor(operands);
      auto unPackOp = rewriter.create<mlir::util::UnPackOp>(op->getLoc(), isNullOpAdaptor.val());
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
      auto packOp = rewriter.create<mlir::util::PackOp>(op->getLoc(), ValueRange({adaptor.null(), adaptor.val()}));
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
class SubStrOpLowering : public ConversionPattern {
   public:
   explicit SubStrOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::SubStrOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto subStrOp = cast<mlir::db::SubStrOp>(op);
      mlir::db::SubStrOpAdaptor adaptor(operands);
      Value pos1AsIndex = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(subStrOp.from() - 1));
      Value lenAsIndex = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(subStrOp.to() - subStrOp.from() + 1));
      Value asMemRef = rewriter.create<util::ToMemrefOp>(op->getLoc(), MemRefType::get({-1}, rewriter.getIntegerType(8)), adaptor.val());
      Value view = rewriter.create<mlir::memref::ViewOp>(op->getLoc(), MemRefType::get({-1}, rewriter.getIntegerType(8)), asMemRef, pos1AsIndex, mlir::ValueRange({lenAsIndex}));
      Value val = rewriter.create<mlir::util::ToGenericMemrefOp>(op->getLoc(), mlir::util::RefType::get(rewriter.getContext(), IntegerType::get(rewriter.getContext(), 8), -1), view);

      rewriter.replaceOp(op, val);
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
      auto loc = op->getLoc();
      auto [arrowType,param1,param2] = mlir::db::codegen::convertTypeToArrow(type.cast<mlir::db::DBType>());
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
      auto parseResult = support::parse(parseArg, arrowType, param1,param2);
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
         rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, stdType, rewriter.getFloatAttr(stdType,  std::get<double>(parseResult)));
         return success();
      } else if (auto refType = stdType.dyn_cast_or_null<mlir::util::RefType>()) {
         Value result;
         ModuleOp parentModule = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();
         auto* context = rewriter.getContext();
         auto i8Type = IntegerType::get(context, 8);
         auto insertionPoint = rewriter.saveInsertionPoint();
         std::string str= std::get<std::string>(parseResult);
         int64_t strLen = str.size();
         std::vector<uint8_t> vec;
         for (auto c : str) {
            vec.push_back(static_cast<uint8_t>(c));
         }
         auto strStaticType = MemRefType::get({strLen}, i8Type);
         auto strDynamicType = MemRefType::get({-1}, IntegerType::get(context, 8));
         rewriter.setInsertionPointToStart(parentModule.getBody());
         Attribute initialValue = DenseIntElementsAttr::get(
            RankedTensorType::get({strLen}, i8Type), vec);
         static int id = 0;
         auto globalop = rewriter.create<mlir::memref::GlobalOp>(loc, "db_constant_string" + std::to_string(id++), rewriter.getStringAttr("private"), strStaticType, initialValue, true, rewriter.getI64IntegerAttr(1));
         rewriter.restoreInsertionPoint(insertionPoint);
         Value conststr = rewriter.create<mlir::memref::GetGlobalOp>(loc, strStaticType, globalop.sym_name());
         result = rewriter.create<memref::CastOp>(loc, conststr, strDynamicType);
         Value strres = rewriter.create<mlir::util::ToGenericMemrefOp>(loc, mlir::util::RefType::get(getContext(), rewriter.getIntegerType(8), -1), result);
         rewriter.replaceOp(op, strres);
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
      auto loc = op->getLoc();
      auto cmpOp = cast<db::CmpOp>(op);
      db::CmpOpAdaptor adaptor(operands);
      auto type = cmpOp.left().getType().cast<db::DBType>().getBaseType();
      if (type.isa<db::StringType>()) {
         return failure();
      }
      db::NullHandler nullHandler(*typeConverter, rewriter, loc);
      Value left = nullHandler.getValue(cmpOp.left(), adaptor.left());
      Value right = nullHandler.getValue(cmpOp.right(), adaptor.right());
      if (type.isa<db::BoolType>() || type.isa<db::IntType>() || type.isa<db::DecimalType>() || type.isa<db::DateType>() || type.isa<db::TimestampType>() || type.isa<db::IntervalType>() || type.isa<db::CharType>()) {
         Value res = rewriter.create<arith::CmpIOp>(loc, translateIPredicate(cmpOp.predicate()), left, right);
         rewriter.replaceOp(op, nullHandler.combineResult(res));
         return success();
      } else if (type.isa<db::FloatType>()) {
         Value res = rewriter.create<arith::CmpFOp>(loc, translateFPredicate(cmpOp.predicate()), left, right);
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
      auto loc = op->getLoc();
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
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, operands[0]);
         isNull = unPackOp.vals()[0];
         value = unPackOp.vals()[1];
      } else {
         isNull = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         value = operands[0];
      }
      if (scalarSourceType == scalarTargetType) {
         //nothing to do here
      } else if (auto intType = scalarSourceType.dyn_cast_or_null<db::IntType>()) {
         if (scalarTargetType.isa<db::FloatType>()) {
            value = rewriter.create<arith::SIToFPOp>(loc, value, convertedTargetType);
         } else if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto sourceScale = decimalTargetType.getS();
            size_t decimalWidth = typeConverter->convertType(decimalTargetType).cast<mlir::IntegerType>().getWidth();
            auto [low, high] = support::getDecimalScaleMultiplier(sourceScale);
            std::vector<uint64_t> parts = {low, high};
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedTargetType, rewriter.getIntegerAttr(convertedTargetType, APInt(decimalWidth, parts)));
            if (intType.getWidth() < decimalWidth) {
               value = rewriter.create<arith::ExtSIOp>(loc, value, convertedTargetType);
            }
            value = rewriter.create<arith::MulIOp>(loc, convertedTargetType, value, multiplier);
         } else if (auto targetType = scalarTargetType.dyn_cast_or_null<db::IntType>()) {
            if (targetType.getWidth() < intType.getWidth()) {
               value = rewriter.create<arith::TruncIOp>(loc, value, convertedTargetType);
            } else {
               value = rewriter.create<arith::ExtSIOp>(loc, value, convertedTargetType);
            }
         } else {
            return failure();
         }
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<db::FloatType>()) {
         if (scalarTargetType.isa<db::IntType>()) {
            value = rewriter.create<arith::FPToSIOp>(loc, value, convertedTargetType);
         } else if (auto decimalTargetType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedSourceType, FloatAttr::get(convertedSourceType, powf(10, decimalTargetType.getS())));
            value = rewriter.create<arith::MulFOp>(loc, convertedSourceType, value, multiplier);
            value = rewriter.create<arith::FPToSIOp>(loc, value, convertedTargetType);
         } else {
            return failure();
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
               value = rewriter.create<arith::MulIOp>(loc, convertedTargetType, value, multiplier);
            } else {
               value = rewriter.create<arith::DivSIOp>(loc, convertedTargetType, value, multiplier);
            }
         } else if (scalarTargetType.isa<db::FloatType>()) {
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedTargetType, FloatAttr::get(convertedTargetType, powf(10, decimalSourceType.getS())));
            value = rewriter.create<arith::SIToFPOp>(loc, value, convertedTargetType);
            value = rewriter.create<arith::DivFOp>(loc, convertedTargetType, value, multiplier);
         } else if (auto intType = scalarTargetType.dyn_cast_or_null<db::IntType>()) {
            auto sourceScale = decimalSourceType.getS();
            auto [low, high] = support::getDecimalScaleMultiplier(sourceScale);
            size_t decimalWidth = convertedSourceType.cast<mlir::IntegerType>().getWidth();

            std::vector<uint64_t> parts = {low, high};

            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedSourceType, rewriter.getIntegerAttr(convertedSourceType, APInt(decimalWidth, parts)));
            value = rewriter.create<arith::DivSIOp>(loc, convertedSourceType, value, multiplier);
            if (intType.getWidth() < decimalWidth) {
               value = rewriter.create<arith::TruncIOp>(loc, value, convertedTargetType);
            }
         } else {
            return failure();
         }
      } else {
         return failure();
      }
      //todo convert types
      if (targetType.isNullable()) {
         Value combined = rewriter.create<mlir::util::PackOp>(loc, ValueRange({isNull, value}));
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
      auto boolType = mlir::db::BoolType::get(rewriter.getContext());
      Type memrefType = util::RefType::get(rewriter.getContext(), boolType, llvm::Optional<int64_t>());
      Value alloca;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         auto func = op->getParentOfType<mlir::FuncOp>();
         rewriter.setInsertionPointToStart(&func.getBody().front());
         alloca = rewriter.create<mlir::util::AllocaOp>(op->getLoc(), memrefType, Value());
      }
      Value falseVal = rewriter.create<mlir::db::ConstantOp>(op->getLoc(), boolType, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.create<util::StoreOp>(op->getLoc(), falseVal, alloca, Value());
      rewriter.replaceOp(op, alloca);
      return success();
   }
};
class SetFlagLowering : public ConversionPattern {
   public:
   explicit SetFlagLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::SetFlag::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::SetFlagAdaptor adaptor(operands);

      rewriter.create<util::StoreOp>(op->getLoc(), adaptor.val(), adaptor.flag(), Value());
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
      auto boolType = mlir::db::BoolType::get(rewriter.getContext());

      Value flagValue = rewriter.create<util::LoadOp>(op->getLoc(), boolType, adaptor.flag(), Value());
      rewriter.replaceOp(op, flagValue);
      return success();
   }
};
class HashLowering : public ConversionPattern {
   Value combineHashes(OpBuilder& builder, Location loc, Value hash1, Value totalHash, bool& required) const {
      Value kMul = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0x9ddfea08eb382d69));
      Value k47 = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(47));

      if (!required) {
         required = true;
         Value shifted2 = builder.create<arith::ShRUIOp>(loc, hash1, k47);
         Value b = builder.create<arith::XOrIOp>(loc, shifted2, hash1);
         Value multiplied4 = builder.create<arith::MulIOp>(loc, b, kMul);
         return multiplied4;
      }

      Value xOred = builder.create<arith::XOrIOp>(loc, totalHash, hash1);
      Value multiplied2 = builder.create<arith::MulIOp>(loc, xOred, kMul);
      Value shifted = builder.create<arith::ShRUIOp>(loc, multiplied2, k47);
      Value a = builder.create<arith::XOrIOp>(loc, shifted, multiplied2);
      Value xOred2 = builder.create<arith::XOrIOp>(loc, a, hash1);
      Value multiplied3 = builder.create<arith::MulIOp>(loc, xOred2, kMul);
      Value shifted2 = builder.create<arith::ShRUIOp>(loc, multiplied3, k47);
      Value b = builder.create<arith::XOrIOp>(loc, shifted2, multiplied3);
      Value multiplied4 = builder.create<arith::MulIOp>(loc, b, kMul);
      return multiplied4;
   }
   Value hashInteger(OpBuilder& builder, Location loc, Value magicConstant, Value integer) const {
      Value asIndex = builder.create<arith::IndexCastOp>(loc, integer, builder.getIndexType());
      Value multiplied = builder.create<arith::MulIOp>(loc, asIndex, magicConstant);
      return multiplied;
   }
   Value hashImpl(OpBuilder& builder, Location loc, Value v, Value totalHash, Value magicConstant, Type originalType, bool& combinationRequired) const {
      if (auto intType = v.getType().dyn_cast_or_null<mlir::IntegerType>()) {
         if (intType.getWidth() == 128) {
            auto i64Type = IntegerType::get(builder.getContext(), 64);
            auto i128Type = IntegerType::get(builder.getContext(), 128);

            Value low = builder.create<arith::TruncIOp>(loc, v, i64Type);
            Value shift = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(i128Type, 64));
            Value high = builder.create<arith::ShRUIOp>(loc, i128Type, v, shift);
            Value first = hashInteger(builder, loc, magicConstant, high);
            Value second = hashInteger(builder, loc, magicConstant, low);
            Value combined1 = combineHashes(builder, loc, first, totalHash, combinationRequired);
            Value combined2 = combineHashes(builder, loc, second, combined1, combinationRequired);
            return combined2;
         } else {
            return combineHashes(builder, loc, hashInteger(builder, loc, magicConstant, v), totalHash, combinationRequired);
         }

      } else if (auto floatType = v.getType().dyn_cast_or_null<mlir::FloatType>()) {
         assert(false && "can not hash float values");
      } else if (auto memrefType = v.getType().dyn_cast_or_null<mlir::util::RefType>()) {
         Value len = builder.create<mlir::util::DimOp>(loc, builder.getIndexType(), v);

         Value const0 = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
         Value const1 = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1));
         Value const5 = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(5));
         Value const27 = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(27));

         auto loop2 = builder.create<scf::ForOp>(
            loc, const0, len, const1, len,
            [&](OpBuilder& b, Location loc, Value iv, ValueRange args) {
               Value hash = args.front();
               Value currVal = b.create<util::LoadOp>(loc, b.getIntegerType(8), v, iv);
               Value asIndex = builder.create<arith::IndexCastOp>(loc, currVal, builder.getIndexType());

               Value shifted5 = builder.create<arith::ShLIOp>(loc, hash, const5);
               Value shifted27 = builder.create<arith::ShRUIOp>(loc, hash, const27);
               Value xOred = builder.create<arith::XOrIOp>(loc, shifted5, shifted27);
               Value xOred2 = builder.create<arith::XOrIOp>(loc, xOred, asIndex);

               b.create<scf::YieldOp>(loc, xOred2);
            });
         Value hash = loop2.getResult(0);

         return combineHashes(builder, loc, hash, totalHash, combinationRequired);
      } else if (auto tupleType = v.getType().dyn_cast_or_null<mlir::TupleType>()) {
         if (auto originalTupleType = originalType.dyn_cast_or_null<mlir::TupleType>()) {
            auto unpacked = builder.create<util::UnPackOp>(loc, v);
            size_t i = 0;
            for (auto v : unpacked->getResults()) {
               totalHash = hashImpl(builder, loc, v, totalHash, magicConstant, originalTupleType.getType(i++), combinationRequired);
            }
            return totalHash;
         } else if (auto dbType = originalType.dyn_cast_or_null<mlir::db::DBType>()) {
            assert(dbType.isNullable());
            auto unpacked = builder.create<util::UnPackOp>(loc, v);
            mlir::Value hashedIfNotNull = hashImpl(builder, loc, unpacked.getResult(1), totalHash, magicConstant, dbType.getBaseType(), combinationRequired);
            return builder.create<mlir::SelectOp>(loc, unpacked.getResult(0), totalHash, hashedIfNotNull);
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
      bool combinationRequired = false;
      Value const0 = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
      Value magicConstant = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0xbf58476d1ce4e5b9));

      rewriter.replaceOp(op, hashImpl(rewriter, op->getLoc(), hashAdaptor.val(), const0, magicConstant, hashOp.val().getType(), combinationRequired));
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
                           if (t.getP() < 19) {
                              return mlir::IntegerType::get(patterns.getContext(), 64);
                           }
                           return mlir::IntegerType::get(patterns.getContext(), 128);
                        })
                        .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
                           return mlir::IntegerType::get(patterns.getContext(), t.getWidth());
                        })
                        .Case<::mlir::db::CharType>([&](::mlir::db::CharType t) {
                           if (t.getBytes() > 8) return mlir::Type();
                           return (Type) mlir::IntegerType::get(patterns.getContext(), t.getBytes() * 8);
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
                           return mlir::util::RefType::get(patterns.getContext(), IntegerType::get(patterns.getContext(), 8), -1);
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
      auto boolType = typeConverter.convertType(mlir::db::BoolType::get(patterns.getContext()));
      Type memrefType = util::RefType::get(patterns.getContext(), boolType, llvm::Optional<int64_t>());
      return memrefType;
   });

   patterns.insert<CmpOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<NotOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<AndOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<OrOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::IntType, arith::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::IntType, arith::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::IntType, arith::MulIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::IntType, arith::DivSIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::IntType, arith::RemSIOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::UIntType, arith::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::UIntType, arith::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::UIntType, arith::MulIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::UIntType, arith::DivUIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::UIntType, arith::RemUIOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::FloatType, arith::AddFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::FloatType, arith::SubFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::FloatType, arith::MulFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::FloatType, arith::DivFOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::FloatType, arith::RemFOp>>(typeConverter, patterns.getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::DecimalType, arith::AddIOp>>(typeConverter, patterns.getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::DecimalType, arith::SubIOp>>(typeConverter, patterns.getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::DivOp, arith::DivSIOp>>(typeConverter, patterns.getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::ModOp, arith::RemSIOp>>(typeConverter, patterns.getContext());
   patterns.insert<SubStrOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<NullOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<IsNullOpLowering>(typeConverter, patterns.getContext());
   patterns.insert<CombineNullOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<ConstantLowering>(typeConverter, patterns.getContext());
   patterns.insert<CastOpLowering>(typeConverter, patterns.getContext());

   patterns.insert<CreateFlagLowering>(typeConverter, patterns.getContext());
   patterns.insert<SetFlagLowering>(typeConverter, patterns.getContext());
   patterns.insert<GetFlagLowering>(typeConverter, patterns.getContext());
   patterns.insert<HashLowering>(typeConverter, patterns.getContext());
}