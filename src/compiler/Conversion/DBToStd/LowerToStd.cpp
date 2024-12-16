#include "lingodb/compiler/Conversion/DBToStd/DBToStd.h"
#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"
#include "lingodb/compiler/Dialect/DB/Passes.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/mlir-support/parsing.h"
#include "lingodb/compiler/runtime/StringRuntime.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
using namespace lingodb::compiler::dialect;
struct DBToStdLoweringPass
   : public PassWrapper<DBToStdLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DBToStdLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "to-arrow-std"; }

   DBToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect>();
   }
   void runOnOperation() final;
};
static TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      Type converted = typeConverter.convertType(t);
      converted = converted ? converted : t;
      types.push_back(converted);
   }
   return TupleType::get(tupleType.getContext(), TypeRange(types));
}
static bool hasDBType(TypeConverter& converter, TypeRange types) {
   return llvm::any_of(types, [&converter](mlir::Type t) { auto converted = converter.convertType(t);return converted&&converted!=t; });
}

template <class Op>
class SimpleTypeConversionPattern : public ConversionPattern {
   public:
   explicit SimpleTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, Op::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type> convertedTypes;
      if (typeConverter->convertTypes(op->getResultTypes(), convertedTypes).failed()) {
         return failure();
      }
      auto newOp = rewriter.create<Op>(op->getLoc(), convertedTypes, ValueRange(operands), op->getAttrs());
      rewriter.replaceOp(op, newOp->getResults());
      return success();
   }
};
mlir::Type convertPhysicalSingle(mlir::Type t, const TypeConverter& typeConverter) {
   MLIRContext* ctxt = t.getContext();
   mlir::Type arrowPhysicalType = typeConverter.convertType(t);
   if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(t)) {
      arrowPhysicalType = dsa::ArrowDecimalType::get(ctxt, decimalType.getP(), decimalType.getS());
   } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(t)) {
      arrowPhysicalType = dateType.getUnit() == db::DateUnitAttr::day ? (mlir::Type) dsa::ArrowDate32Type::get(t.getContext()) : (mlir::Type) dsa::ArrowDate64Type::get(t.getContext());
   } else if (mlir::isa<db::StringType>(t)) {
      arrowPhysicalType = dsa::ArrowStringType::get(t.getContext());
   } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(t)) {
      arrowPhysicalType = dsa::ArrowFixedSizedBinaryType::get(t.getContext(), charType.getBytes());
   } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(t)) {
      switch (timestampType.getUnit()) {
         case db::TimeUnitAttr::second:
            arrowPhysicalType = dsa::ArrowTimeStampType::get(t.getContext(), dsa::TimeUnitAttr::second);
            break;
         case db::TimeUnitAttr::millisecond:
            arrowPhysicalType = dsa::ArrowTimeStampType::get(t.getContext(), dsa::TimeUnitAttr::millisecond);
            break;
         case db::TimeUnitAttr::microsecond:
            arrowPhysicalType = dsa::ArrowTimeStampType::get(t.getContext(), dsa::TimeUnitAttr::nanosecond);
            break;
         case db::TimeUnitAttr::nanosecond:
            arrowPhysicalType = dsa::ArrowTimeStampType::get(t.getContext(), dsa::TimeUnitAttr::nanosecond);
            break;
      }
   } else if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(t)) {
      if (intervalType.getUnit() == db::IntervalUnitAttr::daytime) {
         arrowPhysicalType = dsa::ArrowDayTimeIntervalType::get(t.getContext());
      } else {
         arrowPhysicalType = dsa::ArrowMonthIntervalType::get(t.getContext());
      }
   }
   return arrowPhysicalType;
};
class AtLowering : public OpConversionPattern<dsa::At> {
   public:
   using OpConversionPattern<dsa::At>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::At atOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = atOp->getLoc();
      auto t = atOp.getType(0);
      if (typeConverter->isLegal(t)) {
         rewriter.startOpModification(atOp);
         atOp->setOperands(adaptor.getOperands());
         rewriter.finalizeOpModification(atOp);
         return mlir::success();
      }
      mlir::Type arrowPhysicalType = convertPhysicalSingle(t, *typeConverter);
      llvm::SmallVector<mlir::Type> types;
      types.push_back(arrowPhysicalType);
      if (atOp.getValid()) {
         types.push_back(rewriter.getI1Type());
      }
      std::vector<mlir::Value> values;
      auto newAtOp = rewriter.create<dsa::At>(loc, types, adaptor.getCollection(), atOp.getPos());
      mlir::Value nativeValue = newAtOp.getVal();
      if (typeConverter->convertType(t) != nativeValue.getType()) {
         nativeValue = rewriter.create<dsa::ArrowTypeTo>(loc, typeConverter->convertType(t), nativeValue);
      }
      values.push_back(nativeValue);
      if (atOp.getValid()) {
         values.push_back(newAtOp.getValid());
      }
      rewriter.replaceOp(atOp, values);
      return success();
   }
};

class AppendCBLowering : public ConversionPattern {
   public:
   explicit AppendCBLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, dsa::Append::getOperationName(), 2, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      dsa::AppendAdaptor adaptor(operands);
      auto appendOp = mlir::cast<dsa::Append>(op);
      if (!mlir::isa<dsa::ColumnBuilderType>(appendOp.getDs().getType())) {
         return mlir::failure();
      }
      auto t = appendOp.getVal().getType();
      if (typeConverter->isLegal(t)) {
         rewriter.startOpModification(op);
         appendOp->setOperands(operands);
         rewriter.finalizeOpModification(op);
         return mlir::success();
      }
      mlir::Type arrowPhysicalType = convertPhysicalSingle(t, *typeConverter);

      mlir::Value val = adaptor.getVal();
      if (val.getType() != arrowPhysicalType) {
         val = rewriter.create<dsa::ArrowTypeFrom>(loc, arrowPhysicalType, val);
      }
      rewriter.create<dsa::Append>(loc, adaptor.getDs(), val, adaptor.getValid());

      rewriter.eraseOp(op);
      return success();
   }
};

class StringCastOpLowering : public OpConversionPattern<db::CastOp> {
   public:
   using OpConversionPattern<db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::CastOp castOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      using namespace lingodb::compiler::runtime;
      auto loc = castOp->getLoc();
      auto scalarSourceType = castOp.getVal().getType();
      auto scalarTargetType = castOp.getType();
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (!mlir::isa<db::StringType>(scalarSourceType) && !mlir::isa<db::StringType>(scalarTargetType)) return failure();

      Value valueToCast = adaptor.getVal();
      Value result;
      if (scalarSourceType == scalarTargetType) {
         //nothing to do here
      } else if (auto stringType = mlir::dyn_cast_or_null<db::StringType>(scalarSourceType)) {
         if (auto intWidth = getIntegerWidth(scalarTargetType, false)) {
            result = StringRuntime::toInt(rewriter, loc)({valueToCast})[0];
            if (intWidth < 64) {
               result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, result);
            }
         } else if (auto floatType = mlir::dyn_cast_or_null<FloatType>(scalarTargetType)) {
            result = floatType.getWidth() == 32 ? StringRuntime::toFloat32(rewriter, loc)({valueToCast})[0] : StringRuntime::toFloat64(rewriter, loc)({valueToCast})[0];
         } else if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(scalarTargetType)) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalType.getS()));
            result = StringRuntime::toDecimal(rewriter, loc)({valueToCast, scale})[0];
            if (mlir::cast<mlir::IntegerType>(typeConverter->convertType(decimalType)).getWidth() < 128) {
               auto converted = rewriter.create<arith::TruncIOp>(loc, typeConverter->convertType(decimalType), result);
               result = converted;
            }
         } else if (mlir::isa<db::DateType>(scalarTargetType)) {
            result = StringRuntime::toDate(rewriter, loc)({valueToCast})[0];
         }
      } else if (scalarSourceType.isInteger(1)) {
         result = StringRuntime::fromBool(rewriter, loc)({valueToCast})[0];
      } else if (getIntegerWidth(scalarSourceType, false)) {
         result = StringRuntime::fromInt(rewriter, loc)({valueToCast})[0];
      } else if (auto floatType = mlir::dyn_cast_or_null<FloatType>(scalarSourceType)) {
         result = floatType.getWidth() == 32 ? StringRuntime::fromFloat32(rewriter, loc)({valueToCast})[0] : StringRuntime::fromFloat64(rewriter, loc)({valueToCast})[0];
      } else if (auto decimalSourceType = mlir::dyn_cast_or_null<db::DecimalType>(scalarSourceType)) {
         auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalSourceType.getS()));
         result = StringRuntime::fromDecimal(rewriter, loc)({valueToCast, scale})[0];
      } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(scalarSourceType)) {
         auto bytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(charType.getBytes()));
         result = StringRuntime::fromChar(rewriter, loc)({valueToCast, bytes})[0];
      } else if (mlir::isa<db::DateType>(scalarSourceType)) {
         result = StringRuntime::fromDate(rewriter, loc)({valueToCast})[0];
      }
      if (result) {
         rewriter.replaceOp(castOp, result);
         return success();
      } else {
         return failure();
      }
   }
};

class CharCmpOpLowering : public OpConversionPattern<db::CmpOp> {
   public:
   using OpConversionPattern<db::CmpOp>::OpConversionPattern;

   LogicalResult matchAndRewrite(db::CmpOp cmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = cmpOp.getLeft().getType();
      auto charType = mlir::dyn_cast_or_null<db::CharType>(type);
      if (!charType) {
         return failure();
      }
      if (cmpOp.getPredicate() == db::DBCmpPredicate::eq) {
         rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, mlir::arith::CmpIPredicate::eq, adaptor.getLeft(), adaptor.getRight());
         return success();
      }
      if (cmpOp.getPredicate() == db::DBCmpPredicate::neq) {
         rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, mlir::arith::CmpIPredicate::ne, adaptor.getLeft(), adaptor.getRight());
         return success();
      }
      mlir::Value left = rewriter.create<db::CastOp>(cmpOp->getLoc(), db::StringType::get(getContext()), cmpOp.getLeft());
      mlir::Value right = rewriter.create<db::CastOp>(cmpOp->getLoc(), db::StringType::get(getContext()), cmpOp.getRight());
      rewriter.replaceOpWithNewOp<db::CmpOp>(cmpOp, cmpOp.getPredicate(), left, right);
      return success();
   }
};
class StringCmpOpLowering : public OpConversionPattern<db::CmpOp> {
   public:
   using OpConversionPattern<db::CmpOp>::OpConversionPattern;

   bool stringIsOk(std::string str) const {
      for (auto x : str) {
         if (!std::isalnum(x)) return false;
      }
      return true;
   }
   LogicalResult matchAndRewrite(db::CmpOp cmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      using namespace lingodb::compiler::runtime;
      auto type = cmpOp.getLeft().getType();
      if (!mlir::isa<db::StringType>(type)) {
         return failure();
      }
      Value res;
      Value left = adaptor.getLeft();
      Value right = adaptor.getRight();
      switch (cmpOp.getPredicate()) {
         case db::DBCmpPredicate::eq: {
            auto varlenCmp = rewriter.create<util::VarLenCmp>(cmpOp->getLoc(), rewriter.getI1Type(), rewriter.getI1Type(), left, right);
            res = rewriter.create<mlir::scf::IfOp>(
                             cmpOp->getLoc(), varlenCmp.getNeedsDetailedEval(), [&](mlir::OpBuilder& builder, mlir::Location loc) {
                                                         auto rtCmp=StringRuntime::compareEq(rewriter, cmpOp->getLoc())({left, right})[0];
                                                         builder.create<mlir::scf::YieldOp>(loc, rtCmp); }, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, varlenCmp.getEq()); })
                     .getResult(0);
            break;
         }
         case db::DBCmpPredicate::neq:
            res = StringRuntime::compareNEq(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lt:
            res = StringRuntime::compareLt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gt:
            res = StringRuntime::compareGt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lte:
            res = StringRuntime::compareLte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gte:
            res = StringRuntime::compareGte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::isa:
            assert(false && "should not happen");
            break;
      }
      rewriter.replaceOp(cmpOp, res);
      return success();
   }
};

class RuntimeCallLowering : public OpConversionPattern<db::RuntimeCall> {
   public:
   using OpConversionPattern<db::RuntimeCall>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::RuntimeCall runtimeCallOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto reg = getContext()->getLoadedDialect<db::DBDialect>()->getRuntimeFunctionRegistry();
      auto* fn = reg->lookup(runtimeCallOp.getFn().str());
      if (!fn) return failure();
      Value result;
      mlir::Type resType = runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type();
      if (std::holds_alternative<util::FunctionSpec>(fn->implementation)) {
         auto& implFn = std::get<util::FunctionSpec>(fn->implementation);
         auto resRange = implFn(rewriter, runtimeCallOp->getLoc())(adaptor.getArgs());
         assert((resRange.size() == 1 && resType) || (resRange.empty() && !resType));
         result = resRange.size() == 1 ? resRange[0] : mlir::Value();
      } else if (std::holds_alternative<db::RuntimeFunction::loweringFnT>(fn->implementation)) {
         auto& implFn = std::get<db::RuntimeFunction::loweringFnT>(fn->implementation);
         result = implFn(rewriter, adaptor.getArgs(), runtimeCallOp.getArgs().getTypes(), runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type(), typeConverter, runtimeCallOp->getLoc());
      }

      if (runtimeCallOp->getNumResults() == 0) {
         rewriter.eraseOp(runtimeCallOp);
      } else {
         rewriter.replaceOp(runtimeCallOp, result);
      }
      return success();
   }
};

class NotOpLowering : public OpConversionPattern<db::NotOp> {
   public:
   using OpConversionPattern<db::NotOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::NotOp notOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value falseValue = rewriter.create<arith::ConstantOp>(notOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(notOp, mlir::arith::CmpIPredicate::eq, adaptor.getVal(), falseValue);
      return success();
   }
};
class AndOpLowering : public OpConversionPattern<db::AndOp> {
   public:
   using OpConversionPattern<db::AndOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::AndOp andOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value result;
      Value isNull;
      auto loc = andOp->getLoc();

      for (size_t i = 0; i < adaptor.getVals().size(); i++) {
         auto currType = andOp.getVals()[i].getType();
         bool currNullable = mlir::isa<db::NullableType>(currType);
         Value currNull;
         Value currVal;
         if (currNullable) {
            auto unPackOp = rewriter.create<util::UnPackOp>(loc, adaptor.getVals()[i]);
            currNull = unPackOp.getVals()[0];
            currVal = unPackOp.getVals()[1];
         } else {
            currVal = adaptor.getVals()[i];
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
      if (mlir::isa<db::NullableType>(andOp.getResult().getType())) {
         isNull = rewriter.create<arith::AndIOp>(loc, result, isNull);
         Value combined = rewriter.create<util::PackOp>(loc, ValueRange({isNull, result}));
         rewriter.replaceOp(andOp, combined);
      } else {
         rewriter.replaceOp(andOp, result);
      }
      return success();
   }
};
class OrOpLowering : public OpConversionPattern<db::OrOp> {
   public:
   using OpConversionPattern<db::OrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::OrOp orOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value result;
      Value isNull;
      auto loc = orOp->getLoc();
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      for (size_t i = 0; i < adaptor.getVals().size(); i++) {
         auto currType = orOp.getVals()[i].getType();
         bool currNullable = mlir::isa<db::NullableType>(currType);
         Value currNull;
         Value currVal;
         if (currNullable) {
            auto unPackOp = rewriter.create<util::UnPackOp>(loc, adaptor.getVals()[i]);
            currNull = unPackOp.getVals()[0];
            currVal = unPackOp.getVals()[1];
         } else {
            currVal = adaptor.getVals()[i];
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
      if (mlir::isa<db::NullableType>(orOp.getResult().getType())) {
         isNull = rewriter.create<arith::SelectOp>(loc, result, falseValue, isNull);
         Value combined = rewriter.create<util::PackOp>(loc, ValueRange({isNull, result}));
         rewriter.replaceOp(orOp, combined);
      } else {
         rewriter.replaceOp(orOp, result);
      }
      return success();
   }
};

template <class OpClass, class OperandType, class StdOpClass>
class BinOpLowering : public OpConversionPattern<OpClass> {
   public:
   using OpConversionPattern<OpClass>::OpConversionPattern;
   LogicalResult matchAndRewrite(OpClass binOp, typename OpConversionPattern<OpClass>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = getBaseType(binOp.getLeft().getType());
      if (mlir::isa<OperandType>(type)) {
         rewriter.template replaceOpWithNewOp<StdOpClass>(binOp, this->typeConverter->convertType(binOp.getResult().getType()), adaptor.getLeft(), adaptor.getRight());
         return success();
      }
      return failure();
   }
};
mlir::Value getDecimalScaleMultiplierConstant(mlir::OpBuilder& builder, int32_t s, mlir::Type stdType, mlir::Location loc) {
   auto [low, high] = lingodb::compiler::support::getDecimalScaleMultiplier(s);
   std::vector<uint64_t> parts = {low, high};
   auto multiplier = builder.create<arith::ConstantOp>(loc, stdType, builder.getIntegerAttr(stdType, APInt(mlir::cast<mlir::IntegerType>(stdType).getWidth(), parts)));
   return multiplier;
}
template <class DBOp, class Op>
class DecimalOpScaledLowering : public OpConversionPattern<DBOp> {
   public:
   using OpConversionPattern<DBOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(DBOp decimalOp, typename OpConversionPattern<DBOp>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = getBaseType(decimalOp.getType());
      if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
         auto stdType = this->typeConverter->convertType(decimalType);
         mlir::Value left = adaptor.getLeft();
         mlir::Value right = adaptor.getRight();
         if (stdType != left.getType()) {
            left = rewriter.create<mlir::arith::ExtSIOp>(decimalOp->getLoc(), stdType, left);
         }
         if (stdType != right.getType()) {
            right = rewriter.create<mlir::arith::ExtSIOp>(decimalOp->getLoc(), stdType, right);
         }
         auto scaled = rewriter.create<arith::MulIOp>(decimalOp->getLoc(), stdType, left, getDecimalScaleMultiplierConstant(rewriter, decimalType.getS() + mlir::cast<db::DecimalType>(decimalOp.getRight().getType()).getS() - mlir::cast<db::DecimalType>(decimalOp.getLeft().getType()).getS(), stdType, decimalOp->getLoc()));
         rewriter.template replaceOpWithNewOp<Op>(decimalOp, stdType, scaled, right);
         return success();
      }
      return failure();
   }
};
class DecimalMulOpLowering : public OpConversionPattern<db::MulOp> {
   public:
   using OpConversionPattern<db::MulOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::MulOp mulOp, typename OpConversionPattern<db::MulOp>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(mulOp.getType())) {
         auto leftScale = mlir::cast<db::DecimalType>(mulOp.getLeft().getType()).getS();
         auto rightScale = mlir::cast<db::DecimalType>(mulOp.getRight().getType()).getS();
         auto stdType = this->typeConverter->convertType(decimalType);
         mlir::Value left = adaptor.getLeft();
         mlir::Value right = adaptor.getRight();
         if (stdType != left.getType()) {
            left = rewriter.create<mlir::arith::ExtSIOp>(mulOp->getLoc(), stdType, left);
         }
         if (stdType != right.getType()) {
            right = rewriter.create<mlir::arith::ExtSIOp>(mulOp->getLoc(), stdType, right);
         }
         mlir::Value result = rewriter.create<arith::MulIOp>(mulOp->getLoc(), stdType, left, right);
         if (decimalType.getS() < leftScale + rightScale) {
            result = rewriter.create<arith::DivSIOp>(mulOp->getLoc(), stdType, result, getDecimalScaleMultiplierConstant(rewriter, leftScale + rightScale - decimalType.getS(), stdType, mulOp->getLoc()));
         }
         rewriter.replaceOp(mulOp, result);
         return success();
      }
      return failure();
   }
};
template <class DBOp, class ArithOp>
class DecimalBinOpLowering : public OpConversionPattern<DBOp> {
   public:
   using OpConversionPattern<DBOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(DBOp mulOp, typename OpConversionPattern<DBOp>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(mulOp.getType())) {
         auto stdType = this->typeConverter->convertType(decimalType);
         mlir::Value left = adaptor.getLeft();
         mlir::Value right = adaptor.getRight();
         if (stdType != left.getType()) {
            left = rewriter.create<mlir::arith::ExtSIOp>(mulOp->getLoc(), stdType, left);
         }
         if (stdType != right.getType()) {
            right = rewriter.create<mlir::arith::ExtSIOp>(mulOp->getLoc(), stdType, right);
         }
         mlir::Value multiplied = rewriter.create<ArithOp>(mulOp->getLoc(), stdType, left, right);
         rewriter.replaceOp(mulOp, multiplied);
         return success();
      }
      return failure();
   }
};
class IsNullOpLowering : public OpConversionPattern<db::IsNullOp> {
   public:
   using OpConversionPattern<db::IsNullOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::IsNullOp isNullOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (mlir::isa<db::NullableType>(isNullOp.getVal().getType())) {
         auto unPackOp = rewriter.create<util::UnPackOp>(isNullOp->getLoc(), adaptor.getVal());
         rewriter.replaceOp(isNullOp, unPackOp.getVals()[0]);
      } else {
         rewriter.replaceOp(isNullOp, adaptor.getVal());
      }
      return success();
   }
};
class NullableGetValOpLowering : public OpConversionPattern<db::NullableGetVal> {
   public:
   using OpConversionPattern<db::NullableGetVal>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::NullableGetVal op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto unPackOp = rewriter.create<util::UnPackOp>(op->getLoc(), adaptor.getVal());
      rewriter.replaceOp(op, unPackOp.getVals()[1]);
      return success();
   }
};
class AsNullableOpLowering : public OpConversionPattern<db::AsNullableOp> {
   public:
   using OpConversionPattern<db::AsNullableOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::AsNullableOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Value isNull = adaptor.getNull();
      if (!isNull) {
         isNull = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      }
      auto packOp = rewriter.create<util::PackOp>(op->getLoc(), ValueRange({isNull, adaptor.getVal()}));
      rewriter.replaceOp(op, packOp.getTuple());
      return success();
   }
};
class NullOpLowering : public OpConversionPattern<db::NullOp> {
   public:
   using OpConversionPattern<db::NullOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::NullOp nullOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto tupleType = mlir::cast<mlir::TupleType>(typeConverter->convertType(nullOp.getType()));
      auto undefValue = rewriter.create<util::UndefOp>(nullOp->getLoc(), tupleType.getType(1));
      auto trueValue = rewriter.create<arith::ConstantOp>(nullOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      rewriter.replaceOpWithNewOp<util::PackOp>(nullOp, tupleType, ValueRange({trueValue, undefValue}));
      return success();
   }
};

class ConstantLowering : public OpConversionPattern<db::ConstantOp> {
   static std::tuple<arrow::Type::type, uint32_t, uint32_t> convertTypeToArrow(mlir::Type type) {
      arrow::Type::type typeConstant = arrow::Type::type::NA;
      uint32_t param1 = 0, param2 = 0;
      if (isIntegerType(type, 1)) {
         typeConstant = arrow::Type::type::BOOL;
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         switch (intWidth) {
            case 8: typeConstant = arrow::Type::type::INT8; break;
            case 16: typeConstant = arrow::Type::type::INT16; break;
            case 32: typeConstant = arrow::Type::type::INT32; break;
            case 64: typeConstant = arrow::Type::type::INT64; break;
         }
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         switch (uIntWidth) {
            case 8: typeConstant = arrow::Type::type::UINT8; break;
            case 16: typeConstant = arrow::Type::type::UINT16; break;
            case 32: typeConstant = arrow::Type::type::UINT32; break;
            case 64: typeConstant = arrow::Type::type::UINT64; break;
         }
      } else if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
         typeConstant = arrow::Type::type::DECIMAL128;
         param1 = decimalType.getP();
         param2 = decimalType.getS();
      } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
         switch (floatType.getWidth()) {
            case 16: typeConstant = arrow::Type::type::HALF_FLOAT; break;
            case 32: typeConstant = arrow::Type::type::FLOAT; break;
            case 64: typeConstant = arrow::Type::type::DOUBLE; break;
         }
      } else if (auto stringType = mlir::dyn_cast_or_null<db::StringType>(type)) {
         typeConstant = arrow::Type::type::STRING;
      } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
         if (dateType.getUnit() == db::DateUnitAttr::day) {
            typeConstant = arrow::Type::type::DATE32;
         } else {
            typeConstant = arrow::Type::type::DATE64;
         }
      } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(type)) {
         typeConstant = arrow::Type::type::FIXED_SIZE_BINARY;
         param1 = charType.getBytes();
      } else if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(type)) {
         if (intervalType.getUnit() == db::IntervalUnitAttr::months) {
            typeConstant = arrow::Type::type::INTERVAL_MONTHS;
         } else {
            typeConstant = arrow::Type::type::INTERVAL_DAY_TIME;
         }
      } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(type)) {
         typeConstant = arrow::Type::type::TIMESTAMP;
         param1 = static_cast<uint32_t>(timestampType.getUnit());
      }
      assert(typeConstant != arrow::Type::type::NA);
      return {typeConstant, param1, param2};
   }

   public:
   using OpConversionPattern<db::ConstantOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::ConstantOp constantOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = constantOp.getType();
      auto stdType = typeConverter->convertType(type);
      auto [arrowType, param1, param2] = convertTypeToArrow(type);
      std::variant<int64_t, double, std::string> parseArg;
      if (auto integerAttr = mlir::dyn_cast_or_null<IntegerAttr>(constantOp.getValue())) {
         parseArg = integerAttr.getInt();
      } else if (auto floatAttr = mlir::dyn_cast_or_null<FloatAttr>(constantOp.getValue())) {
         parseArg = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = mlir::dyn_cast_or_null<StringAttr>(constantOp.getValue())) {
         parseArg = stringAttr.str();
      } else {
         return failure();
      }
      auto parseResult = lingodb::compiler::support::parse(parseArg, arrowType, param1, param2);
      if (auto intType = mlir::dyn_cast_or_null<IntegerType>(stdType)) {
         if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
            auto [low, high] = lingodb::compiler::support::parseDecimal(std::get<std::string>(parseResult), decimalType.getS());
            std::vector<uint64_t> parts = {low, high};
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType, rewriter.getIntegerAttr(stdType, APInt(mlir::cast<mlir::IntegerType>(stdType).getWidth(), parts)));
            return success();
         } else {
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType, rewriter.getIntegerAttr(stdType, std::get<int64_t>(parseResult)));
            return success();
         }
      } else if (auto floatType = mlir::dyn_cast_or_null<FloatType>(stdType)) {
         rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType, rewriter.getFloatAttr(stdType, std::get<double>(parseResult)));
         return success();
      } else if (mlir::isa<db::StringType>(type)) {
         std::string str = std::get<std::string>(parseResult);

         rewriter.replaceOpWithNewOp<util::CreateConstVarLen>(constantOp, util::VarLen32Type::get(rewriter.getContext()), rewriter.getStringAttr(str));
         return success();
      } else {
         return failure();
      }
      return failure();
   }
};
class CmpOpLowering : public OpConversionPattern<db::CmpOp> {
   public:
   using OpConversionPattern<db::CmpOp>::OpConversionPattern;
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
         case db::DBCmpPredicate::isa:
            assert(false && "should not happen");
            break;
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
         case db::DBCmpPredicate::isa:
            assert(false && "should not happen");
            break;
      }
      assert(false && "unexpected case");
      return arith::CmpFPredicate::OEQ;
   }
   LogicalResult matchAndRewrite(db::CmpOp cmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (mlir::isa<db::CharType>(cmpOp.getLeft().getType())) {
         return failure();
      }
      if (!adaptor.getLeft().getType().isIntOrIndexOrFloat()) {
         return failure();
      }
      if (adaptor.getLeft().getType().isIntOrIndex()) {
         rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, translateIPredicate(cmpOp.getPredicate()), adaptor.getLeft(), adaptor.getRight());
      } else {
         rewriter.replaceOpWithNewOp<arith::CmpFOp>(cmpOp, translateFPredicate(cmpOp.getPredicate()), adaptor.getLeft(), adaptor.getRight());
      }
      return success();
   }
};
class CastNoneOpLowering : public OpConversionPattern<db::CastOp> {
   public:
   using OpConversionPattern<db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::CastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto scalarSourceType = op.getVal().getType();
      auto scalarTargetType = op.getType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (mlir::isa<db::StringType>(scalarSourceType) || mlir::isa<db::StringType>(scalarTargetType)) return failure();
      if (!mlir::isa<NoneType>(convertedSourceType)) {
         return mlir::failure();
      }
      rewriter.replaceOpWithNewOp<util::UndefOp>(op, convertedTargetType);
      return mlir::success();
   }
};
class CastOpLowering : public OpConversionPattern<db::CastOp> {
   public:
   using OpConversionPattern<db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::CastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto scalarSourceType = op.getVal().getType();
      auto scalarTargetType = op.getType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (mlir::isa<db::StringType>(scalarSourceType) || mlir::isa<db::StringType>(scalarTargetType)) return failure();
      Value value = adaptor.getVal();
      if (scalarSourceType == scalarTargetType) {
         rewriter.replaceOp(op, value);
         return success();
      }
      if (mlir::isa<db::CharType>(scalarSourceType) || mlir::isa<db::CharType>(scalarTargetType)) {
         mlir::Value castedToString = rewriter.create<db::CastOp>(op.getLoc(), db::StringType::get(getContext()), op.getVal());
         rewriter.replaceOpWithNewOp<db::CastOp>(op, scalarTargetType, castedToString);
         return success();
      }
      if (auto sourceIntWidth = getIntegerWidth(scalarSourceType, false)) {
         if (mlir::isa<FloatType>(scalarTargetType)) {
            value = rewriter.create<arith::SIToFPOp>(loc, convertedTargetType, value);
            rewriter.replaceOp(op, value);
            return success();
         } else if (auto decimalTargetType = mlir::dyn_cast_or_null<db::DecimalType>(scalarTargetType)) {
            int decimalWidth = mlir::cast<mlir::IntegerType>(typeConverter->convertType(decimalTargetType)).getWidth();
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
      } else if (auto floatType = mlir::dyn_cast_or_null<FloatType>(scalarSourceType)) {
         if (getIntegerWidth(scalarTargetType, false)) {
            value = rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, convertedTargetType, value);
            return success();
         } else if (auto decimalTargetType = mlir::dyn_cast_or_null<db::DecimalType>(scalarTargetType)) {
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedSourceType, FloatAttr::get(convertedSourceType, powf(10, decimalTargetType.getS())));
            value = rewriter.create<arith::MulFOp>(loc, convertedSourceType, value, multiplier);
            rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, convertedTargetType, value);
            return success();
         }
      } else if (auto decimalSourceType = mlir::dyn_cast_or_null<db::DecimalType>(scalarSourceType)) {
         if (auto decimalTargetType = mlir::dyn_cast_or_null<db::DecimalType>(scalarTargetType)) {
            auto sourceScale = decimalSourceType.getS();
            auto targetScale = decimalTargetType.getS();
            size_t decimalWidth = mlir::cast<mlir::IntegerType>(convertedTargetType).getWidth();
            if (convertedSourceType.getIntOrFloatBitWidth() < convertedTargetType.getIntOrFloatBitWidth()) {
               value = rewriter.create<mlir::arith::ExtSIOp>(loc, convertedTargetType, value);
            }
            auto [low, high] = lingodb::compiler::support::getDecimalScaleMultiplier(std::max(sourceScale, targetScale) - std::min(sourceScale, targetScale));
            std::vector<uint64_t> parts = {low, high};
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedTargetType, rewriter.getIntegerAttr(convertedTargetType, APInt(decimalWidth, parts)));
            if (sourceScale < targetScale) {
               rewriter.replaceOpWithNewOp<arith::MulIOp>(op, convertedTargetType, value, multiplier);
            } else {
               rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, convertedTargetType, value, multiplier);
            }
            return success();
         } else if (mlir::isa<FloatType>(scalarTargetType)) {
            auto multiplier = rewriter.create<arith::ConstantOp>(loc, convertedTargetType, FloatAttr::get(convertedTargetType, powf(10, decimalSourceType.getS())));
            value = rewriter.create<arith::SIToFPOp>(loc, convertedTargetType, value);
            rewriter.replaceOpWithNewOp<arith::DivFOp>(op, convertedTargetType, value, multiplier);
            return success();
         } else if (auto targetIntWidth = getIntegerWidth(scalarTargetType, false)) {
            int decimalWidth = mlir::cast<mlir::IntegerType>(convertedSourceType).getWidth();
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
class BetweenLowering : public OpConversionPattern<db::BetweenOp> {
   public:
   using OpConversionPattern<db::BetweenOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::BetweenOp betweenOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto isGteLower = rewriter.create<db::CmpOp>(betweenOp->getLoc(), betweenOp.getLowerInclusive() ? db::DBCmpPredicate::gte : db::DBCmpPredicate::gt, betweenOp.getVal(), betweenOp.getLower());
      auto isLteUpper = rewriter.create<db::CmpOp>(betweenOp->getLoc(), betweenOp.getUpperInclusive() ? db::DBCmpPredicate::lte : db::DBCmpPredicate::lt, betweenOp.getVal(), betweenOp.getUpper());
      auto isInRange = rewriter.create<db::AndOp>(betweenOp->getLoc(), ValueRange({isGteLower, isLteUpper}));
      rewriter.replaceOp(betweenOp, isInRange.getRes());
      return success();
   }
};
class OneOfLowering : public OpConversionPattern<db::OneOfOp> {
   public:
   using OpConversionPattern<db::OneOfOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::OneOfOp oneOfOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Value> compared;
      for (auto ele : oneOfOp.getVals()) {
         compared.push_back(rewriter.create<db::CmpOp>(oneOfOp->getLoc(), db::DBCmpPredicate::eq, oneOfOp.getVal(), ele));
      }
      auto isInRange = rewriter.create<db::OrOp>(oneOfOp->getLoc(), compared);
      rewriter.replaceOp(oneOfOp, isInRange.getRes());
      return success();
   }
};
class SortCompareLowering : public OpConversionPattern<db::SortCompare> {
   public:
   using OpConversionPattern<db::SortCompare>::OpConversionPattern;
   LogicalResult matchAndRewrite(db::SortCompare sortCompareOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = sortCompareOp->getLoc();
      if (sortCompareOp.getLeft().getType() != sortCompareOp.getRight().getType() || mlir::isa<db::NullableType>(sortCompareOp.getLeft().getType())) {
         return failure();
      }
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
      mlir::Value minus1 = rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, 8);
      mlir::Value one = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 8);
      mlir::Value lt = rewriter.create<db::CmpOp>(loc, db::DBCmpPredicate::lt, sortCompareOp.getLeft(), sortCompareOp.getRight());
      mlir::Value eq = rewriter.create<db::CmpOp>(loc, db::DBCmpPredicate::eq, sortCompareOp.getLeft(), sortCompareOp.getRight());
      mlir::Value res = rewriter.create<mlir::arith::SelectOp>(loc, eq, zero, one);
      res = rewriter.create<mlir::arith::SelectOp>(loc, lt, minus1, res);
      rewriter.replaceOp(sortCompareOp, res);
      return success();
   }
};
class HashLowering : public ConversionPattern {
   Value combineHashes(OpBuilder& builder, Location loc, Value hash1, Value totalHash) const {
      if (!totalHash) {
         return hash1;
      } else {
         return builder.create<util::HashCombine>(loc, builder.getIndexType(), hash1, totalHash);
      }
   }
   Value hashInteger(OpBuilder& builder, Location loc, Value integer) const {
      Value asIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), integer);
      return builder.create<util::Hash64>(loc, builder.getIndexType(), asIndex);
   }
   Value hashImpl(OpBuilder& builder, Location loc, Value v, Value totalHash, Type originalType) const {
      if (auto intType = mlir::dyn_cast_or_null<mlir::IntegerType>(v.getType())) {
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

      } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(v.getType())) {
         Value asInt = builder.create<arith::BitcastOp>(loc, IntegerType::get(builder.getContext(), floatType.getWidth()), v);
         return combineHashes(builder, loc, hashInteger(builder, loc, asInt), totalHash);
      } else if (auto varLenType = mlir::dyn_cast_or_null<util::VarLen32Type>(v.getType())) {
         auto varlenTryCheapHash = builder.create<util::VarLenTryCheapHash>(loc, builder.getI1Type(), builder.getIndexType(), v);
         mlir::Value hash = builder.create<mlir::scf::IfOp>(
                                      loc, varlenTryCheapHash.getComplete(), [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, varlenTryCheapHash.getHash()); }, [&](mlir::OpBuilder& builder, mlir::Location loc) {
                                         mlir::Value hash = builder.create<util::HashVarLen>(loc, builder.getIndexType(), v);
                                         builder.create<mlir::scf::YieldOp>(loc, hash); })
                               .getResult(0);

         return combineHashes(builder, loc, hash, totalHash);
      } else if (auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(v.getType())) {
         if (auto originalTupleType = mlir::dyn_cast_or_null<mlir::TupleType>(originalType)) {
            auto unpacked = builder.create<util::UnPackOp>(loc, v);
            size_t i = 0;
            for (auto v : unpacked->getResults()) {
               totalHash = hashImpl(builder, loc, v, totalHash, originalTupleType.getType(i++));
            }
            return totalHash;
         } else if (mlir::isa<db::NullableType>(originalType)) {
            auto unpacked = builder.create<util::UnPackOp>(loc, v);

            mlir::Value totalHashUpdated = builder.create<mlir::scf::IfOp>(
                                                     loc, unpacked.getResult(0), [totalHashBefore = totalHash](mlir::OpBuilder& builder, mlir::Location loc) {
                                                         mlir::Value totalHash=totalHashBefore;
                                                        if (!totalHash) {
                                                           totalHash = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
                                                        }
                                               builder.create<mlir::scf::YieldOp>(loc,totalHash); }, [&](mlir::OpBuilder& builder, mlir::Location loc) {
                                               mlir::Value hashedIfNotNull = hashImpl(builder, loc, unpacked.getResult(1), totalHash, getBaseType(originalType));

                                               builder.create<mlir::scf::YieldOp>(loc,hashedIfNotNull); })
                                              .getResult(0);
            return totalHashUpdated;
         }
         assert(false && "should not happen");
         return Value();
      }
      assert(false && "should not happen");
      return Value();
   }

   public:
   explicit HashLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, db::Hash::getOperationName(), 1, context) {}
   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      db::HashAdaptor hashAdaptor(operands);
      auto hashOp = mlir::cast<db::Hash>(op);

      rewriter.replaceOp(op, hashImpl(rewriter, op->getLoc(), hashAdaptor.getVal(), Value(), hashOp.getVal().getType()));
      return success();
   }
};
} // end anonymous namespace
void DBToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<util::UtilDialect>()->getFunctionHelper().setParentModule(module);

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalDialect<gpu::GPUDialect>();
   target.addLegalDialect<async::AsyncDialect>();
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   TypeConverter typeConverter;
   auto* ctxt = &getContext();
   typeConverter.addConversion([&](mlir::Type type) { return type; });
   typeConverter.addConversion([&](::db::DateType t) {
      return mlir::IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](::db::DecimalType t) {
      if (t.getP() < 19) {
         return mlir::IntegerType::get(ctxt, 64);

      } else {
         return mlir::IntegerType::get(ctxt, 128);
      }
   });
   typeConverter.addConversion([&](::db::CharType t) {
      if (t.getBytes() > 8) return mlir::Type();
      size_t bits = 0;
      if (t.getBytes() == 1) {
         bits = 8;
      } else if (t.getBytes() == 2) {
         bits = 16;
      } else if (t.getBytes() <= 4) {
         bits = 32;
      } else {
         bits = 64;
      }
      return (Type) mlir::IntegerType::get(ctxt, bits);
   });
   typeConverter.addConversion([&](::db::StringType t) {
      return util::VarLen32Type::get(ctxt);
   });
   typeConverter.addConversion([&](::db::TimestampType t) {
      return mlir::IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](::db::IntervalType t) {
      if (t.getUnit() == db::IntervalUnitAttr::daytime) {
         return mlir::IntegerType::get(ctxt, 64);
      } else {
         return mlir::IntegerType::get(ctxt, 32);
      }
   });

   typeConverter.addConversion([&](db::NullableType type) {
      mlir::Type payloadType = typeConverter.convertType(type.getType());
      if (mlir::isa<mlir::NoneType>(payloadType)) {
         payloadType = IntegerType::get(ctxt, 1);
      }
      return (Type) TupleType::get(ctxt, {IntegerType::get(ctxt, 1), payloadType});
   });
   auto opIsWithoutDBTypes = [&](Operation* op) { return !hasDBType(typeConverter, op->getOperandTypes()) && !hasDBType(typeConverter, op->getResultTypes()); };
   target.addDynamicallyLegalDialect<scf::SCFDialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<dsa::DSADialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<arith::ArithDialect>(opIsWithoutDBTypes);

   target.addLegalDialect<cf::ControlFlowDialect>();

   target.addDynamicallyLegalDialect<util::UtilDialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isLegal = !hasDBType(typeConverter, op.getFunctionType().getInputs()) &&
         !hasDBType(typeConverter, op.getFunctionType().getResults());
      return isLegal;
   });
   target.addDynamicallyLegalOp<mlir::func::ConstantOp>([&](mlir::func::ConstantOp op) {
      if (auto functionType = mlir::dyn_cast_or_null<mlir::FunctionType>(op.getType())) {
         auto isLegal = !hasDBType(typeConverter, functionType.getInputs()) &&
            !hasDBType(typeConverter, functionType.getResults());
         return isLegal;
      } else {
         return true;
      }
   });
   target.addDynamicallyLegalOp<mlir::func::CallOp, mlir::func::CallIndirectOp, mlir::func::ReturnOp>(opIsWithoutDBTypes);

   target.addDynamicallyLegalOp<util::SizeOfOp>(
      [&typeConverter](util::SizeOfOp op) {
         auto isLegal = !hasDBType(typeConverter, op.getType());
         return isLegal;
      });

   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });

   auto convertPhysical = [&](mlir::TupleType tuple) -> mlir::TupleType {
      std::vector<mlir::Type> types;
      for (auto t : tuple.getTypes()) {
         mlir::Type arrowPhysicalType = convertPhysicalSingle(t, typeConverter);
         types.push_back(arrowPhysicalType);
      }
      return mlir::TupleType::get(tuple.getContext(), types);
   };
   typeConverter.addConversion([&](dsa::RecordType r) {
      return dsa::RecordType::get(r.getContext(), convertPhysical(r.getRowType()));
   });
   typeConverter.addConversion([&](dsa::RecordBatchType r) {
      auto res = dsa::RecordBatchType::get(r.getContext(), convertPhysical(r.getRowType()));
      return res;
   });
   typeConverter.addConversion([&](dsa::ColumnBuilderType r) {
      return dsa::ColumnBuilderType::get(r.getContext(), convertPhysicalSingle(r.getType(), typeConverter));
   });
   typeConverter.addConversion([&](dsa::ColumnType r) {
      return dsa::ColumnType::get(r.getContext(), convertPhysicalSingle(r.getType(), typeConverter));
   });
   typeConverter.addConversion([&](dsa::TableType r) { return dsa::TableType::get(r.getContext(), convertPhysical(r.getRowType())); });
   RewritePatternSet patterns(&getContext());

   mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   mlir::scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
   patterns.insert<SimpleTypeConversionPattern<mlir::func::ConstantOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::func::CallIndirectOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::GetRecord>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::GetRecordBatchLen>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::Append>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::DownCast>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::CreateDS>>(typeConverter, &getContext());
   //patterns.insert<SimpleTypeConversionPattern<dsa::YieldOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::CreateTable>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::Concat>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::FinishColumn>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<dsa::SetResultOp>>(typeConverter, &getContext());
   patterns.insert<AtLowering>(typeConverter, &getContext());
   patterns.insert<AppendCBLowering>(typeConverter, &getContext());
   //patterns.insert<SimpleTypeConversionPattern<dsa::ForOp>>(typeConverter, &getContext());
   patterns.insert<StringCmpOpLowering>(typeConverter, ctxt);
   patterns.insert<CharCmpOpLowering>(typeConverter, ctxt);
   patterns.insert<StringCastOpLowering>(typeConverter, ctxt);
   patterns.insert<RuntimeCallLowering>(typeConverter, ctxt);
   patterns.insert<CmpOpLowering>(typeConverter, ctxt);
   patterns.insert<BetweenLowering>(typeConverter, ctxt);
   patterns.insert<OneOfLowering>(typeConverter, ctxt);
   patterns.insert<SortCompareLowering>(typeConverter, ctxt);

   patterns.insert<NotOpLowering>(typeConverter, ctxt);

   patterns.insert<AndOpLowering>(typeConverter, ctxt);
   patterns.insert<OrOpLowering>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::AddOp, mlir::IntegerType, arith::AddIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::SubOp, mlir::IntegerType, arith::SubIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::MulOp, mlir::IntegerType, arith::MulIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::DivOp, mlir::IntegerType, arith::DivSIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::ModOp, mlir::IntegerType, arith::RemSIOp>>(typeConverter, ctxt);

   patterns.insert<BinOpLowering<db::AddOp, mlir::FloatType, arith::AddFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::SubOp, mlir::FloatType, arith::SubFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::MulOp, mlir::FloatType, arith::MulFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::DivOp, mlir::FloatType, arith::DivFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<db::ModOp, mlir::FloatType, arith::RemFOp>>(typeConverter, ctxt);

   patterns.insert<DecimalBinOpLowering<db::AddOp, arith::AddIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalBinOpLowering<db::SubOp, arith::SubIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalMulOpLowering>(typeConverter, ctxt);
   patterns.insert<DecimalOpScaledLowering<db::DivOp, arith::DivSIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalOpScaledLowering<db::ModOp, arith::RemSIOp>>(typeConverter, ctxt);

   patterns.insert<NullOpLowering>(typeConverter, ctxt);
   patterns.insert<IsNullOpLowering>(typeConverter, ctxt);
   patterns.insert<AsNullableOpLowering>(typeConverter, ctxt);
   patterns.insert<NullableGetValOpLowering>(typeConverter, ctxt);

   patterns.insert<ConstantLowering>(typeConverter, ctxt);
   patterns.insert<CastOpLowering>(typeConverter, ctxt);
   patterns.insert<CastNoneOpLowering>(typeConverter, ctxt);

   patterns.insert<HashLowering>(typeConverter, ctxt);

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass>
db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
void db::createLowerDBPipeline(mlir::OpPassManager& pm) {
   pm.addPass(createEliminateNullsPass());
   pm.addPass(createOptimizeRuntimeFunctionsPass());
   pm.addPass(createLowerToStdPass());
}
void db::registerDBConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return createOptimizeRuntimeFunctionsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return createEliminateNullsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return createLowerToStdPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-db",
      "",
      createLowerDBPipeline);
}