#include "mlir-support/parsing.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "runtime-defs/StringRuntime.h"
#include <mlir/Dialect/util/FunctionHelper.h>

using namespace mlir;

namespace {
struct DBToStdLoweringPass
   : public PassWrapper<DBToStdLoweringPass, OperationPass<ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "to-arrow-std"; }

   DBToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, mlir::db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithmeticDialect>();
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
} // end anonymous namespace
static bool hasDBType(TypeConverter& converter, TypeRange types) {
   return llvm::any_of(types, [&converter](mlir::Type t) { auto converted = converter.convertType(t);return converted&&converted!=t; });
}

template <class Op>
class SimpleTypeConversionPattern : public ConversionPattern {
   mlir::LogicalResult safelyMoveRegion(ConversionPatternRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Region& source, mlir::Region& target) const {
      rewriter.inlineRegionBefore(source, target, target.end());
      {
         if (!target.empty()) {
            source.push_back(new Block);
            std::vector<mlir::Location> locs;
            for (size_t i = 0; i < target.front().getArgumentTypes().size(); i++) {
               locs.push_back(target.front().getArgument(i).getLoc());
            }
            source.front().addArguments(target.front().getArgumentTypes(), locs);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&source.front());
            rewriter.create<mlir::dsa::YieldOp>(rewriter.getUnknownLoc());
         }
      }
      if (failed(rewriter.convertRegionTypes(&target, typeConverter))) {
         return rewriter.notifyMatchFailure(source.getParentOp(), "could not convert body types");
      }
      return success();
   }

   public:
   explicit SimpleTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, Op::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type> convertedTypes;
      assert(typeConverter->convertTypes(op->getResultTypes(), convertedTypes).succeeded());
      auto newOp = rewriter.create<Op>(op->getLoc(), convertedTypes, ValueRange(operands), op->getAttrs());
      for (size_t i = 0; i < op->getNumRegions(); i++) {
         if (safelyMoveRegion(rewriter, *typeConverter, op->getRegion(i), newOp->getRegion(i)).failed()) {
            return failure();
         }
      }
      rewriter.replaceOp(op, newOp->getResults());
      return success();
   }
};
class AtLowering : public OpConversionPattern<mlir::dsa::At> {
   public:
   using OpConversionPattern<mlir::dsa::At>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::At atOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = atOp->getLoc();
      auto t = atOp.getType(0);
      if (typeConverter->isLegal(t)) {
         rewriter.startRootUpdate(atOp);
         atOp->setOperands(adaptor.getOperands());
         rewriter.finalizeRootUpdate(atOp);
         return mlir::success();
      }
      auto* context = getContext();
      mlir::Type arrowPhysicalType = typeConverter->convertType(t);
      if (t.isa<mlir::db::DecimalType>()) {
         arrowPhysicalType = mlir::IntegerType::get(context, 128);
      } else if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
         arrowPhysicalType = dateType.getUnit() == mlir::db::DateUnitAttr::day ? mlir::IntegerType::get(context, 32) : mlir::IntegerType::get(context, 64);
      }
      llvm::SmallVector<mlir::Type> types;
      types.push_back(arrowPhysicalType);
      if (atOp.valid()) {
         types.push_back(rewriter.getI1Type());
      }
      std::vector<mlir::Value> values;
      auto newAtOp = rewriter.create<mlir::dsa::At>(loc, types, adaptor.collection(), atOp.pos());
      values.push_back(newAtOp.val());
      if (atOp.valid()) {
         values.push_back(newAtOp.valid());
      }
      if (t.isa<mlir::db::DateType, mlir::db::TimestampType>()) {
         if (values[0].getType() != rewriter.getI64Type()) {
            values[0] = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI64Type(), values[0]);
         }
         size_t multiplier = 1;
         if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
            multiplier = dateType.getUnit() == mlir::db::DateUnitAttr::day ? 86400000000000 : 1000000;
         } else if (auto timeStampType = t.dyn_cast_or_null<mlir::db::TimestampType>()) {
            switch (timeStampType.getUnit()) {
               case mlir::db::TimeUnitAttr::second: multiplier = 1000000000; break;
               case mlir::db::TimeUnitAttr::millisecond: multiplier = 1000000; break;
               case mlir::db::TimeUnitAttr::microsecond: multiplier = 1000; break;
               default: multiplier = 1;
            }
         }
         if (multiplier != 1) {
            mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, multiplier, 64);
            values[0] = rewriter.create<mlir::arith::MulIOp>(loc, values[0], multiplierConst);
         }
      } else if (auto decimalType = t.dyn_cast_or_null<db::DecimalType>()) {
         if (typeConverter->convertType(decimalType).cast<mlir::IntegerType>().getWidth() != 128) {
            values[0] = rewriter.create<arith::TruncIOp>(loc, typeConverter->convertType(decimalType), values[0]);
         }
      }
      rewriter.replaceOp(atOp, values);
      return success();
   }
};
class AppendTBLowering : public ConversionPattern {
   public:
   explicit AppendTBLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::Append::getOperationName(), 2, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      mlir::dsa::AppendAdaptor adaptor(operands);
      auto appendOp = mlir::cast<mlir::dsa::Append>(op);
      if (!appendOp.ds().getType().isa<mlir::dsa::TableBuilderType>()) {
         return mlir::failure();
      }
      auto t = appendOp.val().getType();
      if (typeConverter->isLegal(t)) {
         rewriter.startRootUpdate(op);
         appendOp->setOperands(operands);
         rewriter.finalizeRootUpdate(op);
         return mlir::success();
      }
      auto* context = getContext();
      mlir::Type arrowPhysicalType = typeConverter->convertType(t);
      if (t.isa<mlir::db::DecimalType>()) {
         arrowPhysicalType = mlir::IntegerType::get(context, 128);
      } else if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
         arrowPhysicalType = dateType.getUnit() == mlir::db::DateUnitAttr::day ? mlir::IntegerType::get(context, 32) : mlir::IntegerType::get(context, 64);
      }

      mlir::Value val = adaptor.val();
      if (t.isa<mlir::db::DateType, mlir::db::TimestampType>()) {
         size_t multiplier = 1;
         if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
            multiplier = dateType.getUnit() == mlir::db::DateUnitAttr::day ? 86400000000000 : 1000000;
         } else if (auto timeStampType = t.dyn_cast_or_null<mlir::db::TimestampType>()) {
            switch (timeStampType.getUnit()) {
               case mlir::db::TimeUnitAttr::second: multiplier = 1000000000; break;
               case mlir::db::TimeUnitAttr::millisecond: multiplier = 1000000; break;
               case mlir::db::TimeUnitAttr::microsecond: multiplier = 1000; break;
               default: multiplier = 1;
            }
         }
         if (multiplier != 1) {
            mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, multiplier, 64);
            val = rewriter.create<mlir::arith::DivSIOp>(loc, val, multiplierConst);
         }
         if (arrowPhysicalType != rewriter.getI64Type()) {
            val = rewriter.create<mlir::arith::TruncIOp>(loc, arrowPhysicalType, val);
         }
      } else if (auto decimalType = t.dyn_cast_or_null<db::DecimalType>()) {
         if (typeConverter->convertType(decimalType).cast<mlir::IntegerType>().getWidth() != 128) {
            val = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(128), val);
         }
      }
      rewriter.create<mlir::dsa::Append>(loc, adaptor.ds(), val, adaptor.valid());

      rewriter.eraseOp(op);
      return success();
   }
};
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
            result = rt::StringRuntime::toInt(rewriter, loc)({valueToCast})[0];
            if (intWidth < 64) {
               result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, result);
            }
         } else if (auto floatType = scalarTargetType.dyn_cast_or_null<FloatType>()) {
            result = floatType.getWidth() == 32 ? rt::StringRuntime::toFloat32(rewriter, loc)({valueToCast})[0] : rt::StringRuntime::toFloat64(rewriter, loc)({valueToCast})[0];
         } else if (auto decimalType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
            auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalType.getS()));
            result = rt::StringRuntime::toDecimal(rewriter, loc)({valueToCast, scale})[0];
            if (typeConverter->convertType(decimalType).cast<mlir::IntegerType>().getWidth() < 128) {
               auto converted = rewriter.create<arith::TruncIOp>(loc, typeConverter->convertType(decimalType), result);
               result = converted;
            }
         }
      } else if (auto intWidth = getIntegerWidth(scalarSourceType, false)) {
         result = rt::StringRuntime::fromInt(rewriter, loc)({valueToCast})[0];
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<FloatType>()) {
         result = floatType.getWidth() == 32 ? rt::StringRuntime::fromFloat32(rewriter, loc)({valueToCast})[0] : rt::StringRuntime::fromFloat64(rewriter, loc)({valueToCast})[0];
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
         auto scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(decimalSourceType.getS()));
         result = rt::StringRuntime::fromDecimal(rewriter, loc)({valueToCast, scale})[0];
      } else if (auto charType = scalarSourceType.dyn_cast_or_null<db::CharType>()) {
         auto bytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(charType.getBytes()));
         result = rt::StringRuntime::fromChar(rewriter, loc)({valueToCast, bytes})[0];
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
      Value res;
      Value left = adaptor.left();
      Value right = adaptor.right();
      switch (cmpOp.predicate()) {
         case db::DBCmpPredicate::eq:
            res = rt::StringRuntime::compareEq(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::neq:
            res = rt::StringRuntime::compareNEq(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lt:
            res = rt::StringRuntime::compareLt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gt:
            res = rt::StringRuntime::compareGt(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::lte:
            res = rt::StringRuntime::compareLte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
         case db::DBCmpPredicate::gte:
            res = rt::StringRuntime::compareGte(rewriter, cmpOp->getLoc())({left, right})[0];
            break;
      }
      rewriter.replaceOp(cmpOp, res);
      return success();
   }
};

class RuntimeCallLowering : public OpConversionPattern<mlir::db::RuntimeCall> {
   public:
   using OpConversionPattern<mlir::db::RuntimeCall>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::RuntimeCall runtimeCallOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
      auto* fn = reg->lookup(runtimeCallOp.fn().str());
      if (!fn) return failure();
      Value result;
      mlir::Type resType = runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type();
      if (std::holds_alternative<mlir::util::FunctionSpec>(fn->implementation)) {
         auto& implFn = std::get<mlir::util::FunctionSpec>(fn->implementation);
         auto resRange = implFn(rewriter, rewriter.getUnknownLoc())(adaptor.args());
         assert((resRange.size() == 1 && resType) || (resRange.empty() && !resType));
         result = resRange.size() == 1 ? resRange[0] : mlir::Value();
      } else if (std::holds_alternative<mlir::db::RuntimeFunction::loweringFnT>(fn->implementation)) {
         auto& implFn = std::get<mlir::db::RuntimeFunction::loweringFnT>(fn->implementation);
         result = implFn(rewriter, adaptor.args(), runtimeCallOp.args().getTypes(), runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type(), typeConverter, runtimeCallOp->getLoc());
      }

      if (runtimeCallOp->getNumResults() == 0) {
         rewriter.eraseOp(runtimeCallOp);
      } else {
         rewriter.replaceOp(runtimeCallOp, result);
      }
      return success();
   }
};

class NotOpLowering : public OpConversionPattern<mlir::db::NotOp> {
   public:
   using OpConversionPattern<mlir::db::NotOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::NotOp notOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value falseValue = rewriter.create<arith::ConstantOp>(notOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(notOp, mlir::arith::CmpIPredicate::eq, adaptor.val(), falseValue);
      return success();
   }
};
class AndOpLowering : public OpConversionPattern<mlir::db::AndOp> {
   public:
   using OpConversionPattern<mlir::db::AndOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::AndOp andOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value result;
      Value isNull;
      auto loc = andOp->getLoc();

      for (size_t i = 0; i < adaptor.vals().size(); i++) {
         auto currType = andOp.vals()[i].getType();
         bool currNullable = currType.isa<mlir::db::NullableType>();
         Value currNull;
         Value currVal;
         if (currNullable) {
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.vals()[i]);
            currNull = unPackOp.vals()[0];
            currVal = unPackOp.vals()[1];
         } else {
            currVal = adaptor.vals()[i];
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
         rewriter.replaceOp(andOp, combined);
      } else {
         rewriter.replaceOp(andOp, result);
      }
      return success();
   }
};
class OrOpLowering : public OpConversionPattern<mlir::db::OrOp> {
   public:
   using OpConversionPattern<mlir::db::OrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::OrOp orOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value result;
      Value isNull;
      auto loc = orOp->getLoc();
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      for (size_t i = 0; i < adaptor.vals().size(); i++) {
         auto currType = orOp.vals()[i].getType();
         bool currNullable = currType.isa<mlir::db::NullableType>();
         Value currNull;
         Value currVal;
         if (currNullable) {
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.vals()[i]);
            currNull = unPackOp.vals()[0];
            currVal = unPackOp.vals()[1];
         } else {
            currVal = adaptor.vals()[i];
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
      auto type = getBaseType(binOp.left().getType());
      if (type.template isa<OperandType>()) {
         rewriter.template replaceOpWithNewOp<StdOpClass>(binOp, this->typeConverter->convertType(binOp.result().getType()), adaptor.left(), adaptor.right());
         return success();
      }
      return failure();
   }
};
mlir::Value getDecimalScaleMultiplierConstant(mlir::OpBuilder& builder, int32_t s, mlir::Type stdType, mlir::Location loc) {
   auto [low, high] = support::getDecimalScaleMultiplier(s);
   std::vector<uint64_t> parts = {low, high};
   auto multiplier = builder.create<arith::ConstantOp>(loc, stdType, builder.getIntegerAttr(stdType, APInt(stdType.template cast<mlir::IntegerType>().getWidth(), parts)));
   return multiplier;
}
template <class DBOp, class Op>
class DecimalOpScaledLowering : public OpConversionPattern<DBOp> {
   public:
   using OpConversionPattern<DBOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(DBOp decimalOp, typename OpConversionPattern<DBOp>::OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = getBaseType(decimalOp.getType());
      if (auto decimalType = type.template dyn_cast_or_null<mlir::db::DecimalType>()) {
         auto stdType = this->typeConverter->convertType(decimalType);
         auto scaled = rewriter.create<arith::MulIOp>(decimalOp->getLoc(), stdType, adaptor.left(), getDecimalScaleMultiplierConstant(rewriter, decimalType.getS(), stdType, decimalOp->getLoc()));
         rewriter.template replaceOpWithNewOp<Op>(decimalOp, stdType, scaled, adaptor.right());
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
      if (auto decimalType = mulOp.getType().template dyn_cast_or_null<mlir::db::DecimalType>()) {
         auto stdType = this->typeConverter->convertType(decimalType);
         mlir::Value left = adaptor.left();
         mlir::Value right = adaptor.right();
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
class IsNullOpLowering : public OpConversionPattern<mlir::db::IsNullOp> {
   public:
   using OpConversionPattern<mlir::db::IsNullOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::IsNullOp isNullOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (isNullOp.val().getType().isa<mlir::db::NullableType>()) {
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(isNullOp->getLoc(), adaptor.val());
         rewriter.replaceOp(isNullOp, unPackOp.vals()[0]);
      } else {
         rewriter.replaceOp(isNullOp, adaptor.val());
      }
      return success();
   }
};
class NullableGetValOpLowering : public OpConversionPattern<mlir::db::NullableGetVal> {
   public:
   using OpConversionPattern<mlir::db::NullableGetVal>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::NullableGetVal op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto unPackOp = rewriter.create<mlir::util::UnPackOp>(op->getLoc(), adaptor.val());
      rewriter.replaceOp(op, unPackOp.vals()[1]);
      return success();
   }
};
class AsNullableOpLowering : public OpConversionPattern<mlir::db::AsNullableOp> {
   public:
   using OpConversionPattern<mlir::db::AsNullableOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::AsNullableOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Value isNull = adaptor.null();
      if (!isNull) {
         isNull = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      }
      auto packOp = rewriter.create<mlir::util::PackOp>(op->getLoc(), ValueRange({isNull, adaptor.val()}));
      rewriter.replaceOp(op, packOp.tuple());
      return success();
   }
};
class NullOpLowering : public OpConversionPattern<mlir::db::NullOp> {
   public:
   using OpConversionPattern<mlir::db::NullOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::NullOp nullOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto tupleType = typeConverter->convertType(nullOp.getType()).cast<mlir::TupleType>();
      auto undefValue = rewriter.create<mlir::util::UndefOp>(nullOp->getLoc(), tupleType.getType(1));
      auto trueValue = rewriter.create<arith::ConstantOp>(nullOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      rewriter.replaceOpWithNewOp<mlir::util::PackOp>(nullOp, tupleType, ValueRange({trueValue, undefValue}));
      return success();
   }
};

class ConstantLowering : public OpConversionPattern<mlir::db::ConstantOp> {
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
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         typeConstant = arrow::Type::type::DECIMAL128;
         param1 = decimalType.getP();
         param2 = decimalType.getS();
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 16: typeConstant = arrow::Type::type::HALF_FLOAT; break;
            case 32: typeConstant = arrow::Type::type::FLOAT; break;
            case 64: typeConstant = arrow::Type::type::DOUBLE; break;
         }
      } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
         typeConstant = arrow::Type::type::STRING;
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
            typeConstant = arrow::Type::type::DATE32;
         } else {
            typeConstant = arrow::Type::type::DATE64;
         }
      } else if (auto charType = type.dyn_cast_or_null<mlir::db::CharType>()) {
         typeConstant = arrow::Type::type::FIXED_SIZE_BINARY;
         param1 = charType.getBytes();
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            typeConstant = arrow::Type::type::INTERVAL_MONTHS;
         } else {
            typeConstant = arrow::Type::type::INTERVAL_DAY_TIME;
         }
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         typeConstant = arrow::Type::type::TIMESTAMP;
         param1 = static_cast<uint32_t>(timestampType.getUnit());
      }
      assert(typeConstant != arrow::Type::type::NA);
      return {typeConstant, param1, param2};
   }

   public:
   using OpConversionPattern<mlir::db::ConstantOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::ConstantOp constantOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = constantOp.getType();
      auto stdType = typeConverter->convertType(type);
      auto [arrowType, param1, param2] = convertTypeToArrow(type);
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
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType, rewriter.getIntegerAttr(stdType, APInt(stdType.cast<mlir::IntegerType>().getWidth(), parts)));
            return success();
         } else {
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType, rewriter.getIntegerAttr(stdType, std::get<int64_t>(parseResult)));
            return success();
         }
      } else if (auto floatType = stdType.dyn_cast_or_null<FloatType>()) {
         rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType, rewriter.getFloatAttr(stdType, std::get<double>(parseResult)));
         return success();
      } else if (type.isa<mlir::db::StringType>()) {
         std::string str = std::get<std::string>(parseResult);

         rewriter.replaceOpWithNewOp<mlir::util::CreateConstVarLen>(constantOp, mlir::util::VarLen32Type::get(rewriter.getContext()), rewriter.getStringAttr(str));
         return success();
      } else {
         return failure();
      }
      return failure();
   }
};
class CmpOpLowering : public OpConversionPattern<mlir::db::CmpOp> {
   public:
   using OpConversionPattern<mlir::db::CmpOp>::OpConversionPattern;
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
      }
      assert(false && "unexpected case");
      return arith::CmpFPredicate::OEQ;
   }
   LogicalResult matchAndRewrite(mlir::db::CmpOp cmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!adaptor.left().getType().isIntOrIndexOrFloat()) {
         return failure();
      }
      if (adaptor.left().getType().isIntOrIndex()) {
         rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, translateIPredicate(cmpOp.predicate()), adaptor.left(), adaptor.right());
      } else {
         rewriter.replaceOpWithNewOp<arith::CmpFOp>(cmpOp, translateFPredicate(cmpOp.predicate()), adaptor.left(), adaptor.right());
      }
      return success();
   }
};
class CastNoneOpLowering : public OpConversionPattern<mlir::db::CastOp> {
   public:
   using OpConversionPattern<mlir::db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::CastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto scalarSourceType = op.val().getType();
      auto scalarTargetType = op.getType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (scalarSourceType.isa<mlir::db::StringType>() || scalarTargetType.isa<mlir::db::StringType>()) return failure();
      if (!convertedSourceType.isa<NoneType>()) {
         return mlir::failure();
      }
      rewriter.replaceOpWithNewOp<mlir::util::UndefOp>(op, convertedTargetType);
      return mlir::success();
   }
};
class CastOpLowering : public OpConversionPattern<mlir::db::CastOp> {
   public:
   using OpConversionPattern<mlir::db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::CastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto scalarSourceType = op.val().getType();
      auto scalarTargetType = op.getType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (scalarSourceType.isa<mlir::db::StringType>() || scalarTargetType.isa<mlir::db::StringType>()) return failure();
      Value value = adaptor.val();
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
            size_t decimalWidth = convertedSourceType.cast<mlir::IntegerType>().getWidth();
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
class BetweenLowering : public OpConversionPattern<mlir::db::BetweenOp> {
   public:
   using OpConversionPattern<mlir::db::BetweenOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::BetweenOp betweenOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto isGteLower = rewriter.create<mlir::db::CmpOp>(betweenOp->getLoc(), betweenOp.lowerInclusive() ? mlir::db::DBCmpPredicate::gte : mlir::db::DBCmpPredicate::gt, betweenOp.val(), betweenOp.lower());
      auto isLteUpper = rewriter.create<mlir::db::CmpOp>(betweenOp->getLoc(), betweenOp.upperInclusive() ? mlir::db::DBCmpPredicate::lte : mlir::db::DBCmpPredicate::lt, betweenOp.val(), betweenOp.upper());
      auto isInRange = rewriter.create<mlir::db::AndOp>(betweenOp->getLoc(), ValueRange({isGteLower, isLteUpper}));
      rewriter.replaceOp(betweenOp, isInRange.res());
      return success();
   }
};
class OneOfLowering : public OpConversionPattern<mlir::db::OneOfOp> {
   public:
   using OpConversionPattern<mlir::db::OneOfOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::OneOfOp oneOfOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Value> compared;
      for (auto ele : oneOfOp.vals()) {
         compared.push_back(rewriter.create<mlir::db::CmpOp>(oneOfOp->getLoc(), mlir::db::DBCmpPredicate::eq, oneOfOp.val(), ele));
      }
      auto isInRange = rewriter.create<mlir::db::OrOp>(oneOfOp->getLoc(), compared);
      rewriter.replaceOp(oneOfOp, isInRange.res());
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
void DBToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper().setParentModule(module);

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   TypeConverter typeConverter;
   auto *ctxt = &getContext();
   typeConverter.addConversion([&](mlir::Type type) { return type; });
   typeConverter.addConversion([&](::mlir::db::DateType t) {
      return mlir::IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](::mlir::db::DecimalType t) {
      if (t.getP() < 19) {
         return mlir::IntegerType::get(ctxt, 64);

      } else {
         return mlir::IntegerType::get(ctxt, 128);
      }
   });
   typeConverter.addConversion([&](::mlir::db::CharType t) {
      if (t.getBytes() > 8) return mlir::Type();
      return (Type) mlir::IntegerType::get(ctxt, t.getBytes() * 8);
   });
   typeConverter.addConversion([&](::mlir::db::StringType t) {
      return mlir::util::VarLen32Type::get(ctxt);
   });
   typeConverter.addConversion([&](::mlir::db::TimestampType t) {
      return mlir::IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](::mlir::db::IntervalType t) {
      if (t.getUnit() == mlir::db::IntervalUnitAttr::daytime) {
         return mlir::IntegerType::get(ctxt, 64);
      } else {
         return mlir::IntegerType::get(ctxt, 32);
      }
   });

   typeConverter.addConversion([&](mlir::db::NullableType type) {
      mlir::Type payloadType = typeConverter.convertType(type.getType());
      if (payloadType.isa<mlir::NoneType>()) {
         payloadType = IntegerType::get(ctxt, 1);
      }
      return (Type) TupleType::get(ctxt, {IntegerType::get(ctxt, 1), payloadType});
   });
   auto opIsWithoutDBTypes = [&](Operation* op) { return !hasDBType(typeConverter, op->getOperandTypes()) && !hasDBType(typeConverter, op->getResultTypes()); };
   target.addDynamicallyLegalDialect<scf::SCFDialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<dsa::DSADialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<arith::ArithmeticDialect>(opIsWithoutDBTypes);

   target.addLegalDialect<cf::ControlFlowDialect>();

   target.addDynamicallyLegalDialect<util::UtilDialect>(opIsWithoutDBTypes);
   target.addLegalOp<mlir::dsa::CondSkipOp>();

   target.addDynamicallyLegalOp<mlir::dsa::CondSkipOp>(opIsWithoutDBTypes);
   target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isLegal = !hasDBType(typeConverter, op.getFunctionType().getInputs()) &&
         !hasDBType(typeConverter, op.getFunctionType().getResults());
      return isLegal;
   });
   target.addDynamicallyLegalOp<mlir::func::ConstantOp>([&](mlir::func::ConstantOp op) {
      if (auto functionType = op.getType().dyn_cast_or_null<mlir::FunctionType>()) {
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
         auto isLegal = !hasDBType(typeConverter, op.type());
         return isLegal;
      });

   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });

   auto convertPhysical = [&](mlir::TupleType tuple) -> mlir::TupleType {
      std::vector<mlir::Type> types;
      for (auto t : tuple.getTypes()) {
         mlir::Type arrowPhysicalType = typeConverter.convertType(t);
         if (t.isa<mlir::db::DecimalType>()) {
            arrowPhysicalType = mlir::IntegerType::get(t.getContext(), 128);
         } else if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
            arrowPhysicalType = dateType.getUnit() == mlir::db::DateUnitAttr::day ? mlir::IntegerType::get(t.getContext(), 32) : mlir::IntegerType::get(t.getContext(), 64);
         }
         types.push_back(arrowPhysicalType);
      }
      return mlir::TupleType::get(tuple.getContext(), types);
   };
   typeConverter.addConversion([&](mlir::dsa::RecordType r) {
      return mlir::dsa::RecordType::get(r.getContext(), convertPhysical(r.getRowType()));
   });
   typeConverter.addConversion([&](mlir::dsa::RecordBatchType r) {
      return mlir::dsa::RecordBatchType::get(r.getContext(), convertPhysical(r.getRowType()));
   });
   typeConverter.addConversion([&](mlir::dsa::GenericIterableType r) { return mlir::dsa::GenericIterableType::get(r.getContext(), typeConverter.convertType(r.getElementType()), r.getIteratorName()); });
   typeConverter.addConversion([&](mlir::dsa::VectorType r) { return mlir::dsa::VectorType::get(r.getContext(), typeConverter.convertType(r.getElementType())); });
   typeConverter.addConversion([&](mlir::dsa::JoinHashtableType r) { return mlir::dsa::JoinHashtableType::get(r.getContext(), typeConverter.convertType(r.getKeyType()).cast<mlir::TupleType>(), typeConverter.convertType(r.getValType()).cast<mlir::TupleType>()); });
   typeConverter.addConversion([&](mlir::dsa::AggregationHashtableType r) { return mlir::dsa::AggregationHashtableType::get(r.getContext(), typeConverter.convertType(r.getKeyType()).cast<mlir::TupleType>(), typeConverter.convertType(r.getValType()).cast<mlir::TupleType>()); });
   typeConverter.addConversion([&](mlir::dsa::TableBuilderType r) { return mlir::dsa::TableBuilderType::get(r.getContext(), typeConverter.convertType(r.getRowType()).cast<mlir::TupleType>()); });

   RewritePatternSet patterns(&getContext());

   mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
   patterns.insert<SimpleTypeConversionPattern<mlir::func::ConstantOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::CondSkipOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::ScanSource>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Append>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::CreateDS>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Finalize>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Lookup>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::FreeOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::YieldOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::NextRow>>(typeConverter, &getContext());
   patterns.insert<AtLowering>(typeConverter, &getContext());
   patterns.insert<AppendTBLowering>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::HashtableInsert>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::SortOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::ForOp>>(typeConverter, &getContext());
   patterns.insert<StringCmpOpLowering>(typeConverter, ctxt);
   patterns.insert<StringCastOpLowering>(typeConverter, ctxt);
   patterns.insert<RuntimeCallLowering>(typeConverter, ctxt);
   patterns.insert<CmpOpLowering>(typeConverter, ctxt);
   patterns.insert<BetweenLowering>(typeConverter, ctxt);
   patterns.insert<OneOfLowering>(typeConverter, ctxt);

   patterns.insert<NotOpLowering>(typeConverter, ctxt);

   patterns.insert<AndOpLowering>(typeConverter, ctxt);
   patterns.insert<OrOpLowering>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::IntegerType, arith::AddIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::IntegerType, arith::SubIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::IntegerType, arith::MulIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::IntegerType, arith::DivSIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::IntegerType, arith::RemSIOp>>(typeConverter, ctxt);

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::FloatType, arith::AddFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::FloatType, arith::SubFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::FloatType, arith::MulFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::FloatType, arith::DivFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::FloatType, arith::RemFOp>>(typeConverter, ctxt);

   patterns.insert<DecimalBinOpLowering<mlir::db::AddOp, arith::AddIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalBinOpLowering<mlir::db::SubOp, arith::SubIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalBinOpLowering<mlir::db::MulOp, arith::MulIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalOpScaledLowering<mlir::db::DivOp, arith::DivSIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalOpScaledLowering<mlir::db::ModOp, arith::RemSIOp>>(typeConverter, ctxt);

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
mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
void mlir::db::createLowerDBPipeline(mlir::OpPassManager& pm) {
   pm.addPass(mlir::db::createEliminateNullsPass());
   pm.addPass(mlir::db::createOptimizeRuntimeFunctionsPass());
   pm.addPass(mlir::db::createLowerToStdPass());
}
void mlir::db::registerDBConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::db::createOptimizeRuntimeFunctionsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::db::createEliminateNullsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::db::createLowerToStdPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-db",
      "",
      createLowerDBPipeline);
}