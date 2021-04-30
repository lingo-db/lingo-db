#include "arrow/util/decimal.h"
#include "arrow/util/value_parsing.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "runtime/runtime.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {

//declare external function or return reference to already existing one
static FuncOp getOrInsertFn(PatternRewriter& rewriter,
                            ModuleOp module, const std::string& name, FunctionType fnType) {
   if (FuncOp funcOp = module.lookupSymbol<FuncOp>(name))
      return funcOp;
   PatternRewriter::InsertionGuard insertGuard(rewriter);
   rewriter.setInsertionPointToStart(module.getBody());
   FuncOp funcOp = rewriter.create<FuncOp>(module.getLoc(), name, fnType, rewriter.getStringAttr("private"));
   return funcOp;
}
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
class IsNullOpLowering : public ConversionPattern {
   public:
   explicit IsNullOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::IsNullOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto nullOp = cast<mlir::db::IsNullOp>(op);
      auto tupleType = typeConverter->convertType(nullOp.val().getType()).dyn_cast_or_null<TupleType>();
      auto splitOp = rewriter.create<mlir::util::SplitOp>(rewriter.getUnknownLoc(), tupleType.getTypes(), nullOp.val());
      rewriter.replaceOp(op, splitOp.vals()[0]);
      return success();
   }
};

//Lower Print Operation to an actual printf call
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
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto loc = op->getLoc();
      if (constantOp.getType().isa<mlir::db::IntType>() || constantOp.getType().isa<mlir::db::BoolType>()) {
         if (!constantOp.value().isa<IntegerAttr>()) {
            return failure();
         }
         auto integerVal = constantOp.value().dyn_cast_or_null<IntegerAttr>().getInt();
         rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
         return success();
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         if (auto strAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            std::cout << strAttr.getValue().str() << std::endl;
            int32_t precision;
            int32_t scale;
            arrow::Decimal128 decimalrep;
            if (arrow::Decimal128::FromString(strAttr.getValue().str(), &decimalrep, &precision, &scale) != arrow::Status::OK()) {
               return failure();
            }
            auto x = decimalrep.Rescale(scale, decimalType.getS());
            decimalrep = x.ValueUnsafe();
            std::vector<uint64_t> parts = {decimalrep.low_bits(), (uint64_t) decimalrep.high_bits()};
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, APInt(128, parts)));

            return success();
         }
      } else if (type.isa<mlir::db::DateType>()) {
         if (auto strAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            int32_t integerVal;
            arrow::internal::ParseValue<arrow::Date32Type>(strAttr.getValue().data(), strAttr.getValue().str().length(), &integerVal);
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
            return success();
         }
      } else if (type.isa<mlir::db::TimestampType>()) {
         if (auto strAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            int64_t integerVal;
            arrow::internal::ParseValue<arrow::TimestampType>(arrow::TimestampType(), strAttr.getValue().data(), strAttr.getValue().str().length(), &integerVal);
            integerVal /= 1000;
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, integerVal));
            return success();
         }
      } else if (type.isa<mlir::db::IntervalType>()) {
         if (auto intAttr = constantOp.value().dyn_cast_or_null<IntegerAttr>()) {
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getIntegerAttr(stdType, intAttr.getValue()));
            return success();
         }
      } else if (type.isa<mlir::db::FloatType>()) {
         if (auto floatAttr = constantOp.value().dyn_cast_or_null<FloatAttr>()) {
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, rewriter.getFloatAttr(stdType, floatAttr.getValueAsDouble()));
            return success();
         }
      } else if (type.isa<mlir::db::StringType>()) {
         if (auto stringAttr = constantOp.value().dyn_cast_or_null<StringAttr>()) {
            auto insertionPoint = rewriter.saveInsertionPoint();
            int64_t strLen = stringAttr.getValue().size();
            std::vector<uint8_t> vec;
            for (auto c : stringAttr.getValue()) {
               vec.push_back(c);
            }
            auto strStaticType = MemRefType::get({strLen}, i8Type);
            auto strDynamicType = MemRefType::get({-1}, IntegerType::get(getContext(), 8));
            rewriter.setInsertionPointToStart(op->getParentOfType<ModuleOp>().getBody());
            auto initialValue = DenseIntElementsAttr::get(
               RankedTensorType::get({strLen}, i8Type), vec);
            static int id = 0;
            auto globalop = rewriter.create<mlir::memref::GlobalOp>(rewriter.getUnknownLoc(), "db_constant_string" + std::to_string(id++), rewriter.getStringAttr("private"), strStaticType, initialValue, true);
            rewriter.restoreInsertionPoint(insertionPoint);
            Value conststr = rewriter.create<mlir::memref::GetGlobalOp>(loc, strStaticType, globalop.sym_name());
            rewriter.replaceOpWithNewOp<memref::CastOp>(op, conststr, strDynamicType);
            return success();
         }
      }

      return failure();
   }
};
class NullHandler {
   std::vector<Value> nullValues;
   TypeConverter& typeConverter;
   OpBuilder& builder;

   public:
   NullHandler(TypeConverter& typeConverter, OpBuilder& builder) : typeConverter(typeConverter), builder(builder) {}
   Value getValue(Value v, Value operand = Value()) {
      Type type = v.getType();
      if (auto dbType = type.dyn_cast_or_null<mlir::db::DBType>()) {
         if (dbType.isNullable()) {
            TupleType tupleType = typeConverter.convertType(v.getType()).dyn_cast_or_null<TupleType>();
            auto splitOp = builder.create<mlir::util::SplitOp>(builder.getUnknownLoc(), tupleType.getTypes(), v);
            nullValues.push_back(splitOp.vals()[0]);
            return splitOp.vals()[1];
         } else {
            return operand ? operand : v;
         }
      } else {
         return operand ? operand : v;
      }
   }
   Value combineResult(Value res) {
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      if (nullValues.empty()) {
         return res;
      }
      Value isNull;
      if (nullValues.size() >= 1) {
         isNull = nullValues.front();
      }
      for (size_t i = 1; i < nullValues.size(); i++) {
         isNull = builder.create<mlir::OrOp>(builder.getUnknownLoc(), isNull.getType(), isNull, nullValues[i]);
      }
      return builder.create<mlir::util::CombineOp>(builder.getUnknownLoc(), mlir::TupleType::get(builder.getContext(), {i1Type, res.getType()}), ValueRange({isNull, res}));
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
      NullHandler nullHandler(*typeConverter, rewriter);
      auto type = addOp.lhs().getType();
      Type resType = addOp.result().getType().template cast<db::DBType>().getBaseType();
      Value left = nullHandler.getValue(addOp.lhs());
      Value right = nullHandler.getValue(addOp.rhs());
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
      Value left = addOp.lhs();
      Value right = addOp.rhs();
      if (left.getType() != right.getType()) {
         return failure();
      }
      auto type = left.getType();
      if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         auto decimalrep = arrow::Decimal128::GetScaleMultiplier(decimalType.getS());
         std::vector<uint64_t> parts = {decimalrep.low_bits(), (uint64_t) decimalrep.high_bits()};
         auto stdType = typeConverter->convertType(type);
         auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), stdType, rewriter.getIntegerAttr(stdType, APInt(128, parts)));
         left = rewriter.create<mlir::MulIOp>(rewriter.getUnknownLoc(), stdType, left, multiplier);
         rewriter.replaceOpWithNewOp<Op>(op, stdType, left, right);
         return success();
      }
      return failure();
   }
};
//lower dbexec::If to scf::If
class IfLowering : public ConversionPattern {
   public:
   explicit IfLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::IfOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto ifOp = cast<mlir::db::IfOp>(op);
      auto loc = op->getLoc();
      std::vector<Type> resultTypes;
      for (auto res : ifOp.results()) {
         resultTypes.push_back(typeConverter->convertType(res.getType()));
      }
      auto newIfOp = rewriter.create<mlir::scf::IfOp>(loc, TypeRange(resultTypes), ifOp.condition(), !ifOp.elseRegion().empty());
      {
         scf::IfOp::ensureTerminator(newIfOp.thenRegion(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newIfOp.thenRegion().front());
         Block* originalThenBlock = &ifOp.thenRegion().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalThenBlock, terminator, {});
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      if (!ifOp.elseRegion().empty()) {
         scf::IfOp::ensureTerminator(newIfOp.elseRegion(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newIfOp.elseRegion().front());
         Block* originalElseBlock = &ifOp.elseRegion().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalElseBlock, terminator, {});
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      llvm::dbgs() << newIfOp.results().size() << "\n";
      llvm::dbgs() << ifOp->getNumResults() << "\n";

      rewriter.replaceOp(ifOp, newIfOp.results());

      return success();
   }
};
class YieldLowering : public ConversionPattern {
   public:
   explicit YieldLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::YieldOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
      return success();
   }
};
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
            auto splitOp = rewriter.create<mlir::util::SplitOp>(loc, tupleType.getTypes(), operands[i]);
            currNull = splitOp.vals()[0];
            currVal = splitOp.vals()[1];
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
         Value combined = rewriter.create<mlir::util::CombineOp>(loc, typeConverter->convertType(andOp.getResult().getType()), ValueRange({isNull, result}));
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
            auto splitOp = rewriter.create<mlir::util::SplitOp>(loc, tupleType.getTypes(), operands[i]);
            currNull = splitOp.vals()[0];
            currVal = splitOp.vals()[1];
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
         Value combined = rewriter.create<mlir::util::CombineOp>(loc, typeConverter->convertType(orOp.getResult().getType()), ValueRange({isNull, result}));
         rewriter.replaceOp(op, combined);
      } else {
         rewriter.replaceOp(op, result);
      }
      return success();
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
      NullHandler nullHandler(*typeConverter, rewriter);
      auto cmpOp = cast<db::CmpOp>(op);

      auto type = cmpOp.lhs().getType().cast<db::DBType>().getBaseType();
      Value left = nullHandler.getValue(cmpOp.lhs());
      Value right = nullHandler.getValue(cmpOp.rhs());
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
static time_unit timeUnitFromStr(std::string str) {
   if (str == "days" || str == "day") {
      return time_unit::DAY;
   } else if (str == "months" || str == "month") {
      return time_unit::MONTH;
   } else if (str == "years" || str == "year") {
      return time_unit::YEAR;
   } else
      return time_unit::UNKNOWN;
}
class DateAddLowering : public ConversionPattern {
   public:
   explicit DateAddLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DateAddOp::getOperationName(), 1, context) {}
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      db::DateAddOp::Adaptor dateAddAdaptor(operands);
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);

      auto loc = rewriter.getUnknownLoc();
      NullHandler nullHandler(*typeConverter, rewriter);
      auto dateAddOp = cast<db::DateAddOp>(op);

      auto intervalType = dateAddOp.right().getType().cast<db::IntervalType>();

      Value left = nullHandler.getValue(dateAddOp.left(), dateAddAdaptor.left());
      Value right = nullHandler.getValue(dateAddOp.right(), dateAddAdaptor.right());
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      Value unit = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(i8Type, timeUnitFromStr(intervalType.getUnit())));

      auto printRef = getOrInsertFn(rewriter, parentModule, "dateAdd", rewriter.getFunctionType({i32Type, i32Type, i8Type}, {i32Type}));
      right = rewriter.create<TruncateIOp>(loc, right, i32Type);
      auto call = rewriter.create<CallOp>(loc, printRef, ValueRange({left, right,unit}));
      Value res = call.getResult(0);
      rewriter.replaceOp(op,nullHandler.combineResult(res));
      return success();
   }
};
class DateSubLowering : public ConversionPattern {
   public:
   explicit DateSubLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DateSubOp::getOperationName(), 1, context) {}
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      db::DateSubOp::Adaptor dateSubAdaptor(operands);
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);

      auto loc = rewriter.getUnknownLoc();
      NullHandler nullHandler(*typeConverter, rewriter);
      auto dateAddOp = cast<db::DateSubOp>(op);

      auto intervalType = dateAddOp.right().getType().cast<db::IntervalType>();

      Value left = nullHandler.getValue(dateAddOp.left(), dateSubAdaptor.left());
      Value right = nullHandler.getValue(dateAddOp.right(), dateSubAdaptor.right());
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      Value unit = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(i8Type, timeUnitFromStr(intervalType.getUnit())));

      auto printRef = getOrInsertFn(rewriter, parentModule, "dateSub", rewriter.getFunctionType({i32Type, i32Type, i8Type}, {i32Type}));
      right = rewriter.create<TruncateIOp>(loc, right, i32Type);
      auto call = rewriter.create<CallOp>(loc, printRef, ValueRange({left, right,unit}));
      Value res = call.getResult(0);
      rewriter.replaceOp(op,nullHandler.combineResult(res));
      return success();
   }
};
class DateExtractLowering : public ConversionPattern {
   public:
   explicit DateExtractLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DateExtractOp::getOperationName(), 1, context) {}
   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      db::DateExtractOp::Adaptor dateExtractOpAdaptor(operands);
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);

      auto loc = rewriter.getUnknownLoc();
      NullHandler nullHandler(*typeConverter, rewriter);
      auto dateExtractOp = cast<db::DateExtractOp>(op);

      Value value = nullHandler.getValue(dateExtractOp.date(), dateExtractOpAdaptor.date());
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      Value unit = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(i8Type, timeUnitFromStr(dateExtractOp.unit().str())));

      auto printRef = getOrInsertFn(rewriter, parentModule, "dateExtract", rewriter.getFunctionType({i32Type, i8Type}, {i32Type}));
      auto call = rewriter.create<CallOp>(loc, printRef, ValueRange({value,unit}));
      Value res = call.getResult(0);
      rewriter.replaceOp(op,nullHandler.combineResult(res));
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
      auto loc = rewriter.getUnknownLoc();
      auto sourceType = castOp.val().getType().cast<db::DBType>();
      auto targetType = castOp.getType().cast<db::DBType>();
      auto scalarSourceType = sourceType.getBaseType();
      auto scalarTargetType = targetType.getBaseType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      Value isNull;
      Value value;
      if (sourceType.isNullable()) {
         auto splitOp = rewriter.create<mlir::util::SplitOp>(loc, typeConverter->convertType(sourceType).dyn_cast_or_null<TupleType>().getTypes(), operands[0]);
         isNull = splitOp.vals()[0];
         value = splitOp.vals()[1];
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
            auto decimalrep = arrow::Decimal128::GetScaleMultiplier(sourceScale);
            std::vector<uint64_t> parts = {decimalrep.low_bits(), (uint64_t) decimalrep.high_bits()};
            auto multiplier = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), convertedTargetType, rewriter.getIntegerAttr(convertedTargetType, APInt(128, parts)));
            if (intType.getWidth() < 128) {
               value = rewriter.create<SignExtendIOp>(loc, value, convertedTargetType);
            }
            value = rewriter.create<mlir::MulIOp>(rewriter.getUnknownLoc(), convertedTargetType, value, multiplier);

         } else {
            return failure();
         }
      } else if (scalarSourceType.isa<db::FloatType>()) {
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
            auto decimalrep = arrow::Decimal128::GetScaleMultiplier(std::max(sourceScale, targetScale) - std::min(sourceScale, targetScale));
            std::vector<uint64_t> parts = {decimalrep.low_bits(), (uint64_t) decimalrep.high_bits()};
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
            auto decimalrep = arrow::Decimal128::GetScaleMultiplier(sourceScale);
            std::vector<uint64_t> parts = {decimalrep.low_bits(), (uint64_t) decimalrep.high_bits()};
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
         Value combined = rewriter.create<mlir::util::CombineOp>(loc, typeConverter->convertType(targetType), ValueRange({isNull, value}));
         rewriter.replaceOp(op, combined);
      } else {
         rewriter.replaceOp(op, value);
      }
      return success();
   }
};
//Lower Print Operation to an actual printf call
class DumpOpLowering : public ConversionPattern {
   public:
   explicit DumpOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DumpOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::db::DumpOp::Adaptor dumpOpAdaptor(operands);
      auto loc = op->getLoc();
      auto printOp = cast<mlir::db::DumpOp>(op);
      Value val = printOp.val();
      auto i128Type = IntegerType::get(rewriter.getContext(), 128);
      auto i64Type = IntegerType::get(rewriter.getContext(), 64);
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);
      auto i1Type = IntegerType::get(rewriter.getContext(), 1);
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto type = val.getType().dyn_cast_or_null<mlir::db::DBType>().getBaseType();

      auto f64Type = FloatType::getF64(rewriter.getContext());
      Value isNull;
      if (val.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
         TupleType tupleType = typeConverter->convertType(val.getType()).dyn_cast_or_null<TupleType>();
         auto splitOp = rewriter.create<mlir::util::SplitOp>(rewriter.getUnknownLoc(), tupleType.getTypes(), dumpOpAdaptor.val());
         isNull = splitOp.vals()[0];
         val = splitOp.vals()[1];
      } else {
         isNull = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
         val = dumpOpAdaptor.val();
      }

      operands[0].getType().dump();

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      // Get a symbol reference to the printf function, inserting it if necessary.
      if (auto dbIntType = type.dyn_cast_or_null<mlir::db::IntType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpInt", rewriter.getFunctionType({i1Type, i64Type}, {}));
         if (dbIntType.getWidth() < 64) {
            val = rewriter.create<SignExtendIOp>(loc, val, i64Type);
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::BoolType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpBool", rewriter.getFunctionType({i1Type, i1Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (auto decType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         Value low = rewriter.create<TruncateIOp>(loc, val, i64Type);
         Value shift = rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
         Value scale = rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
         Value high = rewriter.create<UnsignedShiftRightOp>(loc, i128Type, val, shift);
         high = rewriter.create<TruncateIOp>(loc, high, i64Type);

         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpDecimal", rewriter.getFunctionType({i1Type, i64Type, i64Type, i32Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, low, high, scale}));
      } else if (type.isa<mlir::db::DateType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpDate", rewriter.getFunctionType({i1Type, i32Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::TimestampType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpTimestamp", rewriter.getFunctionType({i1Type, i64Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::IntervalType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpInterval", rewriter.getFunctionType({i1Type, i64Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpFloat", rewriter.getFunctionType({i1Type, f64Type}, {}));
         if (floatType.getWidth() < 64) {
            val = rewriter.create<FPExtOp>(loc, val, f64Type);
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::StringType>()) {
         auto strType = MemRefType::get({}, IntegerType::get(getContext(), 8));
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpString", rewriter.getFunctionType({i1Type, strType, i64Type}, {}));
         Value len = rewriter.create<memref::DimOp>(loc, val, 0);
         len = rewriter.create<IndexCastOp>(loc, len, i64Type);
         val = rewriter.create<memref::ReinterpretCastOp>(loc, MemRefType::get({}, i8Type), val, (int64_t) 0, ArrayRef<int64_t>({}), ArrayRef<int64_t>({}));

         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val, len}));
      }
      rewriter.eraseOp(op);

      return success();
   }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// dbToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct DBToStdLoweringPass
   : public PassWrapper<DBToStdLoweringPass, OperationPass<ModuleOp>> {
   DBToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect, util::UtilDialect, memref::MemRefDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void DBToStdLoweringPass::runOnOperation() {
   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalDialect<StandardOpsDialect>();
   target.addLegalDialect<memref::MemRefDialect>();

   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();
   auto hasDBType = [](TypeRange types) {
      for (Type type : types)
         if (type.isa<db::DBType>()) {
            type.dump();
            return true;
         }
      return false;
   };
   target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto isLegal = !hasDBType(op.getType().getInputs()) &&
         !hasDBType(op.getType().getResults());
      op.dump();
      llvm::dbgs() << "isLegal:" << isLegal << "\n";
      return isLegal;
   });
   target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(
      [hasDBType](Operation* op) {
         auto isLegal = !hasDBType(op->getOperandTypes()) &&
            !hasDBType(op->getResultTypes());
         op->dump();
         llvm::dbgs() << "isLegal:" << isLegal << "\n";
         return isLegal;
      });
   //Add own types to LLVMTypeConverter
   TypeConverter typeConverter;
   typeConverter.addConversion([&](mlir::db::DBType type) {
      Type rawType = ::llvm::TypeSwitch<::mlir::db::DBType, mlir::Type>(type)
                        .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
                           return IntegerType::get(&getContext(), 1);
                        })
                        .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
                           return IntegerType::get(&getContext(), 32);
                        })
                        .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                           return IntegerType::get(&getContext(), 128);
                        })
                        .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
                           return IntegerType::get(&getContext(), t.getWidth());
                        })
                        .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
                           Type res;
                           if (t.getWidth() == 32) {
                              res = FloatType::getF32(&getContext());
                           } else if (t.getWidth() == 64) {
                              res = FloatType::getF64(&getContext());
                           }
                           return res;
                        })
                        .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
                           return MemRefType::get({-1}, IntegerType::get(&getContext(), 8));
                        })
                        .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
                           return IntegerType::get(&getContext(), 64);
                        })
                        .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
                           return IntegerType::get(&getContext(), 64);
                        })
                        .Default([](::mlir::Type) { return Type(); });
      if (type.isNullable()) {
         return (Type) TupleType::get(&getContext(), {IntegerType::get(&getContext(), 1), rawType});
      } else {
         return rawType;
      }
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   OwningRewritePatternList patterns(&getContext());
   /*patterns.add<FunctionLikeSignatureConversion>(&getContext(), typeConverter);
   patterns.add<ForwardOperands<CallOp>,
                ForwardOperands<CallIndirectOp>,
                ForwardOperands<ReturnOp>>(typeConverter, &getContext());*/
   mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   // Add own Lowering Patterns
   patterns.insert<NullOpLowering>(typeConverter, &getContext());
   patterns.insert<IsNullOpLowering>(typeConverter, &getContext());
   patterns.insert<DumpOpLowering>(typeConverter, &getContext());
   patterns.insert<ConstantLowering>(typeConverter, &getContext());
   patterns.insert<IfLowering>(typeConverter, &getContext());
   patterns.insert<YieldLowering>(typeConverter, &getContext());
   patterns.insert<AndOpLowering>(typeConverter, &getContext());
   patterns.insert<OrOpLowering>(typeConverter, &getContext());
   patterns.insert<NotOpLowering>(typeConverter, &getContext());
   patterns.insert<CmpOpLowering>(typeConverter, &getContext());
   patterns.insert<DateAddLowering>(typeConverter, &getContext());
   patterns.insert<DateSubLowering>(typeConverter, &getContext());
   patterns.insert<DateExtractLowering>(typeConverter, &getContext());

   patterns.insert<CastOpLowering>(typeConverter, &getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::IntType, mlir::AddIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::IntType, mlir::SubIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::IntType, mlir::MulIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::IntType, mlir::SignedDivIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::IntType, mlir::SignedRemIOp>>(typeConverter, &getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::FloatType, mlir::AddFOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::FloatType, mlir::SubFOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::FloatType, mlir::MulFOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::FloatType, mlir::MulFOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::FloatType, mlir::RemFOp>>(typeConverter, &getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::DecimalType, mlir::AddIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::DecimalType, mlir::SubIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::DecimalType, mlir::MulIOp>>(typeConverter, &getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::DivOp, mlir::SignedDivIOp>>(typeConverter, &getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::ModOp, mlir::SignedRemIOp>>(typeConverter, &getContext());

   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
