#include "mlir-support/mlir-support.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStdPass.h"
#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Conversion/DBToArrowStd/SerializationUtil.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {

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
arrow::TimeUnit::type toArrowTimeUnit(mlir::db::TimeUnitAttr attr) {
   switch (attr) {
      case mlir::db::TimeUnitAttr::second: return arrow::TimeUnit::SECOND;

      case mlir::db::TimeUnitAttr::millisecond: return arrow::TimeUnit::MILLI;
      case mlir::db::TimeUnitAttr::microsecond: return arrow::TimeUnit::MICRO;
      case mlir::db::TimeUnitAttr::nanosecond: return arrow::TimeUnit::NANO;
   }
   return arrow::TimeUnit::SECOND;
}

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
            int64_t integerVal = support::parseTimestamp(strAttr.getValue().str(), toArrowTimeUnit(timestampType.getUnit()));
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
            auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), v);
            nullValues.push_back(unPackOp.vals()[0]);
            return unPackOp.vals()[1];
         } else {
            return operand ? operand : v;
         }
      } else {
         return operand ? operand : v;
      }
   }
   Value isNull() {
      if (nullValues.empty()) {
         return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI1Type(), 0));
      }
      Value isNull;
      if (nullValues.size() >= 1) {
         isNull = nullValues.front();
      }
      for (size_t i = 1; i < nullValues.size(); i++) {
         isNull = builder.create<mlir::OrOp>(builder.getUnknownLoc(), isNull.getType(), isNull, nullValues[i]);
      }
      return isNull;
   };
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
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), mlir::TupleType::get(builder.getContext(), {i1Type, res.getType()}), ValueRange({isNull, res}));
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
      Value condition;
      auto boolType = ifOp.condition().getType().dyn_cast_or_null<db::BoolType>();
      if (boolType && boolType.isNullable()) {
         auto i1Type = rewriter.getI1Type();
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({i1Type, i1Type}), ifOp.condition());
         Value constTrue = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), i1Type, rewriter.getIntegerAttr(i1Type, 1));
         auto negated = rewriter.create<XOrOp>(rewriter.getUnknownLoc(), unpacked.getResult(0), constTrue); //negate
         auto anded = rewriter.create<mlir::AndOp>(rewriter.getUnknownLoc(), i1Type, negated, unpacked.getResult(1));
         condition = anded;
      } else {
         condition = ifOp.condition();
      }
      auto newIfOp = rewriter.create<mlir::scf::IfOp>(loc, TypeRange(resultTypes), condition, !ifOp.elseRegion().empty());
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

      rewriter.replaceOp(ifOp, newIfOp.results());

      return success();
   }
};
class WhileLowering : public ConversionPattern {
   public:
   explicit WhileLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::WhileOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto whileOp = cast<mlir::db::WhileOp>(op);
      auto loc = op->getLoc();
      std::vector<Type> resultTypes;
      for (auto res : whileOp.results()) {
         resultTypes.push_back(typeConverter->convertType(res.getType()));
      }
      auto newWhileOp = rewriter.create<mlir::scf::WhileOp>(loc, TypeRange(resultTypes), whileOp.inits());
      Block* before = new Block;
      Block* after = new Block;
      newWhileOp.before().push_back(before);
      newWhileOp.after().push_back(after);
      for (auto t : resultTypes) {
         before->addArgument(t);
         after->addArgument(t);
      }

      {
         scf::IfOp::ensureTerminator(newWhileOp.before(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newWhileOp.before().front());
         Block* originalThenBlock = &whileOp.before().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalThenBlock, terminator, newWhileOp.before().front().getArguments());
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }
      {
         scf::IfOp::ensureTerminator(newWhileOp.after(), rewriter, loc);
         auto insertPt = rewriter.saveInsertionPoint();
         rewriter.setInsertionPointToStart(&newWhileOp.after().front());
         Block* originalElseBlock = &whileOp.after().front();
         auto* terminator = rewriter.getInsertionBlock()->getTerminator();
         rewriter.mergeBlockBefore(originalElseBlock, terminator, newWhileOp.after().front().getArguments());
         rewriter.eraseOp(terminator);
         rewriter.restoreInsertionPoint(insertPt);
      }

      rewriter.replaceOp(whileOp, newWhileOp.results());

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
class ConditionLowering : public ConversionPattern {
   public:
   explicit ConditionLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ConditionOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      db::ConditionOpAdaptor adaptor(operands);
      db::ConditionOp conditionOp = cast<db::ConditionOp>(op);
      auto boolType = conditionOp.condition().getType().dyn_cast_or_null<db::BoolType>();
      if (boolType && boolType.isNullable()) {
         auto i1Type = rewriter.getI1Type();
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({i1Type, i1Type}), adaptor.condition());
         Value constTrue = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), i1Type, rewriter.getIntegerAttr(i1Type, 1));
         auto negated = rewriter.create<XOrOp>(rewriter.getUnknownLoc(), unpacked.getResult(0), constTrue); //negate
         auto anded = rewriter.create<mlir::AndOp>(rewriter.getUnknownLoc(), i1Type, negated, unpacked.getResult(1));
         rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, anded, adaptor.args());
      } else {
         rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, adaptor.condition(), adaptor.args());
      }
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

class ForOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit ForOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ForOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::ForOpAdaptor forOpAdaptor(operands);
      auto forOp = cast<mlir::db::ForOp>(op);
      auto collectionType = forOp.collection().getType().dyn_cast_or_null<mlir::db::CollectionType>();

      auto iterator = mlir::db::CollectionIterationImpl::getImpl(collectionType, forOp.collection(), functionRegistry);

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      std::vector<Value> results = iterator->implementLoop(forOpAdaptor.initArgs(), *typeConverter, rewriter, parentModule, [&](ValueRange values, OpBuilder builder) {
         auto yieldOp = cast<mlir::db::YieldOp>(forOp.getBody()->getTerminator());
         rewriter.mergeBlockBefore(forOp.getBody(), &*builder.getInsertionPoint(), values);
         std::vector<Value> results(yieldOp.results().begin(), yieldOp.results().end());
         rewriter.eraseOp(yieldOp);
         return results;
      });

      rewriter.replaceOp(op, results);
      return success();
   }
};
class GetTableLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit GetTableLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::GetTable::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto getTableOp = cast<mlir::db::GetTable>(op);
      auto tableName = rewriter.create<mlir::db::ConstantOp>(rewriter.getUnknownLoc(), mlir::db::StringType::get(rewriter.getContext(), false), rewriter.getStringAttr(getTableOp.tablename()));
      auto tablePtr = functionRegistry.call(rewriter, db::codegen::FunctionRegistry::FunctionId::ExecutionContextGetTable, mlir::ValueRange({getTableOp.execution_context(), tableName}))[0];
      rewriter.replaceOp(getTableOp, tablePtr);
      return success();
   }
};
class TableScanLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit TableScanLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::TableScan::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::TableScanAdaptor adaptor(operands);
      auto tablescan = cast<mlir::db::TableScan>(op);
      std::vector<Type> types;
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto ptrType = MemRefType::get({}, i8Type);
      auto indexType = IndexType::get(rewriter.getContext());

      std::vector<Value> values;
      types.push_back(ptrType);
      auto tablePtr = adaptor.table();
      values.push_back(tablePtr);
      for (auto c : tablescan.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         types.push_back(indexType);
         auto columnName = rewriter.create<mlir::db::ConstantOp>(rewriter.getUnknownLoc(), mlir::db::StringType::get(rewriter.getContext(), false), stringAttr);
         auto columnId = functionRegistry.call(rewriter, db::codegen::FunctionRegistry::FunctionId::TableGetColumnId, mlir::ValueRange({tablePtr, columnName}))[0];
         values.push_back(columnId);
      }
      rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, mlir::TupleType::get(rewriter.getContext(), types), values);
      return success();
   }
};
static Value getArrowDataType(OpBuilder& builder, db::codegen::FunctionRegistry& functionRegistry, db::DBType type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;
   int typeConstant = 0;
   int param1 = 0;
   int param2 = 0;
   auto loc = builder.getUnknownLoc();
   if (auto intType = type.dyn_cast_or_null<mlir::db::IntType>()) {
      switch (intType.getWidth()) {
         case 8: typeConstant = arrow::Type::type::INT8; break;
         case 16: typeConstant = arrow::Type::type::INT16; break;
         case 32: typeConstant = arrow::Type::type::INT32; break;
         case 64: typeConstant = arrow::Type::type::INT64; break;
      }
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      typeConstant = arrow::Type::type::DECIMAL128;
      param1 = decimalType.getP();
      param2 = decimalType.getS();
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
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
   }
   //TODO: also implement date types etc

   Value arrowTypeConstant = builder.create<mlir::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(typeConstant));
   Value arrowTypeParam1 = builder.create<mlir::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param1));
   Value arrowTypeParam2 = builder.create<mlir::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param2));

   Value arrowType = functionRegistry.call(builder, FunctionId::ArrowGetType2Param, ValueRange({arrowTypeConstant, arrowTypeParam1, arrowTypeParam2}))[0];
   return arrowType;
}
class CreateTableBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateTableBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateTableBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto createTB = cast<mlir::db::CreateTableBuilder>(op);
      auto loc = rewriter.getUnknownLoc();

      Value schema = functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaCreate, {})[0];
      TupleType rowType = createTB.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
      size_t i = 0;
      for (auto c : createTB.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         auto dbType = rowType.getType(i).cast<mlir::db::DBType>();
         auto arrowType = getArrowDataType(rewriter, functionRegistry, dbType);
         auto columnName = rewriter.create<mlir::db::ConstantOp>(loc, mlir::db::StringType::get(rewriter.getContext(), false), stringAttr);
         Value typeNullable = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), dbType.isNullable()));

         functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaAddField, ValueRange({schema, arrowType, typeNullable, columnName}));
         i += 1;
      }
      schema = functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaBuild, schema)[0];
      Value tableBuilder = functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderCreate, schema)[0];
      rewriter.replaceOp(op, tableBuilder);
      return success();
   }
};
class CreateVectorBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateVectorBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateVectorBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      Value vectorBuilder = functionRegistry.call(rewriter, FunctionId::VectorBuilderCreate, {})[0];
      rewriter.replaceOp(op, vectorBuilder);
      return success();
   }
};

static db::codegen::FunctionRegistry::FunctionId getStoreFunc(db::codegen::FunctionRegistry& functionRegistry, db::DBType type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;
   if (auto intType = type.dyn_cast_or_null<mlir::db::IntType>()) {
      switch (intType.getWidth()) {
         case 8: return FunctionId::ArrowTableBuilderAddInt8;
         case 16: return FunctionId::ArrowTableBuilderAddInt16;
         case 32: return FunctionId::ArrowTableBuilderAddInt32;
         case 64: return FunctionId::ArrowTableBuilderAddInt64;
      }
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      return FunctionId::ArrowTableBuilderAddBool;
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      return FunctionId::ArrowTableBuilderAddDecimal;
   } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
      switch (floatType.getWidth()) {
         case 32: return FunctionId ::ArrowTableBuilderAddFloat32;
         case 64: return FunctionId ::ArrowTableBuilderAddFloat64;
      }
   } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
      return FunctionId::ArrowTableBuilderAddBinary;
   } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
      if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
         return FunctionId::ArrowTableBuilderAddDate32;
      } else {
         return FunctionId::ArrowTableBuilderAddDate64;
      }
   }
   //TODO: implement other types too
   return FunctionId::ArrowTableBuilderAddInt32;
}
Value serialize(OpBuilder& builder, TypeConverter* converter, Value vectorBuilder, Value element, Type type, db::codegen::FunctionRegistry& functionRegistry) {
   if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
      auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
      std::vector<Value> serializedValues;
      std::vector<Type> types;
      auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
      for (size_t i = 0; i < tupleType.size(); i++) {
         Value currVal = unPackOp.getResult(i);
         Value serialized = serialize(builder, converter, vectorBuilder, currVal, originalTupleType.getType(i), functionRegistry);
         serializedValues.push_back(serialized);
         types.push_back(serialized.getType());
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
   } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
      if (stringType.isNullable()) {
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())}), element);
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::VectorBuilderAddNullableVarLen, ValueRange({vectorBuilder, unPackOp.getResult(0), unPackOp.getResult(1)}))[0];
      } else {
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::VectorBuilderAddVarLen, ValueRange({vectorBuilder, element}))[0];
      }
   } else {
      return element;
   }
}

class BuilderMergeLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit BuilderMergeLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BuilderMerge::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::BuilderMergeAdaptor mergeOpAdaptor(operands);
      auto mergeOp = cast<mlir::db::BuilderMerge>(op);
      auto loc = rewriter.getUnknownLoc();
      if (auto tableBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         TupleType rowType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
         auto loweredTypes = mergeOpAdaptor.val().getType().cast<TupleType>().getTypes();
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, loweredTypes, mergeOpAdaptor.val());
         size_t i = 0;
         Value falseValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         for (auto v : unPackOp.vals()) {
            Value isNull;
            if (mergeOp.val().getType().cast<TupleType>().getType(i).cast<db::DBType>().isNullable()) {
               auto nullUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, v.getType().cast<TupleType>().getTypes(), v);
               isNull = nullUnpacked.getResult(0);
               v = nullUnpacked->getResult(1);
            } else {
               isNull = falseValue;
            }
            Value columnId = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            functionRegistry.call(rewriter, getStoreFunc(functionRegistry, rowType.getType(i).cast<mlir::db::DBType>()), ValueRange({mergeOpAdaptor.builder(), columnId, isNull, v}));
            i++;
         }
         functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderFinishRow, mergeOpAdaptor.builder());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      } else if (auto vectorBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         Value v = serialize(rewriter, typeConverter, mergeOpAdaptor.builder(), mergeOpAdaptor.val(), mergeOp.val().getType(), functionRegistry);
         Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), v.getType());
         Value ptr = functionRegistry.call(rewriter, FunctionId::VectorBuilderMerge, ValueRange({mergeOpAdaptor.builder(), elementSize}))[0];
         Value typedPtr = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), v.getType(), llvm::Optional<int64_t>()), ptr);
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), v, typedPtr, Value());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      }

      return success();
   }
};
class SortOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit SortOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::SortOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      static size_t id = 0;
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::SortOpAdaptor sortOpAdaptor(operands);
      auto loweredVectorType = sortOpAdaptor.toSort().getType();
      auto sortOp = cast<mlir::db::SortOp>(op);
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto ptrType = MemRefType::get({}, rewriter.getIntegerType(8));
      Type elementType = sortOp.toSort().getType().cast<mlir::db::VectorType>().getElementType();
      Type serializedType = mlir::db::codegen::SerializationUtil::serializedType(rewriter, *typeConverter, elementType);
      FuncOp funcOp;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto byteRangeType = MemRefType::get({-1}, rewriter.getIntegerType(8));
         funcOp = rewriter.create<FuncOp>(parentModule.getLoc(), "db_sort_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({byteRangeType, ptrType, ptrType}), TypeRange(mlir::db::BoolType::get(rewriter.getContext()))));
         funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({byteRangeType, ptrType, ptrType}));
         funcOp.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value varLenData = funcBody->getArgument(0);
         Value left = funcBody->getArgument(1);
         Value right = funcBody->getArgument(2);

         Value genericMemrefLeft = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedType, llvm::Optional<int64_t>()), left);
         Value genericMemrefRight = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedType, llvm::Optional<int64_t>()), right);
         Value serializedTupleLeft = rewriter.create<util::LoadOp>(sortOp.getLoc(), serializedType, genericMemrefLeft, Value());
         Value serializedTupleRight = rewriter.create<util::LoadOp>(sortOp.getLoc(), serializedType, genericMemrefRight, Value());
         Value tupleLeft = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleLeft, elementType);
         Value tupleRight = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleRight, elementType);
         auto terminator = rewriter.create<mlir::ReturnOp>(sortOp.getLoc());
         Block* sortLambda = &sortOp.region().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.mergeBlockBefore(sortLambda, terminator, {tupleLeft, tupleRight});
         mlir::db::YieldOp yieldOp = mlir::cast<mlir::db::YieldOp>(terminator->getPrevNode());
         Value x = yieldOp.results()[0];
         x.setType(rewriter.getI1Type()); //todo: bad hack ;)
         rewriter.create<mlir::ReturnOp>(sortOp.getLoc(), x);
         rewriter.eraseOp(sortLambdaTerminator);
         rewriter.eraseOp(terminator);
      }
      Value functionPointer = rewriter.create<mlir::ConstantOp>(sortOp->getLoc(), funcOp.type(), rewriter.getSymbolRefAttr(funcOp.sym_name()));
      Type vectorMemrefType = util::GenericMemrefType::get(rewriter.getContext(), loweredVectorType, llvm::Optional<int64_t>());
      Value allocaVec = rewriter.create<mlir::util::AllocaOp>(sortOp->getLoc(), vectorMemrefType, Value());
      Value allocaNewVec = rewriter.create<mlir::util::AllocaOp>(sortOp->getLoc(), vectorMemrefType, Value());
      rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), sortOpAdaptor.toSort(), allocaVec, Value());
      Value plainMemref = rewriter.create<mlir::util::ToMemrefOp>(sortOp->getLoc(), ptrType, allocaVec);
      Value plainMemrefNew = rewriter.create<mlir::util::ToMemrefOp>(sortOp->getLoc(), ptrType, allocaNewVec);
      Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), serializedType);
      functionRegistry.call(rewriter, FunctionId::SortVector, {plainMemref, elementSize, functionPointer, plainMemrefNew});
      Value newVector = rewriter.create<util::LoadOp>(sortOp.getLoc(), loweredVectorType, allocaNewVec, Value());
      rewriter.replaceOp(op, newVector);
      return success();
   }
};
class CreateAggrHTBuilderLowering : public ConversionPattern {

   public:
   explicit CreateAggrHTBuilderLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateAggrHTBuilder::getOperationName(), 1, context){}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      static size_t id = 0;
      mlir::db::SortOpAdaptor createAdaptor(operands);
      auto createOp = cast<mlir::db::CreateAggrHTBuilder>(op);
      auto clonedCreateOp = cast<mlir::db::CreateAggrHTBuilder>(op->clone());

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto ptrType = MemRefType::get({}, rewriter.getIntegerType(8));
      Type keyType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getKeyType();
      TupleType keyTupleType = keyType.cast<mlir::TupleType>();
      Type valType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getValType();

      Type serializedValType = mlir::db::codegen::SerializationUtil::serializedType(rewriter, *typeConverter, valType);
      Type serializedKeyType = mlir::db::codegen::SerializationUtil::serializedType(rewriter, *typeConverter, keyType);
      FuncOp compareFunc;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto byteRangeType = MemRefType::get({-1}, rewriter.getIntegerType(8));
         compareFunc = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({byteRangeType, ptrType, ptrType}), TypeRange(mlir::db::BoolType::get(rewriter.getContext()))));
         compareFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({byteRangeType, ptrType, ptrType}));
         compareFunc.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value varLenData = funcBody->getArgument(0);
         Value left = funcBody->getArgument(1);
         Value right = funcBody->getArgument(2);

         Value genericMemrefLeft = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedKeyType, llvm::Optional<int64_t>()), left);
         Value genericMemrefRight = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedKeyType, llvm::Optional<int64_t>()), right);
         Value serializedTupleLeft = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedKeyType, genericMemrefLeft, Value());
         Value serializedTupleRight = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedKeyType, genericMemrefRight, Value());
         Value tupleLeft = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleLeft, keyType);
         Value tupleRight = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleRight, keyType);
         Value equal = rewriter.create<mlir::db::ConstantOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()),rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         auto leftUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), keyTupleType.getTypes(), tupleLeft);
         auto rightUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), keyTupleType.getTypes(), tupleRight);
         for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
            auto compared = rewriter.create<mlir::db::CmpOp>(rewriter.getUnknownLoc(), mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));

            Value localEqual = rewriter.create<mlir::db::AndOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext(), equal.getType().cast<mlir::db::BoolType>().getNullable() || compared.getType().cast<mlir::db::BoolType>().getNullable()), ValueRange({equal, compared}));
            equal=localEqual;
         }
         //equal.setType(rewriter.getI1Type()); //todo: bad hack ;)
         rewriter.create<mlir::ReturnOp>(createOp->getLoc(), equal);
      }
      FuncOp funcOp;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto byteRangeType = MemRefType::get({-1}, rewriter.getIntegerType(8));
         funcOp = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_update" + std::to_string(id++), rewriter.getFunctionType(TypeRange({byteRangeType, ptrType, ptrType}), TypeRange()));
         funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({byteRangeType, ptrType, ptrType}));
         funcOp.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value varLenData = funcBody->getArgument(0);
         Value left = funcBody->getArgument(1);
         Value right = funcBody->getArgument(2);

         Value genericMemrefLeft = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedValType, llvm::Optional<int64_t>()), left);
         Value genericMemrefRight = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedValType, llvm::Optional<int64_t>()), right);
         Value serializedTupleLeft = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedValType, genericMemrefLeft, Value());
         Value serializedTupleRight = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedValType, genericMemrefRight, Value());
         Value tupleLeft = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleLeft, valType);
         Value tupleRight = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleRight, valType);
         auto terminator = rewriter.create<mlir::ReturnOp>(createOp.getLoc());
         Block* sortLambda = &clonedCreateOp.region().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.mergeBlockBefore(sortLambda, terminator, {tupleLeft, tupleRight});
         mlir::db::YieldOp yieldOp = mlir::cast<mlir::db::YieldOp>(terminator->getPrevNode());
         Value x = yieldOp.results()[0];
         rewriter.setInsertionPoint(terminator);
         //todo: serialize!!
         x.setType(serializedValType); //todo: hacky
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), x, genericMemrefLeft, Value());

         funcOp->dump();
         rewriter.eraseOp(sortLambdaTerminator);
      }

      Value updateFunctionPointer = rewriter.create<mlir::ConstantOp>(createOp->getLoc(), funcOp.type(), rewriter.getSymbolRefAttr(funcOp.sym_name()));
      Value compareFunctionPointer = rewriter.create<mlir::ConstantOp>(createOp->getLoc(), compareFunc.type(), rewriter.getSymbolRefAttr(compareFunc.sym_name()));

      Value packOp = rewriter.create<mlir::util::PackOp>(rewriter.getUnknownLoc(), TupleType::get(rewriter.getContext(), {compareFunc.type(), funcOp.type()}), ValueRange({compareFunctionPointer, updateFunctionPointer}));
      rewriter.replaceOp(op, packOp);
      return success();
   }
};
class BuilderBuildLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit BuilderBuildLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BuilderBuild::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::BuilderBuildAdaptor buildAdaptor(operands);
      auto buildOp = cast<mlir::db::BuilderBuild>(op);
      if (auto tableBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         Value table = functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, table);
      } else if (auto vectorBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         Value vector = functionRegistry.call(rewriter, FunctionId::VectorBuilderBuild, buildAdaptor.builder())[0];

         rewriter.replaceOp(op, vector);
      }

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
   Value hashImpl(OpBuilder& builder, Value v, Value totalHash) const {
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
         auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), v);
         for (auto v : unpacked->getResults()) {
            totalHash = hashImpl(builder, v, totalHash);
         }
         return totalHash;
      }
      assert(false && "should not happen");
      return Value();
   }

   public:
   explicit HashLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::Hash::getOperationName(), 1, context), functionRegistry(functionRegistry) {}
   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::HashAdaptor hashAdaptor(operands);
      hashAdaptor.val().getType().dump();
      Value const0 = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
      rewriter.replaceOp(op, hashImpl(rewriter, hashAdaptor.val(), const0));
      return success();
   }
};
class CreateRangeLowering : public ConversionPattern {
   public:
   explicit CreateRangeLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateRange::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = rewriter.getUnknownLoc();
      auto createRangeOp = cast<mlir::db::CreateRange>(op);
      Type storageType = createRangeOp.range().getType().cast<mlir::db::RangeType>().getElementType();
      Value combined = rewriter.create<mlir::util::PackOp>(loc, TypeRange(TupleType::get(getContext(), {storageType, storageType, storageType})), ValueRange({createRangeOp.lower(), createRangeOp.upper(), createRangeOp.step()}));
      rewriter.replaceOp(op, combined);

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
class CmpOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CmpOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CmpOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}
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
      auto loc = rewriter.getUnknownLoc();
      NullHandler nullHandler(*typeConverter, rewriter);
      auto cmpOp = cast<db::CmpOp>(op);

      auto type = cmpOp.left().getType().cast<db::DBType>().getBaseType();
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
      } else if (type.isa<db::StringType>()) {
         using FuncId = mlir::db::codegen::FunctionRegistry::FunctionId;
         FuncId cmpFunc = funcForStrCompare(cmpOp.predicate());
         Value res = functionRegistry.call(rewriter, cmpFunc, ValueRange({nullHandler.isNull(), cmpOp.left(), cmpOp.right()}))[0];
         rewriter.replaceOp(op, nullHandler.combineResult(res));
         return success();
      }
      return failure();
   }
};

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
      NullHandler nullHandler(*typeConverter, rewriter);
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
      NullHandler nullHandler(*typeConverter, rewriter);
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

class CastOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CastOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
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
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
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
         }
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
         } else if (scalarTargetType.isa<db::StringType>()) {
            if (intType.getWidth() < 64) {
               value = rewriter.create<SignExtendIOp>(loc, value, rewriter.getI64Type());
            }
            value = functionRegistry.call(rewriter, FunctionId ::CastInt64ToString, ValueRange({isNull, value}))[0];
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
         } else if (scalarTargetType.isa<db::StringType>()) {
            FunctionId castFn = floatType.getWidth() == 32 ? FunctionId ::CastFloat32ToString : FunctionId ::CastFloat64ToString;
            value = functionRegistry.call(rewriter, castFn, ValueRange({isNull, value}))[0];
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
         } else if (scalarTargetType.isa<db::StringType>()) {
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
//Lower Print Operation to an actual printf call
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
static Type convertFunctionType(FunctionType type, TypeConverter& typeConverter) {
   TypeConverter::SignatureConversion result(type.getNumInputs());
   SmallVector<Type, 1> newResults;
   if (failed(typeConverter.convertSignatureArgs(type.getInputs(), result)) ||
       failed(typeConverter.convertTypes(type.getResults(), newResults))) {
      return Type();
   }

   auto newType = FunctionType::get(type.getContext(),
                                    result.getConvertedTypes(), newResults);
   return newType;
}
class FuncConstLowering : public ConversionPattern {
   public:
   explicit FuncConstLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::ConstantOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::ConstantOp constantOp = mlir::cast<mlir::ConstantOp>(op);
      if (auto type = constantOp.getType().dyn_cast_or_null<mlir::FunctionType>()) {
         // Convert the original function types.

         rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, convertFunctionType(type, *typeConverter), constantOp.value());
         return success();

      } else {
         return failure();
      }
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
static bool hasDBType(TypeRange types) {
   bool res = false;
   for (Type type : types) {
      if (type.isa<db::DBType>()) {
         res |= true;
      } else if (auto tupleType = type.dyn_cast_or_null<TupleType>()) {
         res |= hasDBType(tupleType.getTypes());
      } else if (auto genericMemrefType = type.dyn_cast_or_null<util::GenericMemrefType>()) {
         res |= hasDBType(genericMemrefType.getElementType());
      } else if (auto functionType = type.dyn_cast_or_null<mlir::FunctionType>()) {
         res |= hasDBType(functionType.getInputs()) ||
            hasDBType(functionType.getResults());
      } else if (type.isa<mlir::db::TableType>() || type.isa<mlir::db::VectorType>()) {
         res = true;
      }
   }
   return res;
}
void DBToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   mlir::db::codegen::FunctionRegistry functionRegistry(&getContext());
   functionRegistry.registerFunctions();
   using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<StandardOpsDialect>();
   target.addLegalDialect<memref::MemRefDialect>();

   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();
   target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto isLegal = !hasDBType(op.getType().getInputs()) &&
         !hasDBType(op.getType().getResults());
      //op->dump();
      //llvm::dbgs() << "isLegal:" << isLegal << "\n";
      return isLegal;
   });
   target.addDynamicallyLegalOp<ConstantOp>([&](ConstantOp op) {
      if (auto functionType = op.getType().dyn_cast_or_null<mlir::FunctionType>()) {
         auto isLegal = !hasDBType(functionType.getInputs()) &&
            !hasDBType(functionType.getResults());
         return isLegal;
      } else {
         return true;
      }
   });
   target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(
      [](Operation* op) {
         auto isLegal = !hasDBType(op->getOperandTypes()) &&
            !hasDBType(op->getResultTypes());
         //op->dump();
         //llvm::dbgs() << "isLegal:" << isLegal << "\n";
         return isLegal;
      });
   target.addDynamicallyLegalOp<util::DimOp, util::SetTupleOp, util::GetTupleOp, util::UndefTupleOp, util::PackOp, util::UnPackOp, util::ToGenericMemrefOp, util::StoreOp, util::LoadOp, util::MemberRefOp, util::FromRawPointerOp, util::ToRawPointerOp, util::AllocOp, util::DeAllocOp, util::AllocaOp, util::AllocaOp>(
      [](Operation* op) {
         auto isLegal = !hasDBType(op->getOperandTypes()) &&
            !hasDBType(op->getResultTypes());

         return isLegal;
      });
   target.addDynamicallyLegalOp<util::SizeOfOp>(
      [](util::SizeOfOp op) {
         auto isLegal = !hasDBType(op.type());
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
                           if (t.getUnit() == mlir::db::DateUnitAttr::day) {
                              return IntegerType::get(&getContext(), 32);
                           } else {
                              return IntegerType::get(&getContext(), 64);
                           }
                        })
                        .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
                           if (t.getUnit() == mlir::db::TimeUnitAttr::second && t.getUnit() == mlir::db::TimeUnitAttr::millisecond) {
                              return IntegerType::get(&getContext(), 32);
                           } else {
                              return IntegerType::get(&getContext(), 64);
                           }
                        })
                        .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                           return IntegerType::get(&getContext(), 128);
                        })
                        .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
                           return IntegerType::get(&getContext(), t.getWidth());
                        })
                        .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
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
                        .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
                           return IntegerType::get(&getContext(), 64);
                        })
                        .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
                           if (t.getUnit() == mlir::db::IntervalUnitAttr::daytime) {
                              return IntegerType::get(&getContext(), 64);
                           } else {
                              return IntegerType::get(&getContext(), 32);
                           }
                        })
                        .Default([](::mlir::Type) { return Type(); });
      if (type.isNullable()) {
         return (Type) TupleType::get(&getContext(), {IntegerType::get(&getContext(), 1), rawType});
      } else {
         return rawType;
      }
   });
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::FunctionType functionType) {
      return convertFunctionType(functionType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::FunctionType functionType) {
      return convertFunctionType(functionType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::db::TableType tableType) {
      return MemRefType::get({}, IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](mlir::db::TableBuilderType tableType) {
      return MemRefType::get({}, IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](mlir::db::VectorBuilderType vectorBuilderType) {
      return MemRefType::get({}, IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](mlir::db::AggrHTBuilderType aggrHtBuilderType) {
      auto ptrType = MemRefType::get({}, IntegerType::get(&getContext(), 8));
      auto strType = MemRefType::get({-1}, IntegerType::get(&getContext(), 8));
      auto compareFuncType = FunctionType::get(&getContext(), {strType, ptrType, ptrType}, {IntegerType::get(&getContext(), 1)});
     auto updateFuncType = FunctionType::get(&getContext(), {strType, ptrType, ptrType}, {});

     return TupleType::get(&getContext(), {compareFuncType, updateFuncType});
   });
   typeConverter.addConversion([&](mlir::db::VectorType vectorType) {
      auto ptrType = MemRefType::get({-1}, IntegerType::get(&getContext(), 8));
      return TupleType::get(&getContext(), {ptrType, ptrType});
   });
   typeConverter.addConversion([&](mlir::db::RangeType rangeType) {
      auto convertedType = typeConverter.convertType(rangeType.getElementType());
      return TupleType::get(&getContext(), {convertedType, convertedType, convertedType});
   });
   typeConverter.addConversion([&](mlir::db::GenericIterableType genericIterableType) {
      Type elementType = genericIterableType.getElementType();
      Type nestedElementType = elementType;
      if (auto nested = elementType.dyn_cast_or_null<mlir::db::GenericIterableType>()) {
         nestedElementType = nested.getElementType();
      }
      if (genericIterableType.getIteratorName() == "table_chunk_iterator") {
         std::vector<Type> types;
         auto i8Type = IntegerType::get(&getContext(), 8);
         auto ptrType = MemRefType::get({}, i8Type);
         auto indexType = IndexType::get(&getContext());
         types.push_back(ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) TupleType::get(&getContext(), types);
      } else if (genericIterableType.getIteratorName() == "table_row_iterator") {
         std::vector<Type> types;
         auto i8Type = IntegerType::get(&getContext(), 8);
         auto ptrType = MemRefType::get({}, i8Type);
         auto indexType = IndexType::get(&getContext());
         types.push_back(ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) TupleType::get(&getContext(), types);
      }
      return Type();
   });
   typeConverter.addConversion([&](mlir::IntegerType iType) { return iType; });
   typeConverter.addConversion([&](mlir::IndexType iType) { return iType; });
   typeConverter.addConversion([&](mlir::FloatType fType) { return fType; });
   typeConverter.addConversion([&](mlir::MemRefType refType) { return refType; });

   typeConverter.addSourceMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, IntegerType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, IntegerType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::TableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::TableBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::VectorBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::VectorBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::VectorType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::VectorType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, MemRefType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, MemRefType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, TupleType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, TupleType type, ValueRange valueRange, Location loc) {
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
   mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   // Add own Lowering Patterns
   patterns.insert<NullOpLowering>(typeConverter, &getContext());
   patterns.insert<IsNullOpLowering>(typeConverter, &getContext());
   patterns.insert<CombineNullOpLowering>(typeConverter, &getContext());
   patterns.insert<FuncConstLowering>(typeConverter, &getContext());

   patterns.insert<DumpOpLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<DumpIndexOpLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<ConstantLowering>(typeConverter, &getContext());
   patterns.insert<IfLowering>(typeConverter, &getContext());
   patterns.insert<WhileLowering>(typeConverter, &getContext());
   patterns.insert<ConditionLowering>(typeConverter, &getContext());
   patterns.insert<YieldLowering>(typeConverter, &getContext());
   patterns.insert<AndOpLowering>(typeConverter, &getContext());
   patterns.insert<OrOpLowering>(typeConverter, &getContext());
   patterns.insert<NotOpLowering>(typeConverter, &getContext());
   patterns.insert<CmpOpLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CastOpLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<TableScanLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateTableBuilderLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateVectorBuilderLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateAggrHTBuilderLowering>(typeConverter, &getContext());

   patterns.insert<BuilderMergeLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<BuilderBuildLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<HashLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<GetTableLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<SortOpLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<ForOpLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateRangeLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::IntType, mlir::AddIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::IntType, mlir::SubIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::IntType, mlir::MulIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::IntType, mlir::SignedDivIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::IntType, mlir::SignedRemIOp>>(typeConverter, &getContext());

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::UIntType, mlir::AddIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::UIntType, mlir::SubIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::UIntType, mlir::MulIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::UIntType, mlir::UnsignedDivIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::db::UIntType, mlir::UnsignedRemIOp>>(typeConverter, &getContext());

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

   auto ensureDate64 = [](mlir::db::DateType dateType, Value v, ConversionPatternRewriter& rewriter) {
      if (dateType.getUnit() == db::DateUnitAttr::day) {
         auto i64Type = IntegerType::get(rewriter.getContext(), 64);
         v = rewriter.template create<ZeroExtendIOp>(rewriter.getUnknownLoc(), v, i64Type);
         Value multiplier = rewriter.template create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(i64Type, 24 * 60 * 60 * 1000));
         v = rewriter.template create<MulIOp>(rewriter.getUnknownLoc(), v, multiplier);
         return v;
      } else {
         return v;
      }
   };
   auto negateInterval = [](mlir::db::IntervalType dateType, Value v, ConversionPatternRewriter& rewriter) {
      Value multiplier = rewriter.template create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(v.getType(), -1));
      return rewriter.template create<MulIOp>(rewriter.getUnknownLoc(), v, multiplier);
   };
   auto transformDateBack = [](mlir::db::DateType dateType, Value v, ConversionPatternRewriter& rewriter) {
      if (dateType.getUnit() == db::DateUnitAttr::day) {
         auto i64Type = IntegerType::get(rewriter.getContext(), 64);
         auto i32Type = IntegerType::get(rewriter.getContext(), 32);
         Value multiplier = rewriter.template create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(i64Type, 24 * 60 * 60 * 1000));
         v = rewriter.template create<UnsignedDivIOp>(rewriter.getUnknownLoc(), v, multiplier);
         v = rewriter.template create<TruncateIOp>(rewriter.getUnknownLoc(), v, i32Type);
         return v;
      }
      return v;
   };
   auto identity = [](auto, Value v, auto&) { return v; };
   auto rightleft = [](Value left, Value right) { return std::vector<Value>({right, left}); };
   auto dateAddFunction = [&](Operation* op, mlir::db::DateType dateType, mlir::db::IntervalType intervalType, ConversionPatternRewriter& rewriter) {
      return functionRegistry.getFunction(rewriter, FunctionId::TimestampAddMonth);
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
      &getContext(), ensureDate64, identity, rightleft, dateAddFunction, transformDateBack);
   patterns.insert<SimpleBinOpToFuncLowering<mlir::db::DateSubOp, mlir::db::DateType, mlir::db::IntervalType, mlir::db::DateType>>(
      &getContext(), ensureDate64, negateInterval, rightleft, dateAddFunction, transformDateBack);
   patterns.insert<SimpleUnOpToFuncLowering<mlir::db::DateExtractOp, mlir::db::DateType, mlir::db::IntType>>(
      &getContext(), ensureDate64, dateExtractFunction, identity);

   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
