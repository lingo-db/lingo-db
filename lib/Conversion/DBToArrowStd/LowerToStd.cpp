#include "mlir-support/mlir-support.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStdPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
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
   funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
   return funcOp;
}
//declare external function or return reference to already existing one
static FuncOp getOrInsertGandivaFn(PatternRewriter& rewriter,
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

static Value insertConstString(std::string str, ModuleOp module, ConversionPatternRewriter& rewriter) {
   auto loc = rewriter.getUnknownLoc();
   auto i8Type = IntegerType::get(rewriter.getContext(), 8);
   auto insertionPoint = rewriter.saveInsertionPoint();
   int64_t strLen = str.size();
   std::vector<uint8_t> vec;
   for (auto c : str) {
      vec.push_back(c);
   }
   auto strStaticType = MemRefType::get({strLen}, i8Type);
   auto strDynamicType = MemRefType::get({-1}, IntegerType::get(rewriter.getContext(), 8));
   rewriter.setInsertionPointToStart(module.getBody());
   auto initialValue = DenseIntElementsAttr::get(
      RankedTensorType::get({strLen}, i8Type), vec);
   static int id = 0;
   auto globalop = rewriter.create<mlir::memref::GlobalOp>(rewriter.getUnknownLoc(), "db_constant_string" + std::to_string(id++), rewriter.getStringAttr("private"), strStaticType, initialValue, true);
   rewriter.restoreInsertionPoint(insertionPoint);
   Value conststr = rewriter.create<mlir::memref::GetGlobalOp>(loc, strStaticType, globalop.sym_name());
   return rewriter.create<memref::CastOp>(loc, conststr, strDynamicType);
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
            ModuleOp parentModule = op->getParentOfType<ModuleOp>();
            rewriter.replaceOp(op, insertConstString(stringAttr.getValue().str(), parentModule, rewriter));
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

class ForOpLowering : public ConversionPattern {
   public:
   explicit ForOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ForOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto forOp = cast<mlir::db::ForOp>(op);
      forOp->dump();
      auto collectionType = forOp.collection().getType().dyn_cast_or_null<mlir::db::CollectionType>();

      auto iterator = mlir::db::CollectionIterationImpl::getImpl(collectionType, forOp.collection());
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto* terminator = forOp.getBody()->getTerminator();
      iterator->implementLoop({}, *typeConverter, rewriter, parentModule, [&](auto val, OpBuilder builder) {
         rewriter.mergeBlockBefore(forOp.getBody(),&*builder.getInsertionPoint(),val);
        rewriter.eraseOp(terminator);
        return std::vector<Value>({}); });
      rewriter.eraseOp(op);
      return success();
   }
};
class TableScanLowering : public ConversionPattern {
   public:
   explicit TableScanLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::TableScan::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto tablescan = cast<mlir::db::TableScan>(op);
      std::vector<Type> types;
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto ptrType = MemRefType::get({}, i8Type);
      auto strType = MemRefType::get({-1}, i8Type);

      auto indexType = IndexType::get(rewriter.getContext());
      auto getTableFn = getOrInsertFn(rewriter, parentModule, "get_table", rewriter.getFunctionType({ptrType, strType}, {ptrType}));
      auto getColumnIdFn = getOrInsertFn(rewriter, parentModule, "get_column_id", rewriter.getFunctionType({ptrType, strType}, {indexType}));

      std::vector<Value> values;
      types.push_back(ptrType);
      auto tableName = insertConstString(tablescan.tablename().str(), parentModule, rewriter);
      auto call = rewriter.create<CallOp>(rewriter.getUnknownLoc(), getTableFn, mlir::ValueRange({tablescan.execution_context(), tableName}));
      Value tablePtr = call->getResult(0);
      values.push_back(tablePtr);
      for (auto c : tablescan.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         types.push_back(indexType);
         auto colName = insertConstString(stringAttr.getValue().str(), parentModule, rewriter);
         auto call = rewriter.create<CallOp>(rewriter.getUnknownLoc(), getColumnIdFn, mlir::ValueRange({tablePtr, colName}));
         values.push_back(call->getResult(0));
      }
      rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, mlir::TupleType::get(rewriter.getContext(), types), values);
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

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      // Get a symbol reference to the printf function, inserting it if necessary.
      if (auto dbIntType = type.dyn_cast_or_null<mlir::db::IntType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dump_int", rewriter.getFunctionType({i1Type, i64Type}, {}));
         if (dbIntType.getWidth() < 64) {
            val = rewriter.create<SignExtendIOp>(loc, val, i64Type);
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (auto dbUIntType = type.dyn_cast_or_null<mlir::db::UIntType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dump_uint", rewriter.getFunctionType({i1Type, i64Type}, {}));
         if (dbUIntType.getWidth() < 64) {
            val = rewriter.create<ZeroExtendIOp>(loc, val, i64Type);
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::BoolType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dump_bool", rewriter.getFunctionType({i1Type, i1Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (auto decType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         Value low = rewriter.create<TruncateIOp>(loc, val, i64Type);
         Value shift = rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
         Value scale = rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
         Value high = rewriter.create<UnsignedShiftRightOp>(loc, i128Type, val, shift);
         high = rewriter.create<TruncateIOp>(loc, high, i64Type);

         auto printRef = getOrInsertFn(rewriter, parentModule, "dump_decimal", rewriter.getFunctionType({i1Type, i64Type, i64Type, i32Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, low, high, scale}));
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         FuncOp printRef;

         if (dateType.getUnit() == mlir::db::DateUnitAttr::millisecond) {
            printRef = getOrInsertFn(rewriter, parentModule, "dump_date_millisecond", rewriter.getFunctionType({i1Type, i64Type}, {}));
         } else {
            printRef = getOrInsertFn(rewriter, parentModule, "dump_date_day", rewriter.getFunctionType({i1Type, i32Type}, {}));
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         std::string unit = mlir::db::stringifyTimeUnitAttr(timestampType.getUnit()).str();
         auto printRef = getOrInsertFn(rewriter, parentModule, "dump_timestamp_" + unit, rewriter.getFunctionType({i1Type, i64Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            auto printRef = getOrInsertFn(rewriter, parentModule, "dump_interval_months", rewriter.getFunctionType({i1Type, i32Type}, {}));
            rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
         } else {
            auto printRef = getOrInsertFn(rewriter, parentModule, "dump_interval_daytime", rewriter.getFunctionType({i1Type, i64Type}, {}));
            rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
         }

      } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dump_float", rewriter.getFunctionType({i1Type, f64Type}, {}));
         if (floatType.getWidth() < 64) {
            val = rewriter.create<FPExtOp>(loc, val, f64Type);
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
      } else if (type.isa<mlir::db::StringType>()) {
         auto strType = MemRefType::get({-1}, IntegerType::get(getContext(), 8));
         auto printRef = getOrInsertFn(rewriter, parentModule, "dump_string", rewriter.getFunctionType({i1Type, strType}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({isNull, val}));
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
   for (Type type : types) {
      if (type.isa<db::DBType>()) {
         return true;
      } else if (auto tupleType = type.dyn_cast_or_null<TupleType>()) {
         return hasDBType(tupleType.getTypes());
      }
   }
   return false;
}
void DBToStdLoweringPass::runOnOperation() {
   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalDialect<StandardOpsDialect>();
   target.addLegalDialect<memref::MemRefDialect>();

   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();
   target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto isLegal = !hasDBType(op.getType().getInputs()) &&
         !hasDBType(op.getType().getResults());
      return isLegal;
   });
   target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(
      [](Operation* op) {
         auto isLegal = !hasDBType(op->getOperandTypes()) &&
            !hasDBType(op->getResultTypes());
         return isLegal;
      });
   target.addDynamicallyLegalOp<util::SetTupleOp, util::GetTupleOp, util::UndefTupleOp, util::PackOp, util::UnPackOp>(
      [](Operation* op) {
         auto isLegal = !hasDBType(op->getOperandTypes()) &&
            !hasDBType(op->getResultTypes());
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
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
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

   patterns.insert<DumpOpLowering>(typeConverter, &getContext());
   patterns.insert<ConstantLowering>(typeConverter, &getContext());
   patterns.insert<IfLowering>(typeConverter, &getContext());
   patterns.insert<YieldLowering>(typeConverter, &getContext());
   patterns.insert<AndOpLowering>(typeConverter, &getContext());
   patterns.insert<OrOpLowering>(typeConverter, &getContext());
   patterns.insert<NotOpLowering>(typeConverter, &getContext());
   patterns.insert<CmpOpLowering>(typeConverter, &getContext());
   patterns.insert<CastOpLowering>(typeConverter, &getContext());
   patterns.insert<TableScanLowering>(typeConverter, &getContext());
   patterns.insert<ForOpLowering>(typeConverter, &getContext());

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
   auto dateAddFunction = [](Operation* op, mlir::db::DateType dateType, mlir::db::IntervalType intervalType, ConversionPatternRewriter& rewriter) {
      auto i64Type = IntegerType::get(rewriter.getContext(), 64);
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);
      return getOrInsertGandivaFn(rewriter, op->getParentOfType<ModuleOp>(), "timestampaddMonth_int32_date64", rewriter.getFunctionType({i32Type, i64Type}, {i64Type}));
   };
   auto dateExtractFunction = [](mlir::db::DateExtractOp dateExtractOp, mlir::db::DateType dateType, ConversionPatternRewriter& rewriter) {
      auto i64Type = IntegerType::get(rewriter.getContext(), 64);
      std::string unitString = mlir::db::stringifyExtractableTimeUnitAttr(dateExtractOp.unit()).str();
      unitString[0] = toupper(unitString[0]);
      return getOrInsertGandivaFn(rewriter, dateExtractOp->getParentOfType<ModuleOp>(), "extract" + unitString + "_date64", rewriter.getFunctionType({i64Type}, {i64Type}));
   };

   patterns.insert<SimpleBinOpToFuncLowering<mlir::db::DateAddOp, mlir::db::DateType, mlir::db::IntervalType, mlir::db::DateType>>(
      &getContext(), ensureDate64, identity, rightleft, dateAddFunction, transformDateBack);
   patterns.insert<SimpleBinOpToFuncLowering<mlir::db::DateSubOp, mlir::db::DateType, mlir::db::IntervalType, mlir::db::DateType>>(
      &getContext(), ensureDate64, negateInterval, rightleft, dateAddFunction, transformDateBack);
   patterns.insert<SimpleUnOpToFuncLowering<mlir::db::DateExtractOp, mlir::db::DateType, mlir::db::IntType>>(
      &getContext(), ensureDate64, dateExtractFunction, identity);

   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
