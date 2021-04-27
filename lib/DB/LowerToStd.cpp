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
            auto globalop = rewriter.create<mlir::memref::GlobalOp>(rewriter.getUnknownLoc(), "abc", rewriter.getStringAttr("private"), strStaticType, initialValue, true);
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
   Value getValue(Value v) {
      Type type = v.getType();
      if (auto dbType = type.dyn_cast_or_null<mlir::db::DBType>()) {
         if (dbType.isNullable()) {
            TupleType tupleType = typeConverter.convertType(v.getType()).dyn_cast_or_null<TupleType>();
            auto splitOp = builder.create<mlir::util::SplitOp>(builder.getUnknownLoc(), tupleType.getTypes(), v);
            nullValues.push_back(splitOp.vals()[0]);
            return splitOp.vals()[1];
         } else {
            return v;
         }
      } else {
         return v;
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
         isNull = builder.create<mlir::AndOp>(builder.getUnknownLoc(), isNull.getType(), isNull, nullValues[i]);
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
template<class DBOp,class Op>
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
   patterns.insert<DecimalOpScaledLowering<mlir::db::DivOp,mlir::SignedDivIOp>>(typeConverter, &getContext());
   patterns.insert<DecimalOpScaledLowering<mlir::db::ModOp,mlir::SignedRemIOp>>(typeConverter, &getContext());

   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
