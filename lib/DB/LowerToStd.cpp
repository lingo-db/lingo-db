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
#include "mlir/Dialect/util/UtilDialect.h"
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
            if(arrow::Decimal128::FromString(strAttr.getValue().str(), &decimalrep, &precision, &scale)!=arrow::Status::OK()){
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
template <class OpClass, class OperandType, class StdOpClass>
class BinOpLowering : public ConversionPattern {
   public:
   explicit BinOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, OpClass::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto addOp = cast<OpClass>(op);
      Value left = addOp.lhs();
      Value right = addOp.rhs();
      if (left.getType() != right.getType()) {
         return failure();
      }
      auto type = left.getType();
      auto resType = addOp.result().getType();
      if (type.isa<OperandType>()) {
         rewriter.replaceOpWithNewOp<StdOpClass>(op, typeConverter->convertType(resType), left, right);
         return success();
      }
      return failure();
   }
};
class DecimalDivOpLowering : public ConversionPattern {
   public:
   explicit DecimalDivOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::DivOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto addOp = cast<mlir::db::DivOp>(op);
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
         rewriter.replaceOpWithNewOp<SignedDivIOp>(op, stdType, left, right);
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
      auto loc = op->getLoc();
      auto printOp = cast<mlir::db::DumpOp>(op);
      Value val = printOp.val();
      auto i128Type = IntegerType::get(rewriter.getContext(), 128);
      auto i64Type = IntegerType::get(rewriter.getContext(), 64);
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);
      auto i1Type = IntegerType::get(rewriter.getContext(), 1);
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);

      auto f64Type = FloatType::getF64(rewriter.getContext());

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      // Get a symbol reference to the printf function, inserting it if necessary.
      if (auto dbIntType = val.getType().dyn_cast_or_null<mlir::db::IntType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpInt", rewriter.getFunctionType({i64Type}, {}));
         if (dbIntType.getWidth() < 64) {
            val = rewriter.create<ZeroExtendIOp>(loc, printOp.val(), i64Type);
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({val}));
      } else if (val.getType().isa<mlir::db::BoolType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpBool", rewriter.getFunctionType({i1Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({val}));
      } else if (auto decType = val.getType().dyn_cast_or_null<mlir::db::DecimalType>()) {
         Value low = rewriter.create<TruncateIOp>(loc, printOp.val(), i64Type);
         Value shift = rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
         Value scale = rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
         Value high = rewriter.create<UnsignedShiftRightOp>(loc, i128Type, val, shift);
         high = rewriter.create<TruncateIOp>(loc, high, i64Type);

         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpDecimal", rewriter.getFunctionType({i64Type, i64Type, i32Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({low, high, scale}));
      } else if (val.getType().isa<mlir::db::DateType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpDate", rewriter.getFunctionType({i32Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({val}));
      } else if (val.getType().isa<mlir::db::TimestampType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpTimestamp", rewriter.getFunctionType({i64Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({val}));
      } else if (val.getType().isa<mlir::db::IntervalType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpInterval", rewriter.getFunctionType({i64Type}, {}));
         rewriter.create<CallOp>(loc, printRef, ValueRange({val}));
      } else if (auto floatType = val.getType().dyn_cast_or_null<mlir::db::FloatType>()) {
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpFloat", rewriter.getFunctionType({f64Type}, {}));
         if (floatType.getWidth() < 64) {
            val = rewriter.create<FPExtOp>(loc, val, f64Type);
         }
         rewriter.create<CallOp>(loc, printRef, ValueRange({val}));
      } else if (val.getType().isa<mlir::db::StringType>()) {
         auto strType = MemRefType::get({}, IntegerType::get(getContext(), 8));
         auto printRef = getOrInsertFn(rewriter, parentModule, "dumpString", rewriter.getFunctionType({strType, i64Type}, {}));
         Value len = rewriter.create<memref::DimOp>(loc, val, 0);
         len = rewriter.create<IndexCastOp>(loc, len, i64Type);
         val = rewriter.create<memref::ReinterpretCastOp>(loc, MemRefType::get({}, i8Type), val, (int64_t) 0, ArrayRef<int64_t>({}), ArrayRef<int64_t>({}));

         rewriter.create<CallOp>(loc, printRef, ValueRange({val, len}));
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
   target.addLegalOp<FuncOp>();
   target.addLegalDialect<StandardOpsDialect>();
   target.addLegalDialect<memref::MemRefDialect>();

   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();

   //Add own types to LLVMTypeConverter
   TypeConverter typeConverter;
   typeConverter.addConversion([&](mlir::db::DBType type) {
      return ::llvm::TypeSwitch<::mlir::db::DBType, mlir::Type>(type)
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
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   OwningRewritePatternList patterns(&getContext());

   // Add own Lowering Patterns
   patterns.insert<DumpOpLowering>(typeConverter, &getContext());
   patterns.insert<ConstantLowering>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::IntType, mlir::AddIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::FloatType, mlir::AddFOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::db::DecimalType, mlir::AddIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::IntType, mlir::SubIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::FloatType, mlir::SubFOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::db::DecimalType, mlir::SubIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::IntType, mlir::MulIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::FloatType, mlir::MulFOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::db::DecimalType, mlir::MulIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::IntType, mlir::SignedDivIOp>>(typeConverter, &getContext());
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::db::FloatType, mlir::MulFOp>>(typeConverter, &getContext());
   patterns.insert<DecimalDivOpLowering>(typeConverter, &getContext());
   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
