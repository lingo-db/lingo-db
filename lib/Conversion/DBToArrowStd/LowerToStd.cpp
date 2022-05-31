#include "mlir-support/parsing.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
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
               locs.push_back(rewriter.getUnknownLoc());
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
   typeConverter.addConversion([&](mlir::Type type) { return type; });

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
   mlir::db::populateScalarToStdPatterns(typeConverter, patterns);
   mlir::db::populateRuntimeSpecificScalarToStdPatterns(typeConverter, patterns);
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