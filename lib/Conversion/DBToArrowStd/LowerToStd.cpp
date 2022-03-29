#include "mlir-support/parsing.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Dialect/DSA/IR/DSAOps.h>
#include <mlir/IR/BuiltinTypes.h>

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
   public:
   explicit SimpleTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, Op::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type> convertedTypes;
      assert(typeConverter->convertTypes(op->getResultTypes(), convertedTypes).succeeded());
      rewriter.replaceOpWithNewOp<Op>(op, convertedTypes, ValueRange(operands), op->getAttrs());
      return success();
   }
};
class AtLowering : public ConversionPattern {
   public:
   explicit AtLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::At::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      mlir::dsa::AtAdaptor adaptor(operands);
      auto atOp = mlir::cast<mlir::dsa::At>(op);
      auto t = atOp.getType(0);
      if (typeConverter->isLegal(t)) {
         rewriter.startRootUpdate(op);
         atOp->setOperands(operands);
         rewriter.finalizeRootUpdate(op);
         return mlir::success();
      }
      auto context = getContext();
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
      } else {
         std::cout << "?" << std::endl;
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
      rewriter.replaceOp(op, values);
      return success();
   }
};
class ConvertForOpTypes : public OpConversionPattern<mlir::dsa::ForOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult
   matchAndRewrite(mlir::dsa::ForOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const override {
      SmallVector<Type, 6> newResultTypes;
      for (auto type : op.getResultTypes()) {
         Type newType = typeConverter->convertType(type);
         if (!newType)
            return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
         newResultTypes.push_back(newType);
      }

      // Clone the op without the regions and inline the regions from the old op.
      //
      // This is a little bit tricky. We have two concerns here:
      //
      // 1. We cannot update the op in place because the dialect conversion
      // framework does not track type changes for ops updated in place, so it
      // won't insert appropriate materializations on the changed result types.
      // PR47938 tracks this issue, but it seems hard to fix. Instead, we need to
      // clone the op.
      //
      // 2. We cannot simply call `op.clone()` to get the cloned op. Besides being
      // inefficient to recursively clone the regions, there is a correctness
      // issue: if we clone with the regions, then the dialect conversion
      // framework thinks that we just inserted all the cloned child ops. But what
      // we want is to "take" the child regions and let the dialect conversion
      // framework continue recursively into ops inside those regions (which are
      // already in its worklist; inlining them into the new op's regions doesn't
      // remove the child ops from the worklist).
      mlir::dsa::ForOp newOp = cast<mlir::dsa::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      // Take the region from the old op and put it in the new op.

      rewriter.inlineRegionBefore(op.getBodyRegion(), newOp.getBodyRegion(),
                                  newOp.getBodyRegion().end());
      op.region().push_back(new Block);
      {
         std::vector<mlir::Location> locs;
         for (auto x : newOp.getBody()->getArgumentTypes()) {
            locs.push_back(rewriter.getUnknownLoc());
         }
         op.getBody()->addArguments(newOp.getBody()->getArgumentTypes(), locs);
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(op.getBody());
         rewriter.create<mlir::dsa::YieldOp>(rewriter.getUnknownLoc());
         op.dump();
      }
      // Now, update all the types.

      // Convert the type of the entry block of the ForOp's body.
      if (failed(rewriter.convertRegionTypes(&newOp.getBodyRegion(), *getTypeConverter()))) {
         return rewriter.notifyMatchFailure(op, "could not convert body types");
      }
      // Change the clone to use the updated operands. We could have cloned with
      // a BlockAndValueMapping, but this seems a bit more direct.
      newOp->setOperands(adaptor.getOperands());
      // Update the result types to the new converted types.
      for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
         std::get<0>(t).setType(std::get<1>(t));

      rewriter.replaceOp(op, newOp.getResults());
      return success();
   }
};
class ConvertSortOpTypes : public OpConversionPattern<mlir::dsa::SortOp> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult
   matchAndRewrite(mlir::dsa::SortOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const override {
      // Clone the op without the regions and inline the regions from the old op.
      //
      // This is a little bit tricky. We have two concerns here:
      //
      // 1. We cannot update the op in place because the dialect conversion
      // framework does not track type changes for ops updated in place, so it
      // won't insert appropriate materializations on the changed result types.
      // PR47938 tracks this issue, but it seems hard to fix. Instead, we need to
      // clone the op.
      //
      // 2. We cannot simply call `op.clone()` to get the cloned op. Besides being
      // inefficient to recursively clone the regions, there is a correctness
      // issue: if we clone with the regions, then the dialect conversion
      // framework thinks that we just inserted all the cloned child ops. But what
      // we want is to "take" the child regions and let the dialect conversion
      // framework continue recursively into ops inside those regions (which are
      // already in its worklist; inlining them into the new op's regions doesn't
      // remove the child ops from the worklist).
      mlir::dsa::SortOp newOp = cast<mlir::dsa::SortOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      // Take the region from the old op and put it in the new op.

      rewriter.inlineRegionBefore(op.region(), newOp.region(),
                                  newOp.region().end());
      op.region().push_back(new Block);
      {
         std::vector<mlir::Location> locs;
         for (auto x : newOp.region().front().getArgumentTypes()) {
            locs.push_back(rewriter.getUnknownLoc());
         }
         op.region().front().addArguments(newOp.getRegion().front().getArgumentTypes(), locs);
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(&op.region().front());
         rewriter.create<mlir::dsa::YieldOp>(rewriter.getUnknownLoc());
         op.dump();
      }
      // Now, update all the types.

      // Convert the type of the entry block of the ForOp's body.
      if (failed(rewriter.convertRegionTypes(&newOp.region(), *getTypeConverter()))) {
         return rewriter.notifyMatchFailure(op, "could not convert body types");
      }
      // Change the clone to use the updated operands. We could have cloned with
      // a BlockAndValueMapping, but this seems a bit more direct.
      newOp->setOperands(adaptor.getOperands());
      // Update the result types to the new converted types.

      rewriter.eraseOp(op);
      return success();
   }
};

class ConvertInsertReduceTypes : public OpConversionPattern<mlir::dsa::HashtableInsertReduce> {
   public:
   using OpConversionPattern::OpConversionPattern;
   LogicalResult
   matchAndRewrite(mlir::dsa::HashtableInsertReduce op, OpAdaptor adaptor,
                   ConversionPatternRewriter& rewriter) const override {
      // Clone the op without the regions and inline the regions from the old op.
      //
      // This is a little bit tricky. We have two concerns here:
      //
      // 1. We cannot update the op in place because the dialect conversion
      // framework does not track type changes for ops updated in place, so it
      // won't insert appropriate materializations on the changed result types.
      // PR47938 tracks this issue, but it seems hard to fix. Instead, we need to
      // clone the op.
      //
      // 2. We cannot simply call `op.clone()` to get the cloned op. Besides being
      // inefficient to recursively clone the regions, there is a correctness
      // issue: if we clone with the regions, then the dialect conversion
      // framework thinks that we just inserted all the cloned child ops. But what
      // we want is to "take" the child regions and let the dialect conversion
      // framework continue recursively into ops inside those regions (which are
      // already in its worklist; inlining them into the new op's regions doesn't
      // remove the child ops from the worklist).
      mlir::dsa::HashtableInsertReduce newOp = cast<mlir::dsa::HashtableInsertReduce>(rewriter.cloneWithoutRegions(*op.getOperation()));
      // Take the region from the old op and put it in the new op.

      rewriter.inlineRegionBefore(op.equal(), newOp.equal(), newOp.equal().end());
      rewriter.inlineRegionBefore(op.reduce(), newOp.reduce(), newOp.reduce().end());
      {
         if (!newOp.equal().empty()) {
            op.equal().push_back(new Block);
            std::vector<mlir::Location> locs;
            for (auto x : newOp.equal().front().getArgumentTypes()) {
               locs.push_back(rewriter.getUnknownLoc());
            }
            op.equal().front().addArguments(newOp.equal().front().getArgumentTypes(), locs);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&op.equal().front());
            rewriter.create<mlir::dsa::YieldOp>(rewriter.getUnknownLoc());
         }
      }
      {
         if (!newOp.reduce().empty()) {
            op.reduce().push_back(new Block);
            std::vector<mlir::Location> locs;
            for (auto x : newOp.reduce().front().getArgumentTypes()) {
               locs.push_back(rewriter.getUnknownLoc());
            }
            op.reduce().front().addArguments(newOp.reduce().front().getArgumentTypes(), locs);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&op.reduce().front());
            rewriter.create<mlir::dsa::YieldOp>(rewriter.getUnknownLoc());
         }
      }
      // Now, update all the types.

      // Convert the type of the entry block of the ForOp's body.
      if (failed(rewriter.convertRegionTypes(&newOp.equal(), *getTypeConverter()))) {
         return rewriter.notifyMatchFailure(op, "could not convert body types");
      }
      if (failed(rewriter.convertRegionTypes(&newOp.reduce(), *getTypeConverter()))) {
         return rewriter.notifyMatchFailure(op, "could not convert body types");
      }
      // Change the clone to use the updated operands. We could have cloned with
      // a BlockAndValueMapping, but this seems a bit more direct.
      newOp->setOperands(adaptor.getOperands());
      // Update the result types to the new converted types.

      rewriter.eraseOp(op);
      return success();
   }
};
void DBToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   mlir::db::codegen::FunctionRegistry functionRegistry(module);
   functionRegistry.registerFunctions();

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<StandardOpsDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   TypeConverter typeConverter;

   auto opIsWithoutDBTypes = [&](Operation* op) { return !hasDBType(typeConverter, op->getOperandTypes()) && !hasDBType(typeConverter, op->getResultTypes()); };
   target.addDynamicallyLegalDialect<scf::SCFDialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<dsa::DSADialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<arith::ArithmeticDialect>(opIsWithoutDBTypes);

   target.addLegalDialect<cf::ControlFlowDialect>();

   target.addDynamicallyLegalDialect<util::UtilDialect>(opIsWithoutDBTypes);
   target.addLegalOp<mlir::dsa::CondSkipOp>();

   target.addDynamicallyLegalOp<mlir::dsa::CondSkipOp>(opIsWithoutDBTypes);
   target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto isLegal = !hasDBType(typeConverter, op.getType().getInputs()) &&
         !hasDBType(typeConverter, op.getType().getResults());
      return isLegal;
   });
   target.addDynamicallyLegalOp<ConstantOp>([&](ConstantOp op) {
      if (auto functionType = op.getType().dyn_cast_or_null<mlir::FunctionType>()) {
         auto isLegal = !hasDBType(typeConverter, functionType.getInputs()) &&
            !hasDBType(typeConverter, functionType.getResults());
         return isLegal;
      } else {
         return true;
      }
   });
   target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(opIsWithoutDBTypes);

   target.addDynamicallyLegalOp<util::SizeOfOp>(
      [&typeConverter](util::SizeOfOp op) {
         auto isLegal = !hasDBType(typeConverter, op.type());
         return isLegal;
      });

   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::db::TableType tableType) {
      return mlir::util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](mlir::IntegerType iType) { return iType; });
   typeConverter.addConversion([&](mlir::IndexType iType) { return iType; });
   typeConverter.addConversion([&](mlir::dsa::FlagType flagType) { return flagType; });
   typeConverter.addConversion([&](mlir::FloatType fType) { return fType; });
   typeConverter.addConversion([&](mlir::MemRefType refType) { return refType; });
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
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::DateType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, dsa::RecordBatchType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addTargetMaterialization([&](OpBuilder&, dsa::RecordBatchType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::DecimalType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::CharType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::StringType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TimestampType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::IntervalType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, IntegerType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::NullableType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::NullableType type, ValueRange valueRange, Location loc) { return valueRange.front(); });
   typeConverter.addSourceMaterialization([&](OpBuilder&, TupleType type, ValueRange valueRange, Location loc) { return valueRange.front(); });

   RewritePatternSet patterns(&getContext());

   mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::FuncOp>(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   mlir::db::populateScalarToStdPatterns(typeConverter, patterns);
   mlir::db::populateRuntimeSpecificScalarToStdPatterns(functionRegistry, typeConverter, patterns);
   mlir::db::populateBuilderToStdPatterns(functionRegistry, typeConverter, patterns);
   mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
   patterns.insert<SimpleTypeConversionPattern<mlir::ConstantOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::CondSkipOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::ScanSource>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Append>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::CreateDS>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::HashtableInsert>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::HashtableFinalize>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Lookup>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::FreeOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::YieldOp>>(typeConverter, &getContext());
   patterns.insert<AtLowering>(typeConverter, &getContext());
   patterns.insert<ConvertForOpTypes>(typeConverter, &getContext());
   patterns.insert<ConvertSortOpTypes>(typeConverter, &getContext());
   patterns.insert<ConvertInsertReduceTypes>(typeConverter, &getContext());
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass>
mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
