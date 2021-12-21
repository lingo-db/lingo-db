#include "mlir-support/mlir-support.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
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
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;

namespace {

class GetTableLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit GetTableLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::GetTable::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto getTableOp = cast<mlir::db::GetTable>(op);
      auto executionContext = functionRegistry.call(rewriter, op->getLoc(), db::codegen::FunctionRegistry::FunctionId::GetExecutionContext, {})[0];
      auto tableName = rewriter.create<mlir::db::ConstantOp>(op->getLoc(), mlir::db::StringType::get(rewriter.getContext(), false), rewriter.getStringAttr(getTableOp.tablename()));
      Value tableNameRef=rewriter.create<mlir::util::VarLenGetRef>(op->getLoc(),mlir::util::RefType::get(getContext(),rewriter.getI8Type(),-1),tableName);
      auto tablePtr = functionRegistry.call(rewriter, op->getLoc(), db::codegen::FunctionRegistry::FunctionId::ExecutionContextGetTable, mlir::ValueRange({executionContext, tableNameRef}))[0];
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
      auto ptrType = mlir::util::RefType::get(rewriter.getContext(), IntegerType::get(rewriter.getContext(), 8), llvm::Optional<int64_t>());
      auto indexType = IndexType::get(rewriter.getContext());

      std::vector<Value> values;
      types.push_back(ptrType);
      auto tablePtr = adaptor.table();
      values.push_back(tablePtr);
      for (auto c : tablescan.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         types.push_back(indexType);
         auto columnName = rewriter.create<mlir::db::ConstantOp>(op->getLoc(), mlir::db::StringType::get(rewriter.getContext(), false), stringAttr);
         Value columnNameRef=rewriter.create<mlir::util::VarLenGetRef>(op->getLoc(),mlir::util::RefType::get(getContext(),rewriter.getI8Type(),-1),columnName);

         auto columnId = functionRegistry.call(rewriter, op->getLoc(),db::codegen::FunctionRegistry::FunctionId::TableGetColumnId, mlir::ValueRange({tablePtr, columnNameRef}))[0];
         values.push_back(columnId);
      }
      rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, mlir::TupleType::get(rewriter.getContext(), types), values);
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

         rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, convertFunctionType(type, *typeConverter), constantOp.getValue());
         return success();

      } else {
         return failure();
      }
   }
};
class TypeCastLowering : public ConversionPattern {
   public:
   explicit TypeCastLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::TypeCastOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, operands);
      return success();
   }
};
} // end anonymous namespace

namespace {
struct DBToStdLoweringPass
   : public PassWrapper<DBToStdLoweringPass, OperationPass<ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "to-arrow-std"; }

   DBToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithmeticDialect>();
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
      } else if (auto genericMemrefType = type.dyn_cast_or_null<util::RefType>()) {
         res |= hasDBType(genericMemrefType.getElementType());
      } else if (auto functionType = type.dyn_cast_or_null<mlir::FunctionType>()) {
         res |= hasDBType(functionType.getInputs()) ||
            hasDBType(functionType.getResults());
      } else if (type.isa<mlir::db::TableType>() || type.isa<mlir::db::VectorType>() || type.isa<mlir::db::FlagType>()) {
         res = true;
      } else {
         if (type.isa<mlir::db::CollectionType>()) {
            res = true;
         }
      }
   }
   return res;
}
void DBToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   mlir::db::codegen::FunctionRegistry functionRegistry(&getContext());
   functionRegistry.registerFunctions();

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<StandardOpsDialect>();
   target.addLegalDialect<arith::ArithmeticDialect>();

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
   target.addDynamicallyLegalOp<util::DimOp, util::SetTupleOp, util::GetTupleOp, util::UndefTupleOp, util::PackOp, util::UnPackOp, util::ToGenericMemrefOp,util::ToMemrefOp, util::StoreOp, util::LoadOp, util::AllocOp, util::DeAllocOp, util::AllocaOp, util::AllocaOp, util::GenericMemrefCastOp, util::ElementPtrOp>(
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
      return mlir::util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8), llvm::Optional<int64_t>());
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

   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::TableType type, ValueRange valueRange, Location loc) {
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

   mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   mlir::db::populateScalarToStdPatterns(typeConverter, patterns);
   mlir::db::populateControlFlowToStdPatterns(typeConverter, patterns);
   mlir::db::populateRuntimeSpecificScalarToStdPatterns(functionRegistry, typeConverter, patterns);
   mlir::db::populateBuilderToStdPatterns(functionRegistry, typeConverter, patterns);
   mlir::db::populateCollectionsToStdPatterns(functionRegistry, typeConverter, patterns);
   mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);

   patterns.insert<FuncConstLowering>(typeConverter, &getContext());
   patterns.insert<TypeCastLowering>(typeConverter, &getContext());

   patterns.insert<TableScanLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<GetTableLowering>(functionRegistry, typeConverter, &getContext());

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
