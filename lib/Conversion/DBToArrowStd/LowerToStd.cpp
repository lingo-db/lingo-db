#include "mlir-support/mlir-support.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
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
#include <llvm/Support/Debug.h>
#include <iostream>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {

class ForOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit ForOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ForOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::ForOpAdaptor forOpAdaptor(operands);
      auto forOp = cast<mlir::db::ForOp>(op);
      auto argumentTypes = forOp.region().getArgumentTypes();
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
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         llvm::dbgs() << forOp.region().getBlocks().size() << "\n";
         forOp.region().push_back(new Block());
         forOp.region().front().addArguments(argumentTypes);
         rewriter.setInsertionPointToStart(&forOp.region().front());
         rewriter.create<mlir::db::YieldOp>(forOp.getLoc());
      }

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
   typeConverter.addConversion([&](mlir::db::AggregationHashtableType aggregationHashtableType) {
      auto ptrType = MemRefType::get({}, IntegerType::get(&getContext(), 8));
      return ptrType;
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
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::AggregationHashtableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::AggregationHashtableType type, ValueRange valueRange, Location loc) {
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

   mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   // Add own Lowering Patterns

   patterns.insert<FuncConstLowering>(typeConverter, &getContext());

   patterns.insert<TableScanLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<HashLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<GetTableLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<SortOpLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<ForOpLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateRangeLowering>(functionRegistry, typeConverter, &getContext());

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
