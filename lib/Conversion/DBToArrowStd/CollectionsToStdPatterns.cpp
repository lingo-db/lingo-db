#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/DBToArrowStd/SerializationUtil.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

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
      Value allocaVec, allocaNewVec;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         auto func = op->getParentOfType<mlir::FuncOp>();
         rewriter.setInsertionPointToStart(&func.getBody().front());
         allocaVec = rewriter.create<mlir::util::AllocaOp>(sortOp->getLoc(), vectorMemrefType, Value());
         allocaNewVec = rewriter.create<mlir::util::AllocaOp>(sortOp->getLoc(), vectorMemrefType, Value());
      }
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
class ForOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit ForOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ForOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::ForOpAdaptor forOpAdaptor(operands, op->getAttrDictionary());
      auto forOp = cast<mlir::db::ForOp>(op);
      auto argumentTypes = forOp.region().getArgumentTypes();
      auto collectionType = forOp.collection().getType().dyn_cast_or_null<mlir::db::CollectionType>();

      auto iterator = mlir::db::CollectionIterationImpl::getImpl(collectionType, forOp.collection(), functionRegistry);

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      std::vector<Value> results = iterator->implementLoop(forOpAdaptor.initArgs(), forOp.until(), *typeConverter, rewriter, parentModule, [&](ValueRange values, OpBuilder builder) {
         auto yieldOp = cast<mlir::db::YieldOp>(forOp.getBody()->getTerminator());
         rewriter.mergeBlockBefore(forOp.getBody(), &*builder.getInsertionPoint(), values);
         std::vector<Value> results(yieldOp.results().begin(), yieldOp.results().end());
         rewriter.eraseOp(yieldOp);
         return results;
      });
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         forOp.region().push_back(new Block());
         forOp.region().front().addArguments(argumentTypes);
         rewriter.setInsertionPointToStart(&forOp.region().front());
         rewriter.create<mlir::db::YieldOp>(forOp.getLoc());
      }

      rewriter.replaceOp(op, results);
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
class LookupOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;
   public:
   explicit LookupOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::Lookup::getOperationName(), 1, context),functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = rewriter.getUnknownLoc();
      mlir::db::LookupAdaptor lookupAdaptor(operands);

      Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), lookupAdaptor.key());
      Value rawValues=functionRegistry.call(rewriter,db::codegen::FunctionRegistry::FunctionId::JoinHtGetRawData,lookupAdaptor.collection())[0];
      Value combined = rewriter.create<mlir::util::PackOp>(loc, TypeRange(TupleType::get(getContext(), {rewriter.getIndexType(),lookupAdaptor.collection().getType(),rawValues.getType()})), ValueRange({hashed,lookupAdaptor.collection(),rawValues}));

      rewriter.replaceOp(op, combined);

      return success();
   }
};
} // namespace

void mlir::db::populateCollectionsToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<SortOpLowering>(functionRegistry, typeConverter, patterns.getContext());

   patterns.insert<ForOpLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateRangeLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<LookupOpLowering>(functionRegistry, typeConverter, patterns.getContext());

   typeConverter.addSourceMaterialization([&](OpBuilder&, db::AggregationHashtableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::AggregationHashtableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::JoinHashtableType type, ValueRange valueRange, Location loc) {
       return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::JoinHashtableType type, ValueRange valueRange, Location loc) {
       return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::VectorType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::VectorType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addConversion([&](mlir::db::AggregationHashtableType aggregationHashtableType) {
      auto ptrType = MemRefType::get({}, IntegerType::get(patterns.getContext(), 8));
      return ptrType;
   });
   typeConverter.addConversion([&](mlir::db::JoinHashtableType aggregationHashtableType) {
     auto ptrType = MemRefType::get({}, IntegerType::get(patterns.getContext(), 8));
     return ptrType;
   });
   typeConverter.addConversion([&](mlir::db::VectorType vectorType) {
      auto ptrType = MemRefType::get({-1}, IntegerType::get(patterns.getContext(), 8));
      return TupleType::get(patterns.getContext(), {ptrType, ptrType});
   });
   typeConverter.addConversion([&](mlir::db::RangeType rangeType) {
      auto convertedType = typeConverter.convertType(rangeType.getElementType());
      return TupleType::get(patterns.getContext(), {convertedType, convertedType, convertedType});
   });
   typeConverter.addConversion([&](mlir::db::GenericIterableType genericIterableType) {
      Type elementType = genericIterableType.getElementType();
      Type nestedElementType = elementType;
      if (auto nested = elementType.dyn_cast_or_null<mlir::db::GenericIterableType>()) {
         nestedElementType = nested.getElementType();
      }
      if (genericIterableType.getIteratorName() == "table_chunk_iterator") {
         std::vector<Type> types;
         auto i8Type = IntegerType::get(patterns.getContext(), 8);
         auto ptrType = MemRefType::get({}, i8Type);
         auto indexType = IndexType::get(patterns.getContext());
         types.push_back(ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) TupleType::get(patterns.getContext(), types);
      } else if (genericIterableType.getIteratorName() == "table_row_iterator") {
         std::vector<Type> types;
         auto i8Type = IntegerType::get(patterns.getContext(), 8);
         auto ptrType = MemRefType::get({}, i8Type);
         auto indexType = IndexType::get(patterns.getContext());
         types.push_back(ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) TupleType::get(patterns.getContext(), types);
      } else if (genericIterableType.getIteratorName() == "join_ht_iterator") {
         auto indexType = IndexType::get(patterns.getContext());
         auto i8Type = IntegerType::get(patterns.getContext(), 8);
         auto ptrType = MemRefType::get({}, i8Type);
         auto rawDataType = MemRefType::get({-1}, i8Type);

         return (Type) TupleType::get(patterns.getContext(), {indexType,ptrType,rawDataType});
      }
      return Type();
   });
}