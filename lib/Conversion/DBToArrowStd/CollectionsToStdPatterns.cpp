#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
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
      auto sortOp = cast<mlir::db::SortOp>(op);
      auto ptrType = mlir::util::RefType::get(getContext(), IntegerType::get(getContext(), 8));

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      Type elementType = sortOp.toSort().getType().cast<mlir::db::VectorType>().getElementType();
      FuncOp funcOp;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         funcOp = rewriter.create<FuncOp>(parentModule.getLoc(), "db_sort_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({ptrType, ptrType}), TypeRange(rewriter.getI1Type())));
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({ptrType, ptrType}), {parentModule->getLoc(), parentModule->getLoc()});
         funcOp.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value left = funcBody->getArgument(0);
         Value right = funcBody->getArgument(1);

         Value genericMemrefLeft = rewriter.create<util::GenericMemrefCastOp>(sortOp.getLoc(), util::RefType::get(rewriter.getContext(), elementType), left);
         Value genericMemrefRight = rewriter.create<util::GenericMemrefCastOp>(sortOp.getLoc(), util::RefType::get(rewriter.getContext(), elementType), right);
         Value tupleLeft = rewriter.create<util::LoadOp>(sortOp.getLoc(), elementType, genericMemrefLeft, Value());
         Value tupleRight = rewriter.create<util::LoadOp>(sortOp.getLoc(), elementType, genericMemrefRight, Value());
         auto terminator = rewriter.create<mlir::ReturnOp>(sortOp.getLoc());
         Block* sortLambda = &sortOp.region().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.mergeBlockBefore(sortLambda, terminator, {tupleLeft, tupleRight});
         mlir::db::YieldOp yieldOp = mlir::cast<mlir::db::YieldOp>(terminator->getPrevNode());
         Value x = yieldOp.results()[0];
         rewriter.create<mlir::ReturnOp>(sortOp.getLoc(), x);
         rewriter.eraseOp(sortLambdaTerminator);
         rewriter.eraseOp(terminator);
      }
      Type typedPtrType = util::RefType::get(rewriter.getContext(), typeConverter->convertType(elementType));
      auto loaded = rewriter.create<mlir::util::LoadOp>(op->getLoc(), mlir::TupleType::get(rewriter.getContext(), TypeRange({rewriter.getIndexType(), rewriter.getIndexType(), typedPtrType})), sortOpAdaptor.toSort(), Value());
      auto unpacked = rewriter.create<util::UnPackOp>(sortOp.getLoc(), loaded);

      auto len = unpacked.getResult(0);
      auto values = unpacked.getResult(2);
      auto rawPtr = rewriter.create<util::GenericMemrefCastOp>(sortOp.getLoc(), ptrType, values);

      Value functionPointer = rewriter.create<mlir::ConstantOp>(sortOp->getLoc(), funcOp.type(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.sym_name())));
      Value elementSize = rewriter.create<util::SizeOfOp>(sortOp.getLoc(), rewriter.getIndexType(), elementType);
      functionRegistry.call(rewriter, sortOp->getLoc(), FunctionId::SortVector, {len, rawPtr, elementSize, functionPointer});
      rewriter.eraseOp(op);
      return success();
   }
};
class ForOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit ForOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ForOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}
   std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder) const {
      for (size_t i = 0; i < values.size(); i++) {
         values[i] = builder.getRemappedValue(values[i]);
      }
      return values;
   }

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::ForOpAdaptor forOpAdaptor(operands, op->getAttrDictionary());
      auto forOp = cast<mlir::db::ForOp>(op);
      std::vector<Type> argumentTypes;
      std::vector<Location> argumentLocs;
      for (auto t : forOp.region().getArgumentTypes()) {
         argumentTypes.push_back(t);
         argumentLocs.push_back(op->getLoc());
      }
      auto collectionType = forOp.collection().getType().dyn_cast_or_null<mlir::db::CollectionType>();
      auto iterator = mlir::db::CollectionIterationImpl::getImpl(collectionType, forOp.collection(), functionRegistry);

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      bool containsCondSkip = false;
      for (auto& op : forOp.region().front().getOperations()) {
         containsCondSkip |= mlir::isa<mlir::db::CondSkipOp>(&op);
      }

      using fn_t = std::function<std::vector<Value>(std::function<Value(OpBuilder&)>, ValueRange, OpBuilder)>;
      fn_t fn1 = [&](std::function<Value(OpBuilder & b)> getElem, ValueRange iterargs, OpBuilder builder) {
         auto yieldOp = cast<mlir::db::YieldOp>(forOp.getBody()->getTerminator());
         std::vector<Type> resTypes;
         std::vector<Location> locs;
         for (auto t : yieldOp.results()) {
            resTypes.push_back(typeConverter->convertType(t.getType()));
            locs.push_back(op->getLoc());
         }
         auto execRegion = builder.create<mlir::scf::ExecuteRegionOp>(op->getLoc(), resTypes);
         auto* execRegionBlock = new Block();
         execRegion.getRegion().push_back(execRegionBlock);
         {
            OpBuilder::InsertionGuard guard(builder);
            OpBuilder::InsertionGuard guard2(rewriter);

            builder.setInsertionPointToStart(execRegionBlock);
            std::vector<Value> values;
            values.push_back(getElem(builder));
            values.insert(values.end(), iterargs.begin(), iterargs.end());
            auto term = builder.create<mlir::scf::YieldOp>(op->getLoc());
            builder.setInsertionPoint(term);
            rewriter.mergeBlockBefore(forOp.getBody(), &*builder.getInsertionPoint(), values);

            std::vector<Value> results(yieldOp.results().begin(), yieldOp.results().end());
            rewriter.eraseOp(yieldOp);
            auto term2 = builder.create<mlir::scf::YieldOp>(op->getLoc(), remap(results, rewriter));

            Block* end = rewriter.splitBlock(builder.getBlock(), builder.getInsertionPoint());
            builder.setInsertionPoint(term2);
            builder.create<mlir::cf::BranchOp>(op->getLoc(), end, remap(results, rewriter));
            rewriter.eraseOp(term2);

            end->addArguments(resTypes, locs);
            builder.setInsertionPointToEnd(end);
            builder.create<mlir::scf::YieldOp>(op->getLoc(), end->getArguments());
            std::vector<Operation*> toErase;
            for (auto it = execRegionBlock->getOperations().rbegin(); it != execRegionBlock->getOperations().rend(); it++) {
               if (auto op = mlir::dyn_cast_or_null<mlir::db::CondSkipOp>(&*it)) {
                  toErase.push_back(op.getOperation());
                  builder.setInsertionPointAfter(op);
                  llvm::SmallVector<mlir::Value> remappedArgs;
                  assert(rewriter.getRemappedValues(op.args(), remappedArgs).succeeded());
                  Block* after = rewriter.splitBlock(builder.getBlock(), builder.getInsertionPoint());
                  builder.setInsertionPointAfter(op);
                  auto cond = rewriter.getRemappedValue(op.condition());
                  if (auto tupleType = cond.getType().dyn_cast_or_null<mlir::TupleType>()) {
                     auto i1Type = builder.getI1Type();
                     auto unpacked = builder.create<util::UnPackOp>(op->getLoc(), cond);
                     Value constTrue = builder.create<arith::ConstantOp>(op->getLoc(), i1Type, builder.getIntegerAttr(i1Type, 1));
                     auto negated = builder.create<arith::XOrIOp>(op->getLoc(), unpacked.getResult(0), constTrue); //negate
                     cond = builder.create<arith::AndIOp>(op->getLoc(), i1Type, negated, unpacked.getResult(1));
                  }
                  builder.create<mlir::cf::CondBranchOp>(op->getLoc(), cond, end, remappedArgs, after, ValueRange());
               }
            }
            for (auto* x : toErase) {
               rewriter.eraseOp(x);
            }
            rewriter.eraseOp(term);
         }
         //yieldOp->erase();
         assert(execRegion.getNumResults() == resTypes.size());
         std::vector<Value> results(execRegion.getResults().begin(), execRegion.getResults().end());

         return results;
      };
      fn_t fn2 = [&](std::function<Value(OpBuilder & b)> getElem, ValueRange iterargs, OpBuilder builder) {
         auto yieldOp = cast<mlir::db::YieldOp>(forOp.getBody()->getTerminator());
         std::vector<Type> resTypes;
         std::vector<Location> locs;
         for (auto t : yieldOp.results()) {
            resTypes.push_back(typeConverter->convertType(t.getType()));
            locs.push_back(op->getLoc());
         }
         std::vector<Value> values;
         values.push_back(getElem(builder));
         values.insert(values.end(), iterargs.begin(), iterargs.end());
         auto term = builder.create<mlir::scf::YieldOp>(op->getLoc());
         builder.setInsertionPoint(term);
         rewriter.mergeBlockBefore(forOp.getBody(), &*builder.getInsertionPoint(), values);

         std::vector<Value> results(yieldOp.results().begin(), yieldOp.results().end());
         rewriter.eraseOp(yieldOp);
         rewriter.eraseOp(term);

         return results;
      };

      std::vector<Value> results = iterator->implementLoop(op->getLoc(), forOpAdaptor.initArgs(), forOp.until(), *typeConverter, rewriter, parentModule, containsCondSkip ? fn1 : fn2);
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);

         forOp.region().push_back(new Block());
         forOp.region().front().addArguments(argumentTypes, argumentLocs);
         rewriter.setInsertionPointToStart(&forOp.region().front());
         rewriter.create<mlir::db::YieldOp>(forOp.getLoc());
      }

      rewriter.replaceOp(op, results);
      return success();
   }
};

class LookupOpLowering : public ConversionPattern {
   public:
   explicit LookupOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::Lookup::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      mlir::db::LookupAdaptor lookupAdaptor(operands);
      auto loaded = rewriter.create<util::LoadOp>(loc, lookupAdaptor.collection().getType().cast<mlir::util::RefType>().getElementType(), lookupAdaptor.collection(), Value());
      auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, loaded);
      Value ht = unpacked.getResult(0);
      Value htMask = unpacked.getResult(1);
      Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), lookupAdaptor.key()); //hash key value
      Value buckedPos = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      Value ptr = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(ht.getType()).cast<mlir::util::RefType>().getElementType(), ht, buckedPos);
      //optimization
      ptr = rewriter.create<mlir::util::FilterTaggedPtr>(loc, ptr.getType(), ptr, hashed);
      Value packed = rewriter.create<mlir::util::PackOp>(loc, ValueRange{ptr, hashed});
      rewriter.replaceOp(op, packed);
      return success();
   }
};
} // namespace

void mlir::db::populateCollectionsToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   auto* context = patterns.getContext();

   patterns.insert<SortOpLowering>(functionRegistry, typeConverter, context);

   patterns.insert<ForOpLowering>(functionRegistry, typeConverter, context);
   patterns.insert<LookupOpLowering>(typeConverter, context);

   auto indexType = IndexType::get(context);
   auto i8ptrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
   auto i8PtrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
   typeConverter.addConversion([&typeConverter, context, indexType, i8ptrType](mlir::db::AggregationHashtableType aggregationHashtableType) {
      if (aggregationHashtableType.getKeyType().getTypes().empty()) {
         return (Type) typeConverter.convertType(aggregationHashtableType.getValType());
      } else {
         Type kvType = typeConverter.convertType(TupleType::get(context, {aggregationHashtableType.getKeyType(), aggregationHashtableType.getValType()}));

         Type entryType = TupleType::get(context, {i8ptrType, indexType, kvType});
         auto entryPtrType = mlir::util::RefType::get(context, entryType);

         auto vecType = mlir::util::RefType::get(context, entryType);
         auto htType = mlir::util::RefType::get(context, entryPtrType);
         auto t = (Type) util::RefType::get(context, TupleType::get(context, {htType,indexType, indexType, vecType, indexType, typeConverter.convertType(aggregationHashtableType.getValType())}));
         return t;
      }
   });

   typeConverter.addConversion([&typeConverter, indexType, i8PtrType, context](mlir::db::JoinHashtableType joinHashtableType) {
      Type kvType = typeConverter.convertType(TupleType::get(context, {joinHashtableType.getKeyType(), joinHashtableType.getValType()}));
      Type entryType = TupleType::get(context, {i8PtrType, kvType});

      auto vecType = mlir::util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, mlir::util::RefType::get(context, entryType));
      return (Type) util::RefType::get(context, TupleType::get(context, {htType, indexType, indexType, indexType, vecType}));
   });

   typeConverter.addConversion([context, &typeConverter, indexType](mlir::db::VectorType vectorType) {
      Type entryType = typeConverter.convertType(vectorType.getElementType());
      auto arrayType = mlir::util::RefType::get(context, entryType);
      return (Type) mlir::util::RefType::get(context, TupleType::get(context, {indexType, indexType, arrayType}));
   });
   typeConverter.addConversion([&typeConverter, context, i8ptrType, indexType](mlir::db::GenericIterableType genericIterableType) {
      Type elementType = genericIterableType.getElementType();
      Type nestedElementType = elementType;
      if (auto nested = elementType.dyn_cast_or_null<mlir::db::GenericIterableType>()) {
         nestedElementType = nested.getElementType();
      }
      if (genericIterableType.getIteratorName() == "table_chunk_iterator") {
         std::vector<Type> types;
         types.push_back(i8ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) mlir::util::RefType::get(context, TupleType::get(context, types));
      } else if (genericIterableType.getIteratorName() == "table_row_iterator") {
         std::vector<Type> types;
         types.push_back(i8ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) TupleType::get(context, types);
      } else if (genericIterableType.getIteratorName() == "join_ht_iterator") {
         auto ptrType = mlir::util::RefType::get(context, typeConverter.convertType(TupleType::get(context, {i8ptrType, genericIterableType.getElementType()})));
         return (Type) TupleType::get(context, {ptrType, indexType});
      } else if (genericIterableType.getIteratorName() == "join_ht_mod_iterator") {
         auto types = genericIterableType.getElementType().cast<mlir::TupleType>().getTypes();
         auto ptrType = mlir::util::RefType::get(context, typeConverter.convertType(TupleType::get(context, {i8ptrType, types[0]})));
         return (Type) TupleType::get(context, {ptrType, indexType});
      }
      return Type();
   });
}