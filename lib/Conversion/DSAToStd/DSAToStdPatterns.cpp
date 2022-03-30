#include "mlir/Conversion/DSAToStd/FunctionRegistry.h"

#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {
class VectorHelper2 {
   Type elementType;
   Location loc;

   public:
   static Type createSType(MLIRContext* context, Type elementType) {
      return mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), mlir::util::RefType::get(context, elementType)});
   }
   static mlir::util::RefType createType(MLIRContext* context, Type elementType) {
      return mlir::util::RefType::get(context, createSType(context, elementType));
   }
   VectorHelper2(Type elementType, Location loc) : elementType(elementType), loc(loc) {
   }
   Value create(mlir::OpBuilder& builder, Value initialCapacity, mlir::dsa::codegen::FunctionRegistry& functionRegistry) {
      auto typeSize = builder.create<mlir::util::SizeOfOp>(loc, builder.getIndexType(), elementType);
      auto ptr = functionRegistry.call(builder, loc, mlir::dsa::codegen::FunctionRegistry::FunctionId::VecCreate, ValueRange({typeSize, initialCapacity}))[0];
      return builder.create<mlir::util::GenericMemrefCastOp>(loc, createType(builder.getContext(), elementType), ptr);
   }
   void insert(mlir::OpBuilder& builder, Value vec, Value newVal, mlir::dsa::codegen::FunctionRegistry& functionRegistry) {
      auto idxType = builder.getIndexType();
      auto idxPtrType = util::RefType::get(builder.getContext(), idxType);
      auto valuesType = mlir::util::RefType::get(builder.getContext(), elementType);
      Value lenAddress = builder.create<util::TupleElementPtrOp>(loc, idxPtrType, vec, 0);
      Value capacityAddress = builder.create<util::TupleElementPtrOp>(loc, idxPtrType, vec, 1);

      auto len = builder.create<mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = builder.create<mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      builder.create<scf::IfOp>(
         loc, TypeRange({}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            Value downCasted = b.create<util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(b.getContext(),b.getI8Type()), vec);
            functionRegistry.call(b,loc,mlir::dsa::codegen::FunctionRegistry::FunctionId::VecResize,ValueRange{downCasted});
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = builder.create<util::TupleElementPtrOp>(loc, util::RefType::get(builder.getContext(), valuesType), vec, 2);
      auto values = builder.create<mlir::util::LoadOp>(loc, valuesType, valuesAddress, Value());

      builder.create<util::StoreOp>(loc, newVal, values, len);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = builder.create<arith::AddIOp>(loc, len, one);

      builder.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
   }
};

class AggrHtHelper2 {
   TupleType entryType;
   Location loc;
   Type keyType, aggrType;
   Type oKeyType, oAggrType;

   public:
   static Type kvType(MLIRContext* context, Type keyType, Type aggrType) {
      return mlir::TupleType::get(context, {keyType, aggrType});
   }
   static TupleType createEntryType(MLIRContext* context, Type keyType, Type aggrType) {
      auto i8PtrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
      return mlir::TupleType::get(context, {i8PtrType, IndexType::get(context), kvType(context, keyType, aggrType)});
   }
   static Type createType(MLIRContext* context, Type keyType, Type aggrType) {
      auto idxType = IndexType::get(context);
      auto entryType = createEntryType(context, keyType, aggrType);
      auto entryPtrType = mlir::util::RefType::get(context, entryType);
      auto valuesType = mlir::util::RefType::get(context, entryType);
      auto htType = mlir::util::RefType::get(context, entryPtrType);
      auto tplType = mlir::TupleType::get(context, {htType, idxType, idxType, valuesType, idxType, aggrType});
      return mlir::util::RefType::get(context, tplType);
   }
   AggrHtHelper2(MLIRContext* context, Type keyType, Type aggrType, Location loc, TypeConverter* converter) : entryType(createEntryType(context, converter->convertType(keyType), converter->convertType(aggrType))), loc(loc), keyType(converter->convertType(keyType)), aggrType(converter->convertType(aggrType)), oKeyType(keyType), oAggrType(aggrType) {
   }

   Value create(mlir::OpBuilder& builder, Value initialValue, mlir::dsa::codegen::FunctionRegistry& functionRegistry) {
      auto typeSize = builder.create<mlir::util::SizeOfOp>(loc, builder.getIndexType(), createEntryType(builder.getContext(), keyType, aggrType));
      Value initialCapacity = builder.create<arith::ConstantIndexOp>(loc, 4);
      auto ptr = functionRegistry.call(builder, loc, mlir::dsa::codegen::FunctionRegistry::FunctionId::AggrHtCreate, ValueRange({typeSize, initialCapacity}))[0];
      auto casted = builder.create<mlir::util::GenericMemrefCastOp>(loc, createType(builder.getContext(), keyType, aggrType), ptr);
      Value initValAddress = builder.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(builder.getContext(), initialValue.getType()), casted, 5);
      builder.create<mlir::util::StoreOp>(loc, initialValue, initValAddress, Value());
      return casted;
   }
   void insert(mlir::ConversionPatternRewriter& rewriter, Value aggrHtBuilder, Value key, Value val, Value hash, std::function<Value(mlir::OpBuilder&, Value, Value)> updateFn, std::function<Value(mlir::OpBuilder&, Value, Value)> equalFn, mlir::dsa::codegen::FunctionRegistry& functionRegistry) {
      auto* context = rewriter.getContext();
      auto i8PtrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = AggrHtHelper2::kvType(context, keyType, aggrType);
      auto kvPtrType = mlir::util::RefType::get(context, kvType);
      auto keyPtrType = mlir::util::RefType::get(context, keyType);
      auto aggrPtrType = mlir::util::RefType::get(context, aggrType);
      auto valuesType = mlir::util::RefType::get(context, entryType);
      auto entryPtrType = mlir::util::RefType::get(context, entryType);
      auto htType = mlir::util::RefType::get(context, entryPtrType);

      Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, aggrHtBuilder, 1);
      Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, aggrHtBuilder, 2);

      Value len = rewriter.create<util::LoadOp>(loc, idxType, lenAddress);
      Value capacityInitial = rewriter.create<util::LoadOp>(loc, idxType, capacityAddress);

      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);

      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacityInitial);
      rewriter.create<scf::IfOp>(
         loc, TypeRange(), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            Value downCasted = b.create<util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(b.getContext(),b.getI8Type()), aggrHtBuilder);
            functionRegistry.call(b,loc,mlir::dsa::codegen::FunctionRegistry::FunctionId::AggrHtResize,ValueRange{downCasted});
            b.create<scf::YieldOp>(loc); });

      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(rewriter.getContext(), htType), aggrHtBuilder, 0);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);

      Value capacity = rewriter.create<util::LoadOp>(loc, idxType, capacityAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      Value htSize = rewriter.create<arith::MulIOp>(loc, capacity, two);

      Value htMask = rewriter.create<arith::SubIOp>(loc, htSize, one);
      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hash);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Type ptrType = util::RefType::get(context, bucketPtrType);
      Type doneType = rewriter.getI1Type();
      Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, ptrType, ht, position);

      auto resultTypes = std::vector<Type>({ptrType});
      auto whileOp = rewriter.create<scf::WhileOp>(loc, resultTypes, ValueRange({ptr}));
      Block* before = rewriter.createBlock(&whileOp.getBefore(), {}, resultTypes, {loc});
      Block* after = rewriter.createBlock(&whileOp.getAfter(), {}, resultTypes, {loc});

      // The conditional block of the while loop.
      {
         rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
         Value ptr = before->getArgument(0);

         Value currEntryPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ptr);
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, TypeRange({doneType, ptrType}), cmp,
            [&](OpBuilder& b, Location loc) {

               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hash);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, TypeRange({doneType,ptrType}), hashMatches, [&](OpBuilder& b, Location loc) {
                     Value kvAddress=rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                     Value entryKeyAddress=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                     Value entryKey = rewriter.create<util::LoadOp>(loc, keyType, entryKeyAddress);

                     Value keyMatches = equalFn(b,entryKey,key);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc, TypeRange({doneType,ptrType}), keyMatches, [&](OpBuilder& b, Location loc) {
                           //          entry.aggr = update(vec.aggr,val)
                           if(updateFn) {
                              Value entryAggrAddress = rewriter.create<util::TupleElementPtrOp>(loc, aggrPtrType, kvAddress, 1);
                              Value entryAggr = rewriter.create<util::LoadOp>(loc, aggrType, entryAggrAddress);
                              Value newAggr = updateFn(b, entryAggr, val);
                              b.create<util::StoreOp>(loc, newAggr, entryAggrAddress, Value());
                           }
                           b.create<scf::YieldOp>(loc, ValueRange{falseValue,ptr});
                        }, [&](OpBuilder& b, Location loc) {

                              //          ptr = &entry.next
                              Value newPtr=b.create<util::GenericMemrefCastOp>(loc, ptrType,currEntryPtr);
                              //          yield ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {

                        //          ptr = &entry.next
                        Value newPtr=b.create<util::GenericMemrefCastOp>(loc, ptrType,currEntryPtr);
                        //          yield ptr,done=false
                        b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) {
               Value initValAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(rewriter.getContext(), aggrType), aggrHtBuilder, 5);
               Value initialVal = b.create<util::LoadOp>(loc, aggrType, initValAddress);
               Value newAggr = updateFn ? updateFn(b,initialVal, val): initialVal;
               Value newKVPair = b.create<util::PackOp>(loc,ValueRange({key, newAggr}));
               Value invalidNext  = b.create<util::InvalidRefOp>(loc,i8PtrType);
               //       %newEntry = ...
               Value newEntry = b.create<util::PackOp>(loc, ValueRange({invalidNext, hash, newKVPair}));
               Value valuesAddress = b.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(b.getContext(),valuesType), aggrHtBuilder, 3);
               Value values = b.create<util::LoadOp>(loc, valuesType, valuesAddress);
               Value newValueLocPtr=b.create<util::ArrayElementPtrOp>(loc,bucketPtrType,values,len);
               //       append(vec,newEntry)
               b.create<util::StoreOp>(loc, newEntry, newValueLocPtr,Value());

               //       *ptr=len
               b.create<util::StoreOp>(loc, newValueLocPtr, ptr, Value());
               Value newLen = b.create<arith::AddIOp>(loc, len, one);
               //       yield 0,0,done=true
               b.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());

               b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });
         //       if(compare(entry.key,key)){

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done,
                                           ValueRange({newPtr}));
      }

      // The body of the while loop: shift right until reaching a value of 0.
      {
         rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      }

      rewriter.setInsertionPointAfter(whileOp);
   }
   Value build(mlir::OpBuilder& builder, Value aggrHtBuilder) {
      return aggrHtBuilder;
   }
};
class CreateDsLowering : public ConversionPattern {
   mlir::dsa::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateDsLowering(TypeConverter& typeConverter, MLIRContext* context, dsa::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::dsa::CreateDS::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto createOp = mlir::cast<mlir::dsa::CreateDS>(op);
      mlir::dsa::CreateDSAdaptor adaptor(operands);
      if (auto joinHtType = createOp.ds().getType().dyn_cast<mlir::dsa::JoinHashtableType>()) {
         auto entryType = mlir::TupleType::get(rewriter.getContext(), {joinHtType.getKeyType(), joinHtType.getValType()});
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
         Value typesize = rewriter.create<mlir::util::SizeOfOp>(op->getLoc(), rewriter.getIndexType(), typeConverter->convertType(tupleType));
         Value ptr = functionRegistry.call(rewriter, op->getLoc(), mlir::dsa::codegen::FunctionRegistry::FunctionId::JoinHtCreate, typesize)[0];
         rewriter.replaceOpWithNewOp<util::GenericMemrefCastOp>(op, typeConverter->convertType(joinHtType), ptr);
         return success();
      } else if (auto vecType = createOp.ds().getType().dyn_cast<mlir::dsa::VectorType>()) {
         Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1024);
         auto elementType = typeConverter->convertType(vecType.getElementType());
         VectorHelper2 vectorHelper(elementType, op->getLoc());
         rewriter.replaceOp(op, vectorHelper.create(rewriter, initialCapacity, functionRegistry));
         return success();
      } else if (auto aggrHtType = createOp.ds().getType().dyn_cast<mlir::dsa::AggregationHashtableType>()) {
         TupleType keyType = aggrHtType.getKeyType();
         TupleType aggrType = aggrHtType.getValType();
         if (keyType.getTypes().empty()) {
            mlir::Value ref = rewriter.create<mlir::util::AllocOp>(op->getLoc(), typeConverter->convertType(createOp.ds().getType()), mlir::Value());
            rewriter.create<mlir::util::StoreOp>(op->getLoc(), adaptor.init_val(), ref, mlir::Value());
            rewriter.replaceOp(op, ref);
            return success();
         } else {
            AggrHtHelper2 helper(rewriter.getContext(), keyType, aggrType, op->getLoc(), typeConverter);
            rewriter.replaceOp(op, helper.create(rewriter, adaptor.init_val(), functionRegistry));
            return success();
         }
      }
      return failure();
   }
};
class HashtableInsertLowering : public ConversionPattern {
   mlir::dsa::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit HashtableInsertLowering(TypeConverter& typeConverter, MLIRContext* context, dsa::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::dsa::HashtableInsert::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::dsa::HashtableInsertAdaptor adaptor(operands);
      auto reduceOp = mlir::cast<mlir::dsa::HashtableInsert>(op);
      if(!reduceOp.ht().getType().isa<mlir::dsa::AggregationHashtableType>()){
         return failure();
      }
      std::function<Value(OpBuilder&, Value, Value)> reduceFnBuilder = reduceOp.reduce().empty() ? std::function<Value(OpBuilder&, Value, Value)>() : [&reduceOp](OpBuilder& rewriter, Value left, Value right) {
         Block* sortLambda = &reduceOp.reduce().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::BlockAndValueMapping mapping;
         mapping.map(sortLambda->getArgument(0), left);
         mapping.map(sortLambda->getArgument(1), right);

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         return mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
      };
      std::function<Value(OpBuilder&, Value, Value)> equalFnBuilder = reduceOp.equal().empty() ? std::function<Value(OpBuilder&, Value, Value)>() : [&reduceOp](OpBuilder& rewriter, Value left, Value right) {
         Block* sortLambda = &reduceOp.equal().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::BlockAndValueMapping mapping;
         mapping.map(sortLambda->getArgument(0), left);
         mapping.map(sortLambda->getArgument(1), right);

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         return mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
      };
      auto loc = op->getLoc();
      if (reduceOp.ht().getType().cast<mlir::dsa::AggregationHashtableType>().getKeyType() == mlir::TupleType::get(getContext())) {
         auto loaded = rewriter.create<mlir::util::LoadOp>(loc, adaptor.ht().getType().cast<mlir::util::RefType>().getElementType(), adaptor.ht(), mlir::Value());
         auto newAggr = reduceFnBuilder(rewriter, loaded, adaptor.val());
         rewriter.create<mlir::util::StoreOp>(loc, newAggr, adaptor.ht(), mlir::Value());
         rewriter.eraseOp(op);
      } else {
         Value hashed;
         {
            Block* sortLambda = &reduceOp.hash().front();
            auto* sortLambdaTerminator = sortLambda->getTerminator();
            mlir::BlockAndValueMapping mapping;
            mapping.map(sortLambda->getArgument(0), adaptor.key());

            for (auto& op : sortLambda->getOperations()) {
               if (&op != sortLambdaTerminator) {
                  rewriter.clone(op, mapping);
               }
            }
            hashed = mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
         }

         AggrHtHelper2 helper(rewriter.getContext(), adaptor.key().getType(), typeConverter->convertType(reduceOp.ht().getType().cast<mlir::dsa::AggregationHashtableType>().getValType()), loc, typeConverter);
         helper.insert(rewriter, adaptor.ht(), adaptor.key(), adaptor.val(), hashed, reduceFnBuilder, equalFnBuilder, functionRegistry);
         rewriter.eraseOp(op);
      }
      return success();
   }
};
class JoinHtHashtableInsertLowering : public ConversionPattern {
   mlir::dsa::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit JoinHtHashtableInsertLowering(TypeConverter& typeConverter, MLIRContext* context, dsa::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::dsa::HashtableInsert::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::dsa::HashtableInsertAdaptor adaptor(operands);
      auto reduceOp = mlir::cast<mlir::dsa::HashtableInsert>(op);
      if(!reduceOp.ht().getType().isa<mlir::dsa::JoinHashtableType>()){
         return failure();
      }
      Value hashed;
      {
         Block* sortLambda = &reduceOp.hash().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         mlir::BlockAndValueMapping mapping;
         mapping.map(sortLambda->getArgument(0), adaptor.key());

         for (auto& op : sortLambda->getOperations()) {
            if (&op != sortLambdaTerminator) {
               rewriter.clone(op, mapping);
            }
         }
         hashed = mapping.lookup(cast<mlir::dsa::YieldOp>(sortLambdaTerminator).results()[0]);
      }
      mlir::Value val = adaptor.val();
      if (!val) {
         val = rewriter.create<mlir::util::UndefTupleOp>(op->getLoc(), mlir::TupleType::get(getContext()));
      }
      auto entry = rewriter.create<mlir::util::PackOp>(op->getLoc(), mlir::ValueRange({adaptor.key(), val}));
      auto bucket = rewriter.create<mlir::util::PackOp>(op->getLoc(), mlir::ValueRange({hashed, entry}));
      auto loc = op->getLoc();
      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto valuesType = mlir::util::RefType::get(rewriter.getContext(), bucket.getType());
      Value lenAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 2);
      Value capacityAddress = rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, adaptor.ht(), 3);

      auto len = rewriter.create<mlir::util::LoadOp>(loc, idxType, lenAddress, Value());
      auto capacity = rewriter.create<mlir::util::LoadOp>(loc, idxType, capacityAddress, Value());
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      rewriter.create<scf::IfOp>(
         loc, TypeRange({}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc); }, [&](OpBuilder& b, Location loc) {
            Value downCasted = b.create<util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(b.getContext(),b.getI8Type()), adaptor.ht());
            functionRegistry.call(b,loc,mlir::dsa::codegen::FunctionRegistry::FunctionId::JoinHtResize,ValueRange{downCasted});
            b.create<scf::YieldOp>(loc); });
      Value valuesAddress = rewriter.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(getContext(), adaptor.ht().getType().cast<mlir::util::RefType>().getElementType().cast<mlir::TupleType>().getType(4)), adaptor.ht(), 4);
      Value castedValuesAddress = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(getContext(), valuesType), valuesAddress);
      auto values = rewriter.create<mlir::util::LoadOp>(loc, valuesType, castedValuesAddress, Value());
      rewriter.create<util::StoreOp>(loc, bucket, values, len);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = rewriter.create<arith::AddIOp>(loc, len, one);

      rewriter.create<mlir::util::StoreOp>(loc, newLen, lenAddress, Value());
      rewriter.eraseOp(op);
      return success();
   }
};
class HashtableFinalizeLowering : public ConversionPattern {
   mlir::dsa::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit HashtableFinalizeLowering(TypeConverter& typeConverter, MLIRContext* context, dsa::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::dsa::HashtableFinalize::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::dsa::HashtableFinalizeAdaptor adaptor(operands);
      Value downCasted = rewriter.create<util::GenericMemrefCastOp>(op->getLoc(), mlir::util::RefType::get(rewriter.getContext(), rewriter.getI8Type()), adaptor.ht());
      functionRegistry.call(rewriter, op->getLoc(), mlir::dsa::codegen::FunctionRegistry::FunctionId::JoinHtFinalize, ValueRange{downCasted});
      rewriter.eraseOp(op);
      return success();
   }
};
class DSAppendLowering : public ConversionPattern {
   mlir::dsa::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit DSAppendLowering(TypeConverter& typeConverter, MLIRContext* context, dsa::codegen::FunctionRegistry& functionRegistry)
      : ConversionPattern(typeConverter, mlir::dsa::Append::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto appendOp = mlir::cast<mlir::dsa::Append>(op);
      mlir::dsa::AppendAdaptor adaptor(operands);
      Value builderVal = adaptor.ds();
      Value v = adaptor.val();
      auto convertedElementType = typeConverter->convertType(appendOp.ds().getType().cast<mlir::dsa::VectorType>().getElementType());
      VectorHelper2 helper(convertedElementType, op->getLoc());
      helper.insert(rewriter, builderVal, v, functionRegistry);
      rewriter.eraseOp(op);
      return success();
   }
};
} // end namespace
namespace mlir::dsa {
void populateDSAToStdPatterns(mlir::dsa::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateDsLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<HashtableInsertLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<HashtableFinalizeLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<DSAppendLowering>(typeConverter, patterns.getContext(), functionRegistry);
   patterns.insert<JoinHtHashtableInsertLowering>(typeConverter, patterns.getContext(), functionRegistry);
}
} // end namespace mlir::dsa