#include "arrow/type_fwd.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

class VectorHelper {
   Type elementType;
   Location loc;

   public:
   static Type createType(MLIRContext* context, Type elementType) {
      return mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), mlir::util::GenericMemrefType::get(context, elementType, -1)});
   }
   VectorHelper(Type elementType, Location loc) : elementType(elementType), loc(loc) {
   }
   Value create(mlir::OpBuilder& builder, Value initialCapacity) {
      Value initialSize = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto type = mlir::util::GenericMemrefType::get(builder.getContext(), elementType, -1);
      Value memref = builder.create<mlir::util::AllocOp>(loc, type, initialCapacity);
      Value vec = builder.create<mlir::util::PackOp>(loc, createType(builder.getContext(), elementType), ValueRange{initialSize, initialCapacity, memref});
      return vec;
   }
   Value insert(mlir::OpBuilder& builder, Value vec, Value newVal) {
      Type typedPtrType = util::GenericMemrefType::get(builder.getContext(), elementType, -1);
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getIndexType(), builder.getIndexType(), typedPtrType}), vec);

      Value values = unpacked.getResult(2);
      Value capacity = unpacked.getResult(1);

      Value len = unpacked.getResult(0);
      Value cmp = builder.create<arith::CmpIOp>(builder.getUnknownLoc(), arith::CmpIPredicate::ult, len, capacity);
      auto ifOp = builder.create<scf::IfOp>(
         loc, TypeRange({builder.getIndexType(), values.getType()}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{capacity, values}); }, [&](OpBuilder& b, Location loc) {
            Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
            Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
            Value two = builder.create<arith::ConstantIndexOp>(loc, 2);
            Value newCapacity = b.create<arith::MulIOp>(loc, len, two);
            Value newMemref = builder.create<mlir::util::AllocOp>(loc, values.getType(), newCapacity);
            b.create<scf::ForOp>(
               loc, zero, len, one, ValueRange({}),
               [&](OpBuilder& b2, Location loc2, Value iv, ValueRange args) {
                  Value currVal = b2.create<util::LoadOp>(loc, elementType, values, iv);
                  b2.create<util::StoreOp>(b2.getUnknownLoc(), currVal, newMemref, iv);
                  b2.create<scf::YieldOp>(loc);
               });
            b.create<mlir::util::DeAllocOp>(loc, values);
            b.create<scf::YieldOp>(loc, ValueRange{newCapacity,newMemref}); });
      capacity = ifOp.getResult(0);
      values = ifOp.getResult(1);
      builder.create<util::StoreOp>(builder.getUnknownLoc(), newVal, values, len);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = builder.create<arith::AddIOp>(loc, len, one);

      Value updatedVector = builder.create<mlir::util::PackOp>(loc, TupleType::get(builder.getContext(), {builder.getIndexType(), builder.getIndexType(), typedPtrType}), ValueRange{newLen, capacity, values});
      return updatedVector;
   }
};
class AggrHtHelper {
   Type keyType;
   Type aggrType;
   TupleType entryType;
   Location loc;

   public:

   static Type kvType(MLIRContext* context, Type keyType,Type aggrType){
      return mlir::TupleType::get(context, {keyType,aggrType});
   }
   static TupleType createEntryType(MLIRContext* context, Type keyType,Type aggrType){
      return mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), kvType(context,keyType,aggrType)});
   }
   static Type createType(MLIRContext* context,Type keyType, Type aggrType, Type valType,Type compareFn, Type updateFn){
      auto idxType=IndexType::get(context);
      auto valuesType = mlir::util::GenericMemrefType::get(context, createEntryType(context,keyType,aggrType), -1);
      auto htType = mlir::util::GenericMemrefType::get(context, idxType, -1);
      //{initialSize.getType(),initialCapacity.getType(),values.getType(),ht.getType(),compareFn.getType(),updateFn.getType(),initialValue.getType()}
      return mlir::TupleType::get(context,{idxType,idxType,valuesType,htType,compareFn, updateFn,aggrType});
   }
   AggrHtHelper(MLIRContext* context,Type keyType,Type aggrType, Location loc) : keyType(keyType),aggrType(aggrType),entryType(createEntryType(context,keyType,aggrType)), loc(loc) {
   }

   Value create(mlir::OpBuilder& builder,Value compareFn, Value updateFn,Value initialValue) {
      Value initialCapacity = builder.create<arith::ConstantIndexOp>(loc, 4);
      Value initialHtSize = builder.create<arith::ConstantIndexOp>(loc, 8);

      Value initialSize = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto valuesType = mlir::util::GenericMemrefType::get(builder.getContext(), entryType, -1);
      auto htType = mlir::util::GenericMemrefType::get(builder.getContext(), builder.getIndexType(), -1);

      Value values = builder.create<mlir::util::AllocOp>(loc, valuesType, initialCapacity);
      Value ht = builder.create<mlir::util::AllocOp>(loc, htType, initialHtSize);
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value maxValue = builder.create<arith::ConstantIndexOp>(loc, 0xFFFFFFFFFFFFFFFF);

      builder.create<scf::ForOp>(
         loc, zero, initialHtSize, one, ValueRange({}),
         [&](OpBuilder& b, Location loc2, Value iv, ValueRange args) {
            b.create<util::StoreOp>(loc2, maxValue, ht, iv);
            b.create<scf::YieldOp>(loc);
         });
      Value vec = builder.create<mlir::util::PackOp>(loc, TupleType::get(builder.getContext(),TypeRange({initialSize.getType(),initialCapacity.getType(),values.getType(),ht.getType(),compareFn.getType(),updateFn.getType(),initialValue.getType()})), ValueRange{initialSize, initialCapacity, values, ht,compareFn,updateFn,initialValue});
      return vec;
   }
   Value insert(mlir::OpBuilder& builder, Value aggrHtBuilder, Value key, Value val, Value hash) {
      auto *context=builder.getContext();
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), aggrHtBuilder.getType().cast<TupleType>().getTypes(), aggrHtBuilder);
      Value len = unpacked.getResult(0);
      Value ht = unpacked.getResult(3);
      Value values = unpacked.getResult(2);
      Value capacity = unpacked.getResult(1);
      Value compareFn=unpacked.getResult(4);
      Value updateFn=unpacked.getResult(5);
      Value initialVal=unpacked.getResult(6);
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value two = builder.create<arith::ConstantIndexOp>(loc, 2);
      Value four = builder.create<arith::ConstantIndexOp>(loc, 4);

      Value maxValue = builder.create<arith::ConstantIndexOp>(loc, 0xFFFFFFFFFFFFFFFF);

      Value cmp = builder.create<arith::CmpIOp>(builder.getUnknownLoc(), arith::CmpIPredicate::ult, len, capacity);
      auto ifOp = builder.create<scf::IfOp>(
         loc, TypeRange({builder.getIndexType(), values.getType(), ht.getType()}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{capacity, values, ht}); }, [&](OpBuilder& b, Location loc) {
            Value newCapacity = b.create<arith::MulIOp>(loc, len, two);
            Value newHtSize = b.create<arith::MulIOp>(loc, len, four);
            Value htMask = b.create<arith::SubIOp>(loc, newHtSize, one);

            Value newValues = builder.create<mlir::util::AllocOp>(loc, values.getType(), newCapacity);
            Value newHt = builder.create<mlir::util::AllocOp>(loc, ht.getType(), newHtSize);
            builder.create<scf::ForOp>(
               loc, zero, newHtSize, one, ValueRange({}),
               [&](OpBuilder& b, Location loc2, Value iv, ValueRange args) {
                  b.create<util::StoreOp>(loc2, maxValue, newHt, iv);
                  b.create<scf::YieldOp>(loc);
               });
            b.create<scf::ForOp>(
               loc, zero, len, one, ValueRange({}),
               [&](OpBuilder& b2, Location loc2, Value iv, ValueRange args) {
                  Value currVal = b2.create<util::LoadOp>(loc, entryType, values, iv);

                  auto unpacked2 = b2.create<util::UnPackOp>(loc, entryType.getTypes(), currVal);
                  Value buckedPos = b2.create<arith::AndIOp>(loc,htMask,unpacked2.getResult(1));

                  Value previousPtr = b2.create<util::LoadOp>(loc, builder.getIndexType(), newHt, buckedPos);
                  b2.create<util::StoreOp>(loc2, iv, newHt, buckedPos);
                  auto repacked= b2.create<util::PackOp>(loc,entryType,ValueRange{previousPtr,unpacked2.getResult(1),unpacked2.getResult(2)});
                  b2.create<util::StoreOp>(loc2, repacked, newValues, iv);

                  b2.create<scf::YieldOp>(loc);
               });
            b.create<mlir::util::DeAllocOp>(loc, values);
            b.create<scf::YieldOp>(loc, ValueRange{newCapacity, newValues,newHt}); });
      capacity = ifOp.getResult(0);
      values = ifOp.getResult(1);
      ht = ifOp.getResult(2);

      Value trueValue = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 1));
      Value falseValue = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 0));

      Value htSize =  builder.create<arith::MulIOp>(loc, capacity, two);

      Value htMask = builder.create<arith::SubIOp>(loc, htSize, one);

      //position = hash & hashTableMask
      Value position = builder.create<arith::AndIOp>(loc, htMask, hash);
      //idx = hashtable[position]
      Value idx = builder.create<util::LoadOp>(loc, builder.getIndexType(), ht, position);
      // ptr = &hashtable[position]
      Type idxType=builder.getIndexType();
      Value ptr = builder.create<util::ElementPtrOp>(loc,util::GenericMemrefType::get(context,idxType,llvm::Optional<int64_t>()),ht,position);
      Type ptrType=util::GenericMemrefType::get(context,idxType,llvm::Optional<int64_t>());
      Type doneType=builder.getI1Type();

      auto resultTypes=std::vector<Type>({idxType,idxType,ptrType});
      auto whileOp = builder.create<scf::WhileOp>(loc, resultTypes, ValueRange({len,idx,ptr}));
      Block* before = builder.createBlock(&whileOp.before(), {}, resultTypes);
      Block* after = builder.createBlock(&whileOp.after(), {}, resultTypes);

      // The conditional block of the while loop.
      {
         builder.setInsertionPointToStart(&whileOp.before().front());
         Value len = before->getArgument(0);
         Value idx = before->getArgument(1);
         Value ptr = before->getArgument(2);

         //    if (idx == 0xFFFFFFFFFFF){
         Value cmp = builder.create<arith::CmpIOp>(builder.getUnknownLoc(), arith::CmpIPredicate::eq, idx, maxValue);
         auto ifOp = builder.create<scf::IfOp>(
            loc, TypeRange({idxType,doneType,idxType,ptrType}), cmp, [&](OpBuilder& b, Location loc) {
               Value newAggr= b.create<mlir::CallIndirectOp>(loc, updateFn, ValueRange({initialVal, val})).results()[0];
               Value newKVPair=b.create<util::PackOp>(loc,TupleType::get(context,TypeRange({key.getType(),newAggr.getType()})),ValueRange({key,newAggr}));
               //       %newEntry = ...
               Value newEntry=b.create<util::PackOp>(loc,TupleType::get(context,TypeRange({idxType,idxType, newKVPair.getType()})),ValueRange({maxValue,hash,newKVPair}));


               //       append(vec,newEntry)
               b.create<util::StoreOp>(builder.getUnknownLoc(), newEntry, values, len);

               //       *ptr=len
               b.create<util::StoreOp>(builder.getUnknownLoc(),len,ptr,Value());
               Value newLen=b.create<arith::AddIOp>(loc,len,one);
               //       yield 0,0,done=true
               b.create<scf::YieldOp>(loc, ValueRange{newLen,falseValue, idx,ptr});

            }, [&](OpBuilder& b, Location loc) {

         //       entry=vec[idx]
                  Value currEntry= b.create<util::LoadOp>(loc, entryType, values, idx);
                  auto entryUnpacked=b.create<util::UnPackOp>(loc,currEntry.getType().cast<TupleType>().getTypes(),currEntry);
                  Value kv=entryUnpacked.getResult(2);
                  auto kvUnpacked=b.create<util::UnPackOp>(loc,kv.getType().cast<TupleType>().getTypes(),kv);

                  Value entryKey=kvUnpacked.getResult(0);
                  Value entryAggr=kvUnpacked.getResult(1);
                  Value entryNext=entryUnpacked.getResult(0);
                  Value entryHash=entryUnpacked.getResult(1);

         //       if(compare(entry.key,key)){
                  Value keyMatches = b.create<mlir::CallIndirectOp>(loc, compareFn, ValueRange({entryKey,key})).results()[0];
                  auto ifOp2 = builder.create<scf::IfOp>(
                     loc, TypeRange({idxType,doneType,idxType,ptrType}), keyMatches, [&](OpBuilder& b, Location loc) {
                        //          entry.aggr = update(vec.aggr,val)
                        Value newAggr= builder.create<mlir::CallIndirectOp>(loc, updateFn, ValueRange({entryAggr, val})).results()[0];
                        Value newKVPair=b.create<util::PackOp>(loc,TupleType::get(context,TypeRange({entryKey.getType(),newAggr.getType()})),ValueRange({entryKey,newAggr}));
                        Value newEntry=b.create<util::PackOp>(loc,TupleType::get(context,TypeRange({idxType,idxType, newKVPair.getType()})),ValueRange({entryNext,entryHash,newKVPair}));
                        b.create<util::StoreOp>(builder.getUnknownLoc(), newEntry, values, idx);
                        b.create<scf::YieldOp>(loc, ValueRange{len,falseValue, idx,ptr});
                        }, [&](OpBuilder& b, Location loc) {

                        //          ptr = &entry.next
                        Value entryPtr=builder.create<util::ElementPtrOp>(loc,util::GenericMemrefType::get(context,entryType,llvm::Optional<int64_t>()),values,idx);

                        ptr=builder.create<util::GenericMemrefCastOp>(loc, util::GenericMemrefType::get(context,idxType,llvm::Optional<int64_t>()),entryPtr);
                        //          yield idx,ptr,done=false
                        b.create<scf::YieldOp>(loc, ValueRange{len,trueValue, entryNext, ptr });});
               b.create<scf::YieldOp>(loc, ifOp2.getResults()); });
         Value newLen=ifOp.getResult(0);
         Value done=ifOp.getResult(1);
         idx =ifOp.getResult(2);
         ptr = ifOp.getResult(3);
         builder.create<scf::ConditionOp>(loc, done,
                                          ValueRange({newLen,idx,ptr}));
      }

      // The body of the while loop: shift right until reaching a value of 0.
      {
         builder.setInsertionPointToStart(&whileOp.after().front());
         builder.create<scf::YieldOp>(loc, after->getArguments());
      }

      builder.setInsertionPointAfter(whileOp);
      // do{

      // } while (!done)
      //
      Value newLen=whileOp.getResult(0);



      Value updatedBuilder = builder.create<mlir::util::PackOp>(loc, aggrHtBuilder.getType(), ValueRange{newLen, capacity, values,ht,compareFn,updateFn,initialVal});
      return updatedBuilder;
   }
   Value build(mlir::OpBuilder& builder, Value aggrHtBuilder){
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), aggrHtBuilder.getType().cast<TupleType>().getTypes(), aggrHtBuilder);
      Value len = unpacked.getResult(0);
      //Value ht = unpacked.getResult(3);
      Value values = unpacked.getResult(2);
      //Value capacity = unpacked.getResult(1);
      //Value compareFn=unpacked.getResult(4);
      //Value updateFn=unpacked.getResult(5);
      //Value initialVal=unpacked.getResult(6);
      return builder.create<util::PackOp>(builder.getUnknownLoc(),TupleType::get(builder.getContext(),TypeRange({len.getType(),values.getType()})),ValueRange({len,values}));
   }
};
class CreateVectorBuilderLowering : public ConversionPattern {
   public:
   explicit CreateVectorBuilderLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateVectorBuilder::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto vecBuilderOp = mlir::cast<mlir::db::CreateVectorBuilder>(op);
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1024);
      auto elementType = typeConverter->convertType(vecBuilderOp.getType().getElementType());
      VectorHelper vectorHelper(elementType, op->getLoc());
      rewriter.replaceOp(op, vectorHelper.create(rewriter, initialCapacity));
      return success();
   }
};

static db::codegen::FunctionRegistry::FunctionId getStoreFunc(db::codegen::FunctionRegistry& functionRegistry, db::DBType type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;
   if (auto intType = type.dyn_cast_or_null<mlir::db::IntType>()) {
      switch (intType.getWidth()) {
         case 8: return FunctionId::ArrowTableBuilderAddInt8;
         case 16: return FunctionId::ArrowTableBuilderAddInt16;
         case 32: return FunctionId::ArrowTableBuilderAddInt32;
         case 64: return FunctionId::ArrowTableBuilderAddInt64;
      }
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      return FunctionId::ArrowTableBuilderAddBool;
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      if (decimalType.getP() < 19) {
         return FunctionId::ArrowTableBuilderAddSmallDecimal;
      }
      return FunctionId::ArrowTableBuilderAddDecimal;
   } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
      switch (floatType.getWidth()) {
         case 32: return FunctionId ::ArrowTableBuilderAddFloat32;
         case 64: return FunctionId ::ArrowTableBuilderAddFloat64;
      }
   } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
      return FunctionId::ArrowTableBuilderAddBinary;
   } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
      if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
         return FunctionId::ArrowTableBuilderAddDate32;
      } else {
         return FunctionId::ArrowTableBuilderAddDate64;
      }
   }
   //TODO: implement other types too
   return FunctionId::ArrowTableBuilderAddInt32;
}
class BuilderMergeLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit BuilderMergeLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BuilderMerge::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::BuilderMergeAdaptor mergeOpAdaptor(operands);
      auto mergeOp = cast<mlir::db::BuilderMerge>(op);
      auto loc = rewriter.getUnknownLoc();
      if (auto tableBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         TupleType rowType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
         auto loweredTypes = mergeOpAdaptor.val().getType().cast<TupleType>().getTypes();
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, loweredTypes, mergeOpAdaptor.val());
         size_t i = 0;
         Value falseValue = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         for (auto v : unPackOp.vals()) {
            Value isNull;
            if (mergeOp.val().getType().cast<TupleType>().getType(i).cast<db::DBType>().isNullable()) {
               auto nullUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, v.getType().cast<TupleType>().getTypes(), v);
               isNull = nullUnpacked.getResult(0);
               v = nullUnpacked->getResult(1);
            } else {
               isNull = falseValue;
            }
            Value columnId = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            functionRegistry.call(rewriter, getStoreFunc(functionRegistry, rowType.getType(i).cast<mlir::db::DBType>()), ValueRange({mergeOpAdaptor.builder(), columnId, isNull, v}));
            i++;
         }
         functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderFinishRow, mergeOpAdaptor.builder());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      } else if (auto vectorBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         Value builderVal = mergeOpAdaptor.builder();
         Value v = mergeOpAdaptor.val();
         auto convertedElementType = typeConverter->convertType(vectorBuilderType.getElementType());
         VectorHelper helper(convertedElementType, op->getLoc());

         rewriter.replaceOp(op, helper.insert(rewriter, builderVal, v));
      } else if (auto aggrHTBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::AggrHTBuilderType>()) {
         if (aggrHTBuilderType.getKeyType().getTypes().empty()) {
            auto keyType = aggrHTBuilderType.getKeyType();

            auto aggrType = aggrHTBuilderType.getAggrType();
            auto valType = aggrHTBuilderType.getValType();
            auto builderUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, TypeRange{typeConverter->convertType(aggrType), typeConverter->convertType(FunctionType::get(rewriter.getContext(), {aggrType, valType}, {aggrType}))}, mergeOpAdaptor.builder())->getResults();
            auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, TypeRange{typeConverter->convertType(keyType), typeConverter->convertType(valType)}, mergeOpAdaptor.val())->getResults();
            auto newAggr = rewriter.create<mlir::CallIndirectOp>(loc, builderUnpacked[1], ValueRange({builderUnpacked[0], unPacked[1]})).results()[0];
            mlir::Value builderPacked = rewriter.create<mlir::util::PackOp>(loc, mergeOpAdaptor.builder().getType(), ValueRange{newAggr, builderUnpacked[1]});
            rewriter.replaceOp(op, builderPacked);
         } else {

            auto loweredTypes = mergeOpAdaptor.val().getType().cast<TupleType>().getTypes();

            auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, loweredTypes, mergeOpAdaptor.val())->getResults();
            auto higherType = rewriter.create<mlir::db::TypeCastOp>(loc, mergeOp.val().getType().cast<TupleType>().getType(0), unPacked[0]);
            Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), higherType);
            Value currKey = unPacked[0];
            Value currVal = unPacked[1];
            auto keyType=typeConverter->convertType(aggrHTBuilderType.getKeyType());
            auto aggrType=typeConverter->convertType(aggrHTBuilderType.getAggrType());

            AggrHtHelper helper(rewriter.getContext(),keyType,aggrType,rewriter.getUnknownLoc());
            rewriter.replaceOp(op, helper.insert(rewriter,mergeOpAdaptor.builder(),currKey,currVal,hashed));
         }
      } else if (auto joinHtBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::JoinHTBuilderType>()) {
         Value v = mergeOpAdaptor.val();
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), v.getType()});
         auto valType = mergeOpAdaptor.val().getType().cast<mlir::TupleType>();

         auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, valType.getTypes(), mergeOpAdaptor.val())->getResults();
         Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), unPacked[0]);
         auto bucket = rewriter.create<mlir::util::PackOp>(rewriter.getUnknownLoc(), tupleType, mlir::ValueRange({hashed, v}));
         VectorHelper helper(typeConverter->convertType(tupleType), op->getLoc());
         rewriter.replaceOp(op, helper.insert(rewriter, mergeOpAdaptor.builder(), bucket));
      } else if (auto joinHtBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::MarkableJoinHTBuilderType>()) {
         Value v = mergeOpAdaptor.val();
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), rewriter.getIndexType(), v.getType()});
         auto valType = mergeOpAdaptor.val().getType().cast<mlir::TupleType>();

         auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, valType.getTypes(), mergeOpAdaptor.val())->getResults();
         Value zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
         Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), unPacked[0]);
         auto bucket = rewriter.create<mlir::util::PackOp>(rewriter.getUnknownLoc(), tupleType, mlir::ValueRange({zero, hashed, v}));
         VectorHelper helper(typeConverter->convertType(tupleType), op->getLoc());
         rewriter.replaceOp(op, helper.insert(rewriter, mergeOpAdaptor.builder(), bucket));
      }

      return success();
   }
};

class CreateAggrHTBuilderLowering : public ConversionPattern {

   public:
   explicit CreateAggrHTBuilderLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateAggrHTBuilder::getOperationName(), 1, context){}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {

      static size_t id = 0;
      auto createOp = cast<mlir::db::CreateAggrHTBuilder>(op);

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      TupleType keyType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getKeyType();
      TupleType valType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getValType();
      TupleType aggrType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getAggrType();

      FuncOp rawUpdateFunc;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         rawUpdateFunc = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_raw_update" + std::to_string(id++), rewriter.getFunctionType(TypeRange({aggrType, valType}), TypeRange(aggrType)));
         rawUpdateFunc.getOperation()->setAttr("passthrough", ArrayAttr::get(rewriter.getContext(), {rewriter.getStringAttr("alwaysinline")}));
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({aggrType, valType}));
         rawUpdateFunc.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value left = funcBody->getArgument(0);
         Value right = funcBody->getArgument(1);
         Value tupleLeft = left;
         Value tupleRight = right;
         auto terminator = rewriter.create<mlir::ReturnOp>(createOp.getLoc());
         Block* sortLambda = &createOp.region().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.mergeBlockBefore(sortLambda, terminator, {tupleLeft, tupleRight});
         mlir::db::YieldOp yieldOp = mlir::cast<mlir::db::YieldOp>(terminator->getPrevNode());
         Value x = yieldOp.results()[0];
         rewriter.setInsertionPoint(terminator);
         rewriter.eraseOp(sortLambdaTerminator);
         Value castedVal = rewriter.create<mlir::db::TypeCastOp>(rewriter.getUnknownLoc(), typeConverter->convertType(aggrType), x);

         rewriter.create<mlir::ReturnOp>(createOp.getLoc(), castedVal);
         rewriter.eraseOp(terminator);
      }
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         createOp.region().push_back(new Block());
         rewriter.setInsertionPointToStart(&createOp.region().front());
         rewriter.create<mlir::db::YieldOp>(createOp.getLoc());
      }
      if (keyType.getTypes().empty()) {
         mlir::Value funcRef = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rawUpdateFunc.getType(), SymbolRefAttr::get(rewriter.getStringAttr(rawUpdateFunc.sym_name())));
         rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, TupleType::get(rewriter.getContext(), {aggrType, rawUpdateFunc.getType()}), ValueRange{createOp.initial(), funcRef});
         return success();
      } else {
         FuncOp rawCompareFunc;
         {
            OpBuilder::InsertionGuard insertionGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            rawCompareFunc = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_raw_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({keyType, keyType}), TypeRange(mlir::db::BoolType::get(rewriter.getContext()))));
            rawCompareFunc.getOperation()->setAttr("passthrough", ArrayAttr::get(rewriter.getContext(), {rewriter.getStringAttr("alwaysinline")}));
            auto* funcBody = new Block;
            funcBody->addArguments(TypeRange({keyType, keyType}));
            rawCompareFunc.body().push_back(funcBody);
            rewriter.setInsertionPointToStart(funcBody);
            Value left = funcBody->getArgument(0);
            Value right = funcBody->getArgument(1);
            Value equal = rewriter.create<mlir::db::ConstantOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
            auto leftUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), keyType.getTypes(), left);
            auto rightUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), keyType.getTypes(), right);
            for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
               Value compared = rewriter.create<mlir::db::CmpOp>(rewriter.getUnknownLoc(), mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));
               if (leftUnpacked->getResult(i).getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() && rightUnpacked.getResult(i).getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
                  Value isNull1 = rewriter.create<mlir::db::IsNullOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), leftUnpacked->getResult(i));
                  Value isNull2 = rewriter.create<mlir::db::IsNullOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), rightUnpacked->getResult(i));
                  Value bothNull = rewriter.create<mlir::db::AndOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), ValueRange({isNull1, isNull2}));
                  Value casted = rewriter.create<mlir::db::CastOp>(rewriter.getUnknownLoc(), compared.getType(), bothNull);
                  Value tmp = rewriter.create<mlir::db::SelectOp>(rewriter.getUnknownLoc(), bothNull, casted, compared);
                  compared = tmp;
               }
               Value localEqual = rewriter.create<mlir::db::AndOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext(), equal.getType().cast<mlir::db::BoolType>().getNullable() || compared.getType().cast<mlir::db::BoolType>().getNullable()), ValueRange({equal, compared}));
               equal = localEqual;
            }
            if (equal.getType().cast<mlir::db::DBType>().isNullable()) {
               Value isNull = rewriter.create<mlir::db::IsNullOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), equal);
               Value notNull = rewriter.create<mlir::db::NotOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), isNull);
               Value value = rewriter.create<mlir::db::CastOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), equal);
               equal = rewriter.create<mlir::db::AndOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), ValueRange({notNull, value}));
            }
            rewriter.create<mlir::ReturnOp>(createOp->getLoc(), equal);
         }


         Value initialVal = createOp.initial();
         AggrHtHelper helper(rewriter.getContext(),keyType,aggrType,rewriter.getUnknownLoc());
         mlir::Value compareFuncRef = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rawCompareFunc.getType(), SymbolRefAttr::get(rewriter.getStringAttr(rawCompareFunc.sym_name())));
         mlir::Value updateFuncRef = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rawUpdateFunc.getType(), SymbolRefAttr::get(rewriter.getStringAttr(rawUpdateFunc.sym_name())));
         rewriter.replaceOp(op,helper.create(rewriter,compareFuncRef,updateFuncRef,initialVal));


         return success();
      }
   }
};

static Value getArrowDataType(OpBuilder& builder, db::codegen::FunctionRegistry& functionRegistry, db::DBType type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;
   int typeConstant = 0;
   int param1 = 0;
   int param2 = 0;
   auto loc = builder.getUnknownLoc();
   if (auto intType = type.dyn_cast_or_null<mlir::db::IntType>()) {
      switch (intType.getWidth()) {
         case 8: typeConstant = arrow::Type::type::INT8; break;
         case 16: typeConstant = arrow::Type::type::INT16; break;
         case 32: typeConstant = arrow::Type::type::INT32; break;
         case 64: typeConstant = arrow::Type::type::INT64; break;
      }
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      typeConstant = arrow::Type::type::DECIMAL128;
      param1 = decimalType.getP();
      param2 = decimalType.getS();
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
      switch (floatType.getWidth()) {
         case 16: typeConstant = arrow::Type::type::HALF_FLOAT; break;
         case 32: typeConstant = arrow::Type::type::FLOAT; break;
         case 64: typeConstant = arrow::Type::type::DOUBLE; break;
      }
   } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
      typeConstant = arrow::Type::type::STRING;
   } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
      if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
         typeConstant = arrow::Type::type::DATE32;
      } else {
         typeConstant = arrow::Type::type::DATE64;
      }
   }
   //TODO: also implement date types etc

   Value arrowTypeConstant = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(typeConstant));
   Value arrowTypeParam1 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param1));
   Value arrowTypeParam2 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param2));

   Value arrowType = functionRegistry.call(builder, FunctionId::ArrowGetType2Param, ValueRange({arrowTypeConstant, arrowTypeParam1, arrowTypeParam2}))[0];
   return arrowType;
}
class BuilderBuildLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;
   mlir::Value nextPow2(OpBuilder& builder, mlir::Value v) const {
      auto loc = builder.getUnknownLoc();
      Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value c2 = builder.create<arith::ConstantIndexOp>(loc, 2);
      Value c4 = builder.create<arith::ConstantIndexOp>(loc, 4);
      Value c8 = builder.create<arith::ConstantIndexOp>(loc, 8);
      Value c16 = builder.create<arith::ConstantIndexOp>(loc, 16);
      Value c32 = builder.create<arith::ConstantIndexOp>(loc, 32);
      Value vs, ored;
      Value vmm = builder.create<arith::SubIOp>(loc, v, c1);
      v = vmm;
      vs = builder.create<arith::ShRUIOp>(loc, vmm, c1);
      ored = builder.create<arith::OrIOp>(loc, v, vs);
      v = ored;
      vs = builder.create<arith::ShRUIOp>(loc, vmm, c2);
      ored = builder.create<arith::OrIOp>(loc, v, vs);
      v = ored;
      vs = builder.create<arith::ShRUIOp>(loc, vmm, c4);
      ored = builder.create<arith::OrIOp>(loc, v, vs);
      v = ored;
      vs = builder.create<arith::ShRUIOp>(loc, vmm, c8);
      ored = builder.create<arith::OrIOp>(loc, v, vs);
      v = ored;
      vs = builder.create<arith::ShRUIOp>(loc, vmm, c16);
      ored = builder.create<arith::OrIOp>(loc, v, vs);
      v = ored;
      vs = builder.create<arith::ShRUIOp>(loc, vmm, c32);
      ored = builder.create<arith::OrIOp>(loc, v, vs);
      v = builder.create<arith::AddIOp>(loc, ored, c1);
      return v;
   }

   public:
   explicit BuilderBuildLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BuilderBuild::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::BuilderBuildAdaptor buildAdaptor(operands);
      auto buildOp = cast<mlir::db::BuilderBuild>(op);
      if (auto tableBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         Value table = functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, table);
      } else if (auto vectorBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         rewriter.replaceOp(op, buildAdaptor.builder());
      } else if (auto aggrHTBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::AggrHTBuilderType>()) {
         if (aggrHTBuilderType.getKeyType().getTypes().empty()) {
            auto builderUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), buildAdaptor.builder().getType().cast<mlir::TupleType>().getTypes(), buildAdaptor.builder())->getResults();
            rewriter.replaceOp(op, builderUnpacked[0]);
         } else {
           AggrHtHelper helper(rewriter.getContext(),aggrHTBuilderType.getKeyType(),aggrHTBuilderType.getAggrType(),   rewriter.getUnknownLoc());
           rewriter.replaceOp(op, helper.build(rewriter,buildAdaptor.builder()));
         }
      } else if (auto joinHtBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::JoinHTBuilderType>()) {
         Type kvType = TupleType::get(getContext(), {joinHtBuilderType.getKeyType(), joinHtBuilderType.getValType()});
         Type entryType = TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), kvType});

         Type typedPtrType = typeConverter->convertType(util::GenericMemrefType::get(rewriter.getContext(), entryType, -1));
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({rewriter.getIndexType(), rewriter.getIndexType(), typedPtrType}), buildAdaptor.builder());
         Value len = unpacked.getResult(0);
         Value vec = unpacked.getResult(2);
         Value zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
         Value maxValue = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0xFFFFFFFFFFFFFFFF);

         Value one = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
         auto loc = op->getLoc();
         Value htSize = nextPow2(rewriter, len);
         Value htMask = rewriter.create<arith::SubIOp>(loc, htSize, one);

         Value ht = rewriter.create<mlir::util::AllocOp>(loc, util::GenericMemrefType::get(rewriter.getContext(), rewriter.getIndexType(), -1), htSize);
         rewriter.create<scf::ForOp>(
            loc, zero, htSize, one, ValueRange({}),
            [&](OpBuilder& b, Location loc2, Value iv, ValueRange args) {
               b.create<util::StoreOp>(loc2, maxValue, ht, iv);
               b.create<scf::YieldOp>(loc);
            });
         rewriter.create<scf::ForOp>(
            loc, zero, len, one, ValueRange({}),
            [&](OpBuilder& b, Location loc2, Value iv, ValueRange args) {
               auto currVal = b.create<util::LoadOp>(loc, entryType, vec, iv);
               auto unpacked2 = b.create<util::UnPackOp>(loc, TypeRange({rewriter.getIndexType(), typeConverter->convertType(kvType)}), currVal);
               Value buckedPos = b.create<arith::AndIOp>(loc, htMask, unpacked2.getResult(0));

               Value previousPtr = b.create<util::LoadOp>(loc, rewriter.getIndexType(), ht, buckedPos);
               b.create<util::StoreOp>(loc2, iv, ht, buckedPos);
               auto repacked = b.create<util::PackOp>(loc, typeConverter->convertType(entryType), ValueRange{previousPtr, unpacked2.getResult(1)});
               b.create<util::StoreOp>(loc2, repacked, vec, iv);
               b.create<scf::YieldOp>(loc);
            });
         mlir::Value packed = rewriter.create<util::PackOp>(loc, TupleType::get(getContext(), {vec.getType(), len.getType(), ht.getType(), htMask.getType()}), ValueRange{vec, len, ht, htMask});
         rewriter.replaceOp(op, packed);
      } else if (auto joinHtBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::MarkableJoinHTBuilderType>()) {
         Type kvType = TupleType::get(getContext(), {joinHtBuilderType.getKeyType(), joinHtBuilderType.getValType()});
         Type entryType = TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), rewriter.getIndexType(), kvType});

         Type typedPtrType = typeConverter->convertType(util::GenericMemrefType::get(rewriter.getContext(), entryType, -1));
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({rewriter.getIndexType(), rewriter.getIndexType(), typedPtrType}), buildAdaptor.builder());
         Value len = unpacked.getResult(0);
         Value vec = unpacked.getResult(2);
         Value zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
         Value maxValue = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0xFFFFFFFFFFFFFFFF);

         Value one = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
         auto loc = op->getLoc();
         Value htSize = nextPow2(rewriter, len);
         Value htMask = rewriter.create<arith::SubIOp>(loc, htSize, one);

         Value ht = rewriter.create<mlir::util::AllocOp>(loc, util::GenericMemrefType::get(rewriter.getContext(), rewriter.getIndexType(), -1), htSize);
         rewriter.create<scf::ForOp>(
            loc, zero, htSize, one, ValueRange({}),
            [&](OpBuilder& b, Location loc2, Value iv, ValueRange args) {
               b.create<util::StoreOp>(loc2, maxValue, ht, iv);
               b.create<scf::YieldOp>(loc);
            });
         rewriter.create<scf::ForOp>(
            loc, zero, len, one, ValueRange({}),
            [&](OpBuilder& b, Location loc2, Value iv, ValueRange args) {
               auto currVal = b.create<util::LoadOp>(loc, entryType, vec, iv);
               auto unpacked2 = b.create<util::UnPackOp>(loc, TypeRange({rewriter.getIndexType(), rewriter.getIndexType(), typeConverter->convertType(kvType)}), currVal);
               Value buckedPos = b.create<arith::AndIOp>(loc, htMask, unpacked2.getResult(1));

               Value previousPtr = b.create<util::LoadOp>(loc, rewriter.getIndexType(), ht, buckedPos);
               b.create<util::StoreOp>(loc2, iv, ht, buckedPos);
               auto repacked = b.create<util::PackOp>(loc, typeConverter->convertType(entryType), ValueRange{unpacked2.getResult(0), previousPtr, unpacked2.getResult(2)});
               b.create<util::StoreOp>(loc2, repacked, vec, iv);
               b.create<scf::YieldOp>(loc);
            });
         mlir::Value packed = rewriter.create<util::PackOp>(loc, TupleType::get(getContext(), {vec.getType(), len.getType(), ht.getType(), htMask.getType()}), ValueRange{vec, len, ht, htMask});
         rewriter.replaceOp(op, packed);
      }

      return success();
   }
};
class CreateTableBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateTableBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateTableBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto createTB = cast<mlir::db::CreateTableBuilder>(op);
      auto loc = rewriter.getUnknownLoc();

      Value schema = functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaCreate, {})[0];
      TupleType rowType = createTB.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
      size_t i = 0;
      for (auto c : createTB.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         auto dbType = rowType.getType(i).cast<mlir::db::DBType>();
         auto arrowType = getArrowDataType(rewriter, functionRegistry, dbType);
         auto columnName = rewriter.create<mlir::db::ConstantOp>(loc, mlir::db::StringType::get(rewriter.getContext(), false), stringAttr);
         Value typeNullable = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), dbType.isNullable()));

         functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaAddField, ValueRange({schema, arrowType, typeNullable, columnName}));
         i += 1;
      }
      schema = functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaBuild, schema)[0];
      Value tableBuilder = functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderCreate, schema)[0];
      rewriter.replaceOp(op, tableBuilder);
      return success();
   }
};
class CreateJoinHtBuilderLowering : public ConversionPattern {
   public:
   explicit CreateJoinHtBuilderLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateJoinHTBuilder::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto createOp = mlir::cast<mlir::db::CreateJoinHTBuilder>(op);
      auto builderType = createOp.builder().getType().cast<mlir::db::JoinHTBuilderType>();
      auto entryType = mlir::TupleType::get(rewriter.getContext(), {builderType.getKeyType(), builderType.getValType()});
      auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1024);
      VectorHelper helper(typeConverter->convertType(tupleType), op->getLoc());
      rewriter.replaceOp(op, helper.create(rewriter, initialCapacity));
      return success();
   }
};
class CreateMarkableJoinHtBuilderLowering : public ConversionPattern {
   public:
   explicit CreateMarkableJoinHtBuilderLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateMarkableJoinHTBuilder::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto createOp = mlir::cast<mlir::db::CreateMarkableJoinHTBuilder>(op);
      auto builderType = createOp.builder().getType().cast<mlir::db::MarkableJoinHTBuilderType>();
      auto entryType = mlir::TupleType::get(rewriter.getContext(), {builderType.getKeyType(), builderType.getValType()});
      auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), rewriter.getIndexType(), entryType});
      Value initialCapacity = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1024);
      VectorHelper helper(typeConverter->convertType(tupleType), op->getLoc());
      rewriter.replaceOp(op, helper.create(rewriter, initialCapacity));
      return success();
   }
};
} // namespace
void mlir::db::populateBuilderToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateTableBuilderLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateVectorBuilderLowering>(typeConverter, patterns.getContext());
   patterns.insert<CreateAggrHTBuilderLowering>( typeConverter, patterns.getContext());
   patterns.insert<BuilderMergeLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<BuilderBuildLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateJoinHtBuilderLowering>(typeConverter, patterns.getContext());
   patterns.insert<CreateMarkableJoinHtBuilderLowering>(typeConverter, patterns.getContext());
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::AggrHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::AggrHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::JoinHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::JoinHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::MarkableJoinHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::MarkableJoinHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::VectorBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::VectorBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::TableBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addConversion([&](mlir::db::TableBuilderType tableType) {
      return mlir::util::GenericMemrefType::get(patterns.getContext(), IntegerType::get(patterns.getContext(), 8), llvm::Optional<int64_t>());
   });
   typeConverter.addConversion([&](mlir::db::VectorBuilderType vectorBuilderType) {
      return VectorHelper::createType(patterns.getContext(), typeConverter.convertType(vectorBuilderType.getElementType()));
   });
   typeConverter.addConversion([&](mlir::db::AggrHTBuilderType aggrHtBuilderType) {
      auto updateFnType = FunctionType::get(patterns.getContext(), {aggrHtBuilderType.getAggrType(), aggrHtBuilderType.getValType()}, {aggrHtBuilderType.getAggrType()});
      auto compareFnType = FunctionType::get(patterns.getContext(), {aggrHtBuilderType.getKeyType(), aggrHtBuilderType.getKeyType()}, {mlir::db::BoolType::get(patterns.getContext())});

      if (aggrHtBuilderType.getKeyType().getTypes().empty()) {
         return (Type) TupleType::get(patterns.getContext(), {typeConverter.convertType(aggrHtBuilderType.getAggrType()), typeConverter.convertType(updateFnType)});
      } else {
         return AggrHtHelper::createType(patterns.getContext(),typeConverter.convertType(aggrHtBuilderType.getKeyType()),typeConverter.convertType(aggrHtBuilderType.getAggrType()),typeConverter.convertType(aggrHtBuilderType.getValType()),typeConverter.convertType(compareFnType), typeConverter.convertType(updateFnType));
      }
   });
   typeConverter.addConversion([&](mlir::db::JoinHTBuilderType joinHtBuilderType) {
      auto elemType = typeConverter.convertType(TupleType::get(patterns.getContext(), {IndexType::get(patterns.getContext()), TupleType::get(patterns.getContext(), {joinHtBuilderType.getKeyType(), joinHtBuilderType.getValType()})}));
      return VectorHelper::createType(patterns.getContext(), elemType);
   });
   typeConverter.addConversion([&](mlir::db::MarkableJoinHTBuilderType joinHtBuilderType) {
      auto elemType = typeConverter.convertType(TupleType::get(patterns.getContext(), {IndexType::get(patterns.getContext()), IndexType::get(patterns.getContext()), TupleType::get(patterns.getContext(), {joinHtBuilderType.getKeyType(), joinHtBuilderType.getValType()})}));
      return VectorHelper::createType(patterns.getContext(), elemType); });
}
