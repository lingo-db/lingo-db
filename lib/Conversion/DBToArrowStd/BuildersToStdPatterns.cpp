#include "arrow/type_fwd.h"

#include "mlir/Conversion/DBToArrowStd/ArrowTypes.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

class BufferHelper {
   OpBuilder& b;
   ModuleOp module;
   Location loc;

   public:
   BufferHelper(OpBuilder& b, Location loc, ModuleOp module) : b(b), module(module), loc(loc) {
   }
   FuncOp getCpyFn() {
      auto func = module.lookupSymbol<FuncOp>("rt_cpy");
      if (func)
         return func;
      OpBuilder::InsertionGuard insertionGuard(b);
      b.setInsertionPointToStart(module.getBody());
      FuncOp funcOp = b.create<FuncOp>(module.getLoc(), "rt_cpy", b.getFunctionType(TypeRange({ptrType(), ptrType()}), TypeRange()), b.getStringAttr("private"));
      return funcOp;
   }
   FuncOp getFillFn() {
      auto func = module.lookupSymbol<FuncOp>("rt_fill");
      if (func)
         return func;
      OpBuilder::InsertionGuard insertionGuard(b);
      b.setInsertionPointToStart(module.getBody());
      FuncOp funcOp = b.create<FuncOp>(module.getLoc(), "rt_fill", b.getFunctionType(TypeRange({ptrType(), b.getIntegerType(8)}), TypeRange()), b.getStringAttr("private"));
      return funcOp;
   }
   void copy(Value to, Value from) {
      b.create<CallOp>(loc, getCpyFn(), ValueRange({to, from}));
   }

   Type ptrType() {
      return util::RefType::get(b.getContext(), b.getIntegerType(8), -1);
   }
   Value resize(Value old, Value newLen) {
      Value newMemref = b.create<mlir::util::AllocOp>(loc, old.getType(), newLen);
      Value toPtr = b.create<util::GenericMemrefCastOp>(loc, ptrType(), newMemref);
      Value fromPtr = b.create<util::GenericMemrefCastOp>(loc, ptrType(), old);
      copy(toPtr, fromPtr);
      b.create<mlir::util::DeAllocOp>(loc, old);
      return newMemref;
   }
   void fill(Value memref, Value val) {
      Value toPtr = b.create<util::GenericMemrefCastOp>(loc, ptrType(), memref);
      b.create<CallOp>(loc, getFillFn(), ValueRange({toPtr, val}));
   }
};
class VectorHelper {
   Type elementType;
   Location loc;

   public:
   static Type createSType(MLIRContext* context, Type elementType) {
      return mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), mlir::util::RefType::get(context, elementType, -1)});
   }
   static Type createType(MLIRContext* context, Type elementType) {
      return mlir::util::RefType::get(context, createSType(context, elementType), llvm::Optional<int64_t>());
   }
   VectorHelper(Type elementType, Location loc) : elementType(elementType), loc(loc) {
   }
   Value create(mlir::OpBuilder& builder, Value initialCapacity) {
      Value vec = builder.create<mlir::util::AllocOp>(loc, createType(builder.getContext(), elementType), mlir::Value());

      Value initialSize = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto type = mlir::util::RefType::get(builder.getContext(), elementType, -1);
      Value memref = builder.create<mlir::util::AllocOp>(loc, type, initialCapacity);
      Value packed = builder.create<mlir::util::PackOp>(loc, ValueRange{initialSize, initialCapacity, memref});
      builder.create<mlir::util::StoreOp>(loc, packed, vec, Value());
      return vec;
   }
   void insert(mlir::OpBuilder& builder, Value vec, Value newVal) {
      BufferHelper helper(builder, loc, builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>());
      auto loaded = builder.create<mlir::util::LoadOp>(loc, createSType(builder.getContext(), elementType), vec, Value());

      auto unpacked = builder.create<util::UnPackOp>(loc, loaded);

      Value values = unpacked.getResult(2);
      Value capacity = unpacked.getResult(1);

      Value len = unpacked.getResult(0);
      Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      auto ifOp = builder.create<scf::IfOp>(
         loc, TypeRange({builder.getIndexType(), values.getType()}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{capacity, values}); }, [&](OpBuilder& b, Location loc) {
            Value two = builder.create<arith::ConstantIndexOp>(loc, 2);
            Value newCapacity = b.create<arith::MulIOp>(loc, len, two);
            b.create<scf::YieldOp>(loc, ValueRange{newCapacity,helper.resize(values,newCapacity)}); });
      capacity = ifOp.getResult(0);
      values = ifOp.getResult(1);
      builder.create<util::StoreOp>(loc, newVal, values, len);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

      Value newLen = builder.create<arith::AddIOp>(loc, len, one);

      Value updatedVector = builder.create<mlir::util::PackOp>(loc, ValueRange{newLen, capacity, values});
      builder.create<mlir::util::StoreOp>(loc, updatedVector, vec, Value());
   }
};
class AggrHtHelper {
   TupleType entryType;
   Location loc;

   public:
   static Type kvType(MLIRContext* context, Type keyType, Type aggrType) {
      return mlir::TupleType::get(context, {keyType, aggrType});
   }
   static TupleType createEntryType(MLIRContext* context, Type keyType, Type aggrType) {
      return mlir::TupleType::get(context, {IndexType::get(context), IndexType::get(context), kvType(context, keyType, aggrType)});
   }
   static Type createType(MLIRContext* context, Type keyType, Type aggrType, Type valType) {
      auto idxType = IndexType::get(context);
      auto valuesType = mlir::util::RefType::get(context, createEntryType(context, keyType, aggrType), -1);
      auto htType = mlir::util::RefType::get(context, idxType, -1);
      return mlir::TupleType::get(context, {idxType, idxType, valuesType, htType, aggrType});
   }
   AggrHtHelper(MLIRContext* context, Type keyType, Type aggrType, Location loc) : entryType(createEntryType(context, keyType, aggrType)), loc(loc) {
   }

   Value create(mlir::OpBuilder& builder, Value initialValue) {
      BufferHelper helper(builder, loc, builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>());
      Value initialCapacity = builder.create<arith::ConstantIndexOp>(loc, 4);
      Value initialHtSize = builder.create<arith::ConstantIndexOp>(loc, 8);

      Value initialSize = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto valuesType = mlir::util::RefType::get(builder.getContext(), entryType, -1);
      auto htType = mlir::util::RefType::get(builder.getContext(), builder.getIndexType(), -1);

      Value values = builder.create<mlir::util::AllocOp>(loc, valuesType, initialCapacity);
      Value ht = builder.create<mlir::util::AllocOp>(loc, htType, initialHtSize);
      Value fillValue = builder.create<arith::ConstantOp>(loc, builder.getIntegerType(8), builder.getIntegerAttr(builder.getIntegerType(8), 0xFF));
      helper.fill(ht, fillValue);

      Value vec = builder.create<mlir::util::PackOp>(loc, ValueRange{initialSize, initialCapacity, values, ht, initialValue});
      return vec;
   }
   Value compareKeys(mlir::OpBuilder& rewriter, Value left, Value right) {
      Value equal = rewriter.create<mlir::db::ConstantOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      auto leftUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, left);
      auto rightUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, right);
      for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
         Value compared = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));
         if (leftUnpacked->getResult(i).getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() && rightUnpacked.getResult(i).getType().dyn_cast_or_null<mlir::db::DBType>().isNullable()) {
            Value isNull1 = rewriter.create<mlir::db::IsNullOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), leftUnpacked->getResult(i));
            Value isNull2 = rewriter.create<mlir::db::IsNullOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), rightUnpacked->getResult(i));
            Value bothNull = rewriter.create<mlir::db::AndOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), ValueRange({isNull1, isNull2}));
            Value casted = rewriter.create<mlir::db::CastOp>(loc, compared.getType(), bothNull);
            Value tmp = rewriter.create<mlir::db::SelectOp>(loc, bothNull, casted, compared);
            compared = tmp;
         }
         Value localEqual = rewriter.create<mlir::db::AndOp>(loc, mlir::db::BoolType::get(rewriter.getContext(), equal.getType().cast<mlir::db::BoolType>().getNullable() || compared.getType().cast<mlir::db::BoolType>().getNullable()), ValueRange({equal, compared}));
         equal = localEqual;
      }
      if (equal.getType().cast<mlir::db::DBType>().isNullable()) {
         Value isNull = rewriter.create<mlir::db::IsNullOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), equal);
         Value notNull = rewriter.create<mlir::db::NotOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), isNull);
         Value value = rewriter.create<mlir::db::CastOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), equal);
         equal = rewriter.create<mlir::db::AndOp>(loc, mlir::db::BoolType::get(rewriter.getContext()), ValueRange({notNull, value}));
      }
      return equal;
   }
   Value insert(mlir::ConversionPatternRewriter& rewriter, Value aggrHtBuilder, Value key, Value val, Value hash, std::function<Value(mlir::OpBuilder&, Value, Value)> updateFn) {
      BufferHelper helper(rewriter, loc, rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>());
      auto* context = rewriter.getContext();
      auto unpacked = rewriter.create<util::UnPackOp>(loc, aggrHtBuilder);
      Value len = unpacked.getResult(0);
      Value ht = unpacked.getResult(3);
      Value values = unpacked.getResult(2);
      Value capacity = unpacked.getResult(1);

      Value initialVal = unpacked.getResult(4);
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
      Value four = rewriter.create<arith::ConstantIndexOp>(loc, 4);

      Value maxValue = rewriter.create<arith::ConstantIndexOp>(loc, 0xFFFFFFFFFFFFFFFF);

      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, len, capacity);
      auto ifOp = rewriter.create<scf::IfOp>(
         loc, TypeRange({rewriter.getIndexType(), values.getType(), ht.getType()}), cmp, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{capacity, values, ht}); }, [&](OpBuilder& b, Location loc) {
            Value newCapacity = b.create<arith::MulIOp>(loc, len, two);
            Value newHtSize = b.create<arith::MulIOp>(loc, len, four);
            Value htMask = b.create<arith::SubIOp>(loc, newHtSize, one);

            Value newValues = b.create<mlir::util::AllocOp>(loc, values.getType(), newCapacity);
            Value newHt = b.create<mlir::util::AllocOp>(loc, ht.getType(), newHtSize);
            Value fillValue = b.create<arith::ConstantOp>(loc, b.getIntegerType(8), b.getIntegerAttr(b.getIntegerType(8), 0xFF));
            helper.fill(newHt, fillValue);
            b.create<scf::ForOp>(
               loc, zero, len, one, ValueRange({}),
               [&](OpBuilder& b2, Location loc2, Value iv, ValueRange args) {
                  Value currVal = b2.create<util::LoadOp>(loc, entryType, values, iv);

                  auto unpacked2 = b2.create<util::UnPackOp>(loc, currVal);
                  Value buckedPos = b2.create<arith::AndIOp>(loc,htMask,unpacked2.getResult(1));

                  Value previousPtr = b2.create<util::LoadOp>(loc, b2.getIndexType(), newHt, buckedPos);
                  b2.create<util::StoreOp>(loc2, iv, newHt, buckedPos);
                  auto repacked= b2.create<util::PackOp>(loc,ValueRange{previousPtr,unpacked2.getResult(1),unpacked2.getResult(2)});
                  b2.create<util::StoreOp>(loc2, repacked, newValues, iv);

                  b2.create<scf::YieldOp>(loc);
               });
            b.create<mlir::util::DeAllocOp>(loc, values);
            b.create<scf::YieldOp>(loc, ValueRange{newCapacity, newValues,newHt}); });
      capacity = ifOp.getResult(0);
      values = ifOp.getResult(1);
      ht = ifOp.getResult(2);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      Value htSize = rewriter.create<arith::MulIOp>(loc, capacity, two);

      Value htMask = rewriter.create<arith::SubIOp>(loc, htSize, one);

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hash);
      //idx = hashtable[position]
      Value idx = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), ht, position);
      // ptr = &hashtable[position]
      Type idxType = rewriter.getIndexType();
      Value ptr = rewriter.create<util::ElementPtrOp>(loc, util::RefType::get(context, idxType, llvm::Optional<int64_t>()), ht, position);
      Type ptrType = util::RefType::get(context, idxType, llvm::Optional<int64_t>());
      Type doneType = rewriter.getI1Type();

      auto resultTypes = std::vector<Type>({idxType, idxType, ptrType});
      auto whileOp = rewriter.create<scf::WhileOp>(loc, resultTypes, ValueRange({len, idx, ptr}));
      Block* before = rewriter.createBlock(&whileOp.getBefore(), {}, resultTypes,{loc,loc,loc});
      Block* after = rewriter.createBlock(&whileOp.getAfter(), {}, resultTypes,{loc,loc,loc});

      // The conditional block of the while loop.
      {
         rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
         Value len = before->getArgument(0);
         Value idx = before->getArgument(1);
         Value ptr = before->getArgument(2);

         //    if (idx == 0xFFFFFFFFFFF){
         Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, idx, maxValue);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, TypeRange({idxType, doneType, idxType, ptrType}), cmp, [&](OpBuilder& b, Location loc) {
               Value newAggr = updateFn(b,initialVal, val);
               Value newKVPair = b.create<util::PackOp>(loc,ValueRange({key, newAggr}));
               //       %newEntry = ...
               Value newEntry = b.create<util::PackOp>(loc, ValueRange({maxValue, hash, newKVPair}));

               //       append(vec,newEntry)
               b.create<util::StoreOp>(loc, newEntry, values, len);

               //       *ptr=len
               b.create<util::StoreOp>(loc, len, ptr, Value());
               Value newLen = b.create<arith::AddIOp>(loc, len, one);
               //       yield 0,0,done=true
               b.create<scf::YieldOp>(loc, ValueRange{newLen, falseValue, idx, ptr}); },
            [&](OpBuilder& b, Location loc) {

         //       entry=vec[idx]
                  Value currEntry= b.create<util::LoadOp>(loc, entryType, values, idx);
                  auto entryUnpacked=b.create<util::UnPackOp>(loc,currEntry);
                  Value kv=entryUnpacked.getResult(2);
                  auto kvUnpacked=b.create<util::UnPackOp>(loc,kv);

                  Value entryKey=kvUnpacked.getResult(0);
                  Value entryAggr=kvUnpacked.getResult(1);
                  Value entryNext=entryUnpacked.getResult(0);
                  Value entryHash=entryUnpacked.getResult(1);
                  Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hash);
                  auto ifOpH = b.create<scf::IfOp>(
                     loc, TypeRange({idxType,doneType,idxType,ptrType}), hashMatches, [&](OpBuilder& b, Location loc) {
                        Value keyMatches = b.create<mlir::db::TypeCastOp>(loc,b.getI1Type(),compareKeys(b,entryKey,key));
                        auto ifOp2 = b.create<scf::IfOp>(
                           loc, TypeRange({idxType,doneType,idxType,ptrType}), keyMatches, [&](OpBuilder& b, Location loc) {
                              //          entry.aggr = update(vec.aggr,val)
                              Value newAggr= updateFn(b,entryAggr, val);
                              Value newKVPair=b.create<util::PackOp>(loc,ValueRange({entryKey,newAggr}));
                              Value newEntry=b.create<util::PackOp>(loc,ValueRange({entryNext,entryHash,newKVPair}));
                              b.create<util::StoreOp>(loc, newEntry, values, idx);
                              b.create<scf::YieldOp>(loc, ValueRange{len,falseValue, idx,ptr});
                              }, [&](OpBuilder& b, Location loc) {

                              //          ptr = &entry.next
                              Value entryPtr=b.create<util::ElementPtrOp>(loc,util::RefType::get(context,entryType,llvm::Optional<int64_t>()),values,idx);

                              ptr=b.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context,idxType,llvm::Optional<int64_t>()),entryPtr);
                              //          yield idx,ptr,done=false
                              b.create<scf::YieldOp>(loc, ValueRange{len,trueValue, entryNext, ptr });});
                        b.create<scf::YieldOp>(loc, ifOp2.getResults());
                        }, [&](OpBuilder& b, Location loc) {

                        //          ptr = &entry.next
                        Value entryPtr=b.create<util::ElementPtrOp>(loc,util::RefType::get(context,entryType,llvm::Optional<int64_t>()),values,idx);

                        ptr=b.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context,idxType,llvm::Optional<int64_t>()),entryPtr);
                        //          yield idx,ptr,done=false
                        b.create<scf::YieldOp>(loc, ValueRange{len,trueValue, entryNext, ptr });});
                  b.create<scf::YieldOp>(loc, ifOpH.getResults()); });
         //       if(compare(entry.key,key)){

         Value newLen = ifOp.getResult(0);
         Value done = ifOp.getResult(1);
         idx = ifOp.getResult(2);
         ptr = ifOp.getResult(3);
         rewriter.create<scf::ConditionOp>(loc, done,
                                           ValueRange({newLen, idx, ptr}));
      }

      // The body of the while loop: shift right until reaching a value of 0.
      {
         rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      }

      rewriter.setInsertionPointAfter(whileOp);
      Value newLen = whileOp.getResult(0);

      Value updatedBuilder = rewriter.create<mlir::util::PackOp>(loc, ValueRange{newLen, capacity, values, ht, initialVal});
      return updatedBuilder;
   }
   Value build(mlir::OpBuilder& builder, Value aggrHtBuilder) {
      auto unpacked = builder.create<util::UnPackOp>(loc, aggrHtBuilder);
      Value len = unpacked.getResult(0);
      //Value ht = unpacked.getResult(3);//todo: free
      Value values = unpacked.getResult(2);
      return builder.create<util::PackOp>(loc, ValueRange({len, values}));
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
   } else if (type.isa<mlir::db::CharType>()) {
      return FunctionId ::ArrowTableBuilderAddFixedBinary;
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
      auto loc = op->getLoc();
      if (auto tableBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         TupleType rowType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, mergeOpAdaptor.val());
         size_t i = 0;
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         for (auto v : unPackOp.vals()) {
            Value val = v;
            Value isNull;
            if (mergeOp.val().getType().cast<TupleType>().getType(i).cast<db::DBType>().isNullable()) {
               auto nullUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, val);
               isNull = nullUnpacked.getResult(0);
               val = nullUnpacked->getResult(1);
            } else {
               isNull = falseValue;
            }
            if (auto charType = mergeOp.val().getType().cast<TupleType>().getType(i).dyn_cast_or_null<mlir::db::CharType>()) {
               if (charType.getBytes() < 8) {
                  val = rewriter.create<arith::ExtSIOp>(loc, val, rewriter.getI64Type());
               }
            }
            Value columnId = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            functionRegistry.call(rewriter, loc, getStoreFunc(functionRegistry, rowType.getType(i).cast<mlir::db::DBType>()), ValueRange({mergeOpAdaptor.builder(), columnId, isNull, val}));
            i++;
         }
         functionRegistry.call(rewriter, loc, FunctionId::ArrowTableBuilderFinishRow, mergeOpAdaptor.builder());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      } else if (auto vectorBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         Value builderVal = mergeOpAdaptor.builder();
         Value v = mergeOpAdaptor.val();
         auto convertedElementType = typeConverter->convertType(vectorBuilderType.getElementType());
         VectorHelper helper(convertedElementType, op->getLoc());
         helper.insert(rewriter, builderVal, v);
         rewriter.replaceOp(op, builderVal);
      } else if (auto aggrHTBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::AggrHTBuilderType>()) {
         auto updateFnBuilder = [&mergeOp](OpBuilder& rewriter, Value left, Value right) {
            Block* sortLambda = &mergeOp.fn().front();
            auto* sortLambdaTerminator = sortLambda->getTerminator();
            mlir::BlockAndValueMapping mapping;
            mapping.map(sortLambda->getArgument(0), left);
            mapping.map(sortLambda->getArgument(1), right);

            for (auto& op : sortLambda->getOperations()) {
               if (&op != sortLambdaTerminator) {
                  rewriter.clone(op, mapping);
               }
            }
            return mapping.lookup(cast<mlir::db::YieldOp>(sortLambdaTerminator).results()[0]);
         };
         if (aggrHTBuilderType.getKeyType().getTypes().empty()) {
            auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, mergeOpAdaptor.val())->getResults();
            auto newAggr = updateFnBuilder(rewriter, mergeOpAdaptor.builder(), unPacked[1]);
            rewriter.replaceOp(op, newAggr);
         } else {
            auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, mergeOp.val())->getResults();
            Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), unPacked[0]);
            Value currKey = unPacked[0];
            Value currVal = unPacked[1];

            AggrHtHelper helper(rewriter.getContext(), aggrHTBuilderType.getKeyType(), aggrHTBuilderType.getAggrType(), loc);
            rewriter.replaceOp(op, helper.insert(rewriter, mergeOpAdaptor.builder(), currKey, currVal, hashed, updateFnBuilder));
         }
      } else if (auto joinHtBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::JoinHTBuilderType>()) {
         Value v = mergeOpAdaptor.val();
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), v.getType()});

         auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, mergeOpAdaptor.val())->getResults();
         Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), unPacked[0]);
         auto bucket = rewriter.create<mlir::util::PackOp>(loc, mlir::ValueRange({hashed, v}));
         VectorHelper helper(typeConverter->convertType(tupleType), op->getLoc());
         helper.insert(rewriter, mergeOpAdaptor.builder(), bucket);
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      }

      return success();
   }
};

class CreateAggrHTBuilderLowering : public ConversionPattern {
   public:
   explicit CreateAggrHTBuilderLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateAggrHTBuilder::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto createOp = cast<mlir::db::CreateAggrHTBuilder>(op);

      TupleType keyType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getKeyType();
      TupleType aggrType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getAggrType();

      if (keyType.getTypes().empty()) {
         rewriter.replaceOp(op, createOp.initial());
         return success();
      } else {
         Value initialVal = createOp.initial();
         AggrHtHelper helper(rewriter.getContext(), keyType, aggrType, op->getLoc());
         rewriter.replaceOp(op, helper.create(rewriter, initialVal));
         return success();
      }
   }
};

static Value getArrowDataType(OpBuilder& builder, Location loc, db::codegen::FunctionRegistry& functionRegistry, db::DBType type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;

   auto [typeConstant, param1, param2] = db::codegen::convertTypeToArrow(type);
   //TODO: also implement date types etc

   Value arrowTypeConstant = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(typeConstant));
   Value arrowTypeParam1 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param1));
   Value arrowTypeParam2 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param2));

   Value arrowType = functionRegistry.call(builder, loc, FunctionId::ArrowGetType2Param, ValueRange({arrowTypeConstant, arrowTypeParam1, arrowTypeParam2}))[0];
   return arrowType;
}
class BuilderBuildLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit BuilderBuildLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BuilderBuild::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::BuilderBuildAdaptor buildAdaptor(operands);
      auto buildOp = cast<mlir::db::BuilderBuild>(op);
      if (auto tableBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         Value table = functionRegistry.call(rewriter, op->getLoc(), FunctionId::ArrowTableBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, table);
      } else if (auto vectorBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         rewriter.replaceOp(op, buildAdaptor.builder());
      } else if (auto aggrHTBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::AggrHTBuilderType>()) {
         if (aggrHTBuilderType.getKeyType().getTypes().empty()) {
            rewriter.replaceOp(op, buildAdaptor.builder());
         } else {
            AggrHtHelper helper(rewriter.getContext(), aggrHTBuilderType.getKeyType(), aggrHTBuilderType.getAggrType(), op->getLoc());
            rewriter.replaceOp(op, helper.build(rewriter, buildAdaptor.builder()));
         }
      } else if (auto joinHtBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::JoinHTBuilderType>()) {
         ModuleOp module = op->getParentOfType<ModuleOp>();
         FuncOp buildFn = module.lookupSymbol<FuncOp>("build_join_ht");
         auto idxType=rewriter.getIndexType();
         auto lowBType=mlir::util::RefType::get(op->getContext(),mlir::TupleType::get(op->getContext(),{idxType,idxType,mlir::util::RefType::get(op->getContext(),idxType,-1)}),llvm::Optional<int64_t>());
         if (!buildFn) {
            {
               auto loc = op->getLoc();
               auto htType = util::RefType::get(rewriter.getContext(), idxType, -1);

               mlir::OpBuilder::InsertionGuard g(rewriter);
               rewriter.setInsertionPointToStart(&module.body().front());
               buildFn = rewriter.create<FuncOp>(op->getLoc(), "build_join_ht", rewriter.getFunctionType({idxType, lowBType}, {htType, idxType}));
               buildFn->setAttr("passthrough", rewriter.getArrayAttr({rewriter.getStringAttr("noinline"), rewriter.getStringAttr("optnone")}));
               mlir::Block* functionBlock = new mlir::Block;
               functionBlock->addArguments({idxType, lowBType},{loc,loc});
               buildFn.body().push_back(functionBlock);
               rewriter.setInsertionPointToStart(functionBlock);
               Value const8 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 8);
               Value elemSize = rewriter.create<mlir::arith::DivUIOp>(loc, functionBlock->getArgument(0), const8);
               auto loaded = rewriter.create<mlir::util::LoadOp>(op->getLoc(), lowBType.getElementType(), functionBlock->getArgument(1), Value());
               auto unpacked = rewriter.create<util::UnPackOp>(op->getLoc(), loaded);
               Value numValues = unpacked.getResult(0);
               Value vec = unpacked.getResult(2);
               Value zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);

               Value one = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
               Value htSize = functionRegistry.call(rewriter, op->getLoc(), FunctionId::NextPow2, numValues)[0];
               Value htMask = rewriter.create<arith::SubIOp>(loc, htSize, one);

               Value ht = rewriter.create<mlir::util::AllocOp>(loc, htType, htSize);
               Value fillValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getIntegerAttr(rewriter.getIntegerType(8), 0xFF));
               BufferHelper helper(rewriter, loc, rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>());
               helper.fill(ht, fillValue);
               rewriter.create<scf::ForOp>(
                  loc, zero, numValues, one, ValueRange({}),
                  [&](OpBuilder& b, Location loc2, Value iv, ValueRange args) {
                     Value vecPos = b.create<mlir::arith::MulIOp>(loc, iv, elemSize);
                     auto currVal = b.create<util::LoadOp>(loc, idxType, vec, vecPos);
                     Value buckedPos = b.create<arith::AndIOp>(loc, htMask, currVal);
                     Value previousPtr = b.create<util::LoadOp>(loc, rewriter.getIndexType(), ht, buckedPos);
                     b.create<util::StoreOp>(loc2, iv, ht, buckedPos);
                     b.create<util::StoreOp>(loc2, previousPtr, vec, vecPos);
                     b.create<scf::YieldOp>(loc);
                  });
               rewriter.create<mlir::ReturnOp>(op->getLoc(), ValueRange({ht, htMask}));
            }
         }

         Type kvType = TupleType::get(getContext(), {joinHtBuilderType.getKeyType(), joinHtBuilderType.getValType()});
         Type entryType = TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), kvType});
         auto elemsize = rewriter.create<mlir::util::SizeOfOp>(op->getLoc(), rewriter.getIndexType(), entryType);
         auto loaded = rewriter.create<mlir::util::LoadOp>(op->getLoc(), VectorHelper::createSType(rewriter.getContext(), entryType), buildAdaptor.builder(), Value());
         auto unpacked = rewriter.create<util::UnPackOp>(op->getLoc(), loaded);
         Value len = unpacked.getResult(0);
         Value vec = unpacked.getResult(2);
         Value toLB = rewriter.create<util::GenericMemrefCastOp>(op->getLoc(), lowBType, buildAdaptor.builder());

         auto called = rewriter.create<mlir::CallOp>(op->getLoc(), buildFn, ValueRange({elemsize, toLB}));
         mlir::Value packed = rewriter.create<util::PackOp>(op->getLoc(), ValueRange{vec, len, called->getResult(0), called->getResult(1)});
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
      auto loc = op->getLoc();

      Value schema = functionRegistry.call(rewriter, loc, FunctionId::ArrowTableSchemaCreate, {})[0];
      TupleType rowType = createTB.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
      size_t i = 0;
      for (auto c : createTB.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         auto dbType = rowType.getType(i).cast<mlir::db::DBType>();
         auto arrowType = getArrowDataType(rewriter, op->getLoc(), functionRegistry, dbType);
         Value columnNameRef = mlir::db::createStringConstant(loc, rewriter, op->getParentOfType<ModuleOp>(), stringAttr.getValue().str());
         Value typeNullable = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), dbType.isNullable()));

         functionRegistry.call(rewriter, loc, FunctionId::ArrowTableSchemaAddField, ValueRange({schema, arrowType, typeNullable, columnNameRef}));
         i += 1;
      }
      schema = functionRegistry.call(rewriter, loc, FunctionId::ArrowTableSchemaBuild, schema)[0];
      Value tableBuilder = functionRegistry.call(rewriter, loc, FunctionId::ArrowTableBuilderCreate, schema)[0];
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
} // namespace
void mlir::db::populateBuilderToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateTableBuilderLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateVectorBuilderLowering>(typeConverter, patterns.getContext());
   patterns.insert<CreateAggrHTBuilderLowering>(typeConverter, patterns.getContext());
   patterns.insert<BuilderMergeLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<BuilderBuildLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateJoinHtBuilderLowering>(typeConverter, patterns.getContext());
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
      return mlir::util::RefType::get(patterns.getContext(), IntegerType::get(patterns.getContext(), 8), llvm::Optional<int64_t>());
   });
   typeConverter.addConversion([&](mlir::db::VectorBuilderType vectorBuilderType) {
      return VectorHelper::createType(patterns.getContext(), typeConverter.convertType(vectorBuilderType.getElementType()));
   });
   typeConverter.addConversion([&](mlir::db::AggrHTBuilderType aggrHtBuilderType) {
      if (aggrHtBuilderType.getKeyType().getTypes().empty()) {
         return (Type) typeConverter.convertType(aggrHtBuilderType.getAggrType());
      } else {
         return AggrHtHelper::createType(patterns.getContext(), typeConverter.convertType(aggrHtBuilderType.getKeyType()), typeConverter.convertType(aggrHtBuilderType.getAggrType()), typeConverter.convertType(aggrHtBuilderType.getValType()));
      }
   });
   typeConverter.addConversion([&](mlir::db::JoinHTBuilderType joinHtBuilderType) {
      auto elemType = typeConverter.convertType(TupleType::get(patterns.getContext(), {IndexType::get(patterns.getContext()), TupleType::get(patterns.getContext(), {joinHtBuilderType.getKeyType(), joinHtBuilderType.getValType()})}));
      return VectorHelper::createType(patterns.getContext(), elemType);
   });
}
