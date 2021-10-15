#include "arrow/type_fwd.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {
class CreateVectorBuilderLowering : public ConversionPattern {

   public:
   explicit CreateVectorBuilderLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateVectorBuilder::getOperationName(), 1, context){}

   LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto vecBuilderOp=mlir::cast<mlir::db::CreateVectorBuilder>(op);
      Value initialCapacity=rewriter.create<arith::ConstantIndexOp>(op->getLoc(),2);
      Value initialSize=rewriter.create<arith::ConstantIndexOp>(op->getLoc(),0);

      auto type=mlir::util::GenericMemrefType::get(getContext(), typeConverter->convertType(vecBuilderOp.getType().getElementType()), -1);
      Value memref = rewriter.create<mlir::util::AllocOp>(op->getLoc(),type,initialCapacity);
      Value builder=rewriter.create<mlir::util::PackOp>(op->getLoc(),typeConverter->convertType(vecBuilderOp.getType()),ValueRange{initialSize,memref});
      rewriter.replaceOp(op, builder);
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
/*
Value serializeForVector(OpBuilder& builder, TypeConverter* converter, Value vectorBuilder, Value element, Type type, db::codegen::FunctionRegistry& functionRegistry) {
   if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
      if (originalTupleType.getTypes().empty()) {
         return element;
      }
      auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
      std::vector<Value> serializedValues;
      std::vector<Type> types;
      auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
      for (size_t i = 0; i < tupleType.size(); i++) {
         Value currVal = unPackOp.getResult(i);
         Value serialized = serializeForVector(builder, converter, vectorBuilder, currVal, originalTupleType.getType(i), functionRegistry);
         serializedValues.push_back(serialized);
         types.push_back(serialized.getType());
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
   } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
      if (stringType.isNullable()) {
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())}), element);
         auto persisted = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::VectorBuilderAddNullableVarLen, ValueRange({vectorBuilder, unPackOp.getResult(0), unPackOp.getResult(1)}))[0];
         return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())})), ValueRange({unPackOp.getResult(0), persisted}));
      } else {
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::VectorBuilderAddVarLen, ValueRange({vectorBuilder, element}))[0];
      }
   } else {
      return element;
   }
}
Value serializeForAggrHT(OpBuilder& builder, TypeConverter* converter, Value vectorBuilder, Value element, Type type, db::codegen::FunctionRegistry& functionRegistry) {
   if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
      if (originalTupleType.getTypes().empty()) {
         return element;
      }
      auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
      std::vector<Value> serializedValues;
      std::vector<Type> types;
      auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
      for (size_t i = 0; i < tupleType.size(); i++) {
         Value currVal = unPackOp.getResult(i);
         Value serialized = serializeForAggrHT(builder, converter, vectorBuilder, currVal, originalTupleType.getType(i), functionRegistry);
         serializedValues.push_back(serialized);
         types.push_back(serialized.getType());
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
   } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
      if (stringType.isNullable()) {
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())}), element);
         auto persisted = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::AggrHtBuilderAddNullableVarLen, ValueRange({vectorBuilder, unPackOp.getResult(0), unPackOp.getResult(1)}))[0];
         return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())})), ValueRange({unPackOp.getResult(0), persisted}));
      } else {
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::AggrHtBuilderAddVarLen, ValueRange({vectorBuilder, element}))[0];
      }
   } else {
      return element;
   }
}

Value serializeForJoinHT(OpBuilder& builder, TypeConverter* converter, Value vectorBuilder, Value element, Type type, db::codegen::FunctionRegistry& functionRegistry) {
   if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
      if (originalTupleType.getTypes().empty()) {
         return element;
      }
      auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
      std::vector<Value> serializedValues;
      std::vector<Type> types;
      auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
      for (size_t i = 0; i < tupleType.size(); i++) {
         Value currVal = unPackOp.getResult(i);
         Value serialized = serializeForJoinHT(builder, converter, vectorBuilder, currVal, originalTupleType.getType(i), functionRegistry);
         serializedValues.push_back(serialized);
         types.push_back(serialized.getType());
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
   } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
      if (stringType.isNullable()) {
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())}), element);
         auto persisted = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::JoinHtBuilderAddNullableVarLen, ValueRange({vectorBuilder, unPackOp.getResult(0), unPackOp.getResult(1)}))[0];
         return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())})), ValueRange({unPackOp.getResult(0), persisted}));
      } else {
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::JoinHtBuilderAddVarLen, ValueRange({vectorBuilder, element}))[0];
      }
   } else {
      return element;
   }
}
Value serializeForMJoinHT(OpBuilder& builder, TypeConverter* converter, Value vectorBuilder, Value element, Type type, db::codegen::FunctionRegistry& functionRegistry) {
   if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
      if (originalTupleType.getTypes().empty()) {
         return element;
      }
      auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
      std::vector<Value> serializedValues;
      std::vector<Type> types;
      auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
      for (size_t i = 0; i < tupleType.size(); i++) {
         Value currVal = unPackOp.getResult(i);
         Value serialized = serializeForMJoinHT(builder, converter, vectorBuilder, currVal, originalTupleType.getType(i), functionRegistry);
         serializedValues.push_back(serialized);
         types.push_back(serialized.getType());
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
   } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
      if (stringType.isNullable()) {
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())}), element);
         auto persisted = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::MJoinHtBuilderAddNullableVarLen, ValueRange({vectorBuilder, unPackOp.getResult(0), unPackOp.getResult(1)}))[0];
         return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())})), ValueRange({unPackOp.getResult(0), persisted}));
      } else {
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::MJoinHtBuilderAddVarLen, ValueRange({vectorBuilder, element}))[0];
      }
   } else {
      return element;
   }
}*/
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
         Value builderVal=mergeOpAdaptor.builder();
         Value v = mergeOpAdaptor.val();//serializeForVector(rewriter, typeConverter, mergeOpAdaptor.builder(), mergeOpAdaptor.val(), mergeOp.val().getType(), functionRegistry);
         auto convertedElementType=typeConverter->convertType(vectorBuilderType.getElementType());
          Type typedPtrType = util::GenericMemrefType::get(rewriter.getContext(), convertedElementType, -1);
         auto unpacked = rewriter.create<util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange({rewriter.getIndexType(), typedPtrType}), builderVal);

         Value values=unpacked.getResult(1);
         Value len=unpacked.getResult(0);
         Value capacity = rewriter.create<util::DimOp>(rewriter.getUnknownLoc(),rewriter.getIndexType(), values);
         Value cmp = rewriter.create<arith::CmpIOp>(rewriter.getUnknownLoc(),arith::CmpIPredicate::ult,len,capacity);
         auto ifOp=    rewriter.create<scf::IfOp>(loc, values.getType(),cmp, [&](OpBuilder &b, Location loc) {
            b.create<scf::YieldOp>(loc, ValueRange{values});
            },[&](OpBuilder &b, Location loc) {
            Value zero=rewriter.create<arith::ConstantIndexOp>(op->getLoc(),0);
            Value one=rewriter.create<arith::ConstantIndexOp>(op->getLoc(),1);
            Value two=rewriter.create<arith::ConstantIndexOp>(op->getLoc(),2);
            Value newCapacity=  b.create<arith::MulIOp>(op->getLoc(),len,two);
            Value newMemref = rewriter.create<mlir::util::AllocOp>(op->getLoc(),values.getType(),newCapacity);
            auto loop = b.create<scf::ForOp>(
               loc, zero, len, one,ValueRange({}),
               [&](OpBuilder& b2, Location loc2, Value iv, ValueRange args) {
                  Value currVal = b.create<util::LoadOp>(loc, convertedElementType, values, iv);
                  rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), currVal, newMemref,iv);
                  b.create<scf::YieldOp>(loc);
               });
            b.create<mlir::util::DeAllocOp>(op->getLoc(),values);
            b.create<scf::YieldOp>(loc, ValueRange{newMemref});

         });
         values=ifOp.getResult(0);
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), v, values,len);
         Value one=rewriter.create<arith::ConstantIndexOp>(op->getLoc(),1);

         Value newLen=rewriter.create<arith::AddIOp>(op->getLoc(),len,one);

         Value updatedBuilder=rewriter.create<mlir::util::PackOp>(op->getLoc(),TupleType::get(getContext(),{rewriter.getIndexType(), typedPtrType}),ValueRange{newLen,values});

         rewriter.replaceOp(op, updatedBuilder);
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
            auto ptrType = mlir::util::GenericMemrefType::get(getContext(), IntegerType::get(getContext(), 8), llvm::Optional<int64_t>());
            auto updateFnType = FunctionType::get(getContext(), {aggrHTBuilderType.getAggrType(), aggrHTBuilderType.getValType()}, {aggrHTBuilderType.getAggrType()});
            auto compareFnType = FunctionType::get(getContext(), {aggrHTBuilderType.getKeyType(), aggrHTBuilderType.getKeyType()}, {mlir::db::BoolType::get(getContext())});
            auto builderUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, TypeRange{ptrType, typeConverter->convertType(compareFnType), typeConverter->convertType(updateFnType)}, mergeOpAdaptor.builder())->getResults();
            auto builderPtr = builderUnpacked[0];

            auto loweredTypes = mergeOpAdaptor.val().getType().cast<TupleType>().getTypes();

            auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, loweredTypes, mergeOpAdaptor.val())->getResults();
            auto higherType = rewriter.create<mlir::db::TypeCastOp>(loc, mergeOp.val().getType().cast<TupleType>().getType(0), unPacked[0]);
            Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), higherType);
            Value currKey = unPacked[0];
            Value currVal = unPacked[1];

            Type keyMemrefType = util::GenericMemrefType::get(rewriter.getContext(), currKey.getType(), llvm::Optional<int64_t>());
            Type valMemrefType = util::GenericMemrefType::get(rewriter.getContext(), currVal.getType(), llvm::Optional<int64_t>());
            Value allocaKey, allocaVal;
            {
               OpBuilder::InsertionGuard insertionGuard(rewriter);
               auto func = op->getParentOfType<mlir::FuncOp>();
               rewriter.setInsertionPointToStart(&func.getBody().front());
               allocaKey = rewriter.create<mlir::util::AllocaOp>(loc, keyMemrefType, Value());
               allocaVal = rewriter.create<mlir::util::AllocaOp>(loc, valMemrefType, Value());
            }

            Value plainMemrefKey = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, ptrType, allocaKey);
            Value plainMemrefVal = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, ptrType, allocaVal);
            Value tuple = functionRegistry.call(rewriter, FunctionId::AggrHtBuilderFastLookup, ValueRange({builderPtr, hashed}))[0];
            auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, TypeRange{rewriter.getI1Type(), ptrType}, tuple)->getResults();
            auto newIfOp = rewriter.create<mlir::scf::IfOp>(loc, TypeRange(), unpacked[0], true);
            {
               OpBuilder::InsertionGuard insertionGuard(rewriter);
               scf::IfOp::ensureTerminator(newIfOp.thenRegion(), rewriter, loc);
               rewriter.setInsertionPointToStart(&newIfOp.thenRegion().front());
               auto keyType = typeConverter->convertType(aggrHTBuilderType.getKeyType());
               auto aggrType = typeConverter->convertType(aggrHTBuilderType.getAggrType());
               auto tupleType = TupleType::get(getContext(), {keyType, aggrType});
               Value genericMemref = rewriter.create<util::GenericMemrefCastOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), tupleType, llvm::Optional<int64_t>()), unpacked[1]);
               Value loaded = rewriter.create<util::LoadOp>(rewriter.getUnknownLoc(), tupleType, genericMemref, Value());
               auto unpackedTuple = rewriter.create<mlir::util::UnPackOp>(loc, TypeRange{keyType, aggrType}, loaded)->getResults();
               auto keyMatches = rewriter.create<mlir::CallIndirectOp>(loc, builderUnpacked[1], ValueRange({unpackedTuple[0], currKey})).getResults()[0];
               {
                  auto secondIfOp = rewriter.create<mlir::scf::IfOp>(loc, TypeRange(), keyMatches, true);
                  {
                     OpBuilder::InsertionGuard insertionGuard(rewriter);
                     scf::IfOp::ensureTerminator(secondIfOp.thenRegion(), rewriter, loc);
                     rewriter.setInsertionPointToStart(&secondIfOp.thenRegion().front());
                     auto newAggr = rewriter.create<mlir::CallIndirectOp>(loc, builderUnpacked[2], ValueRange({unpackedTuple[1], currVal})).results()[0];
                     mlir::Value rePacked = rewriter.create<mlir::util::PackOp>(loc, TupleType::get(getContext(), {keyType, aggrType}), ValueRange{unpackedTuple[0], newAggr});
                     rewriter.create<util::StoreOp>(loc, rePacked, genericMemref, Value());
                  }
                  {
                     OpBuilder::InsertionGuard insertionGuard(rewriter);
                     scf::IfOp::ensureTerminator(secondIfOp.elseRegion(), rewriter, loc);
                     rewriter.setInsertionPointToStart(&secondIfOp.elseRegion().front());
                     Value serializedKey = currKey;//serializeForAggrHT(rewriter, typeConverter, builderPtr, currKey, aggrHTBuilderType.getKeyType(), functionRegistry);
                     Value serializedVal = currVal;//serializeForAggrHT(rewriter, typeConverter, builderPtr, currVal, aggrHTBuilderType.getValType(), functionRegistry);
                     rewriter.create<util::StoreOp>(loc, serializedKey, allocaKey, Value());
                     rewriter.create<util::StoreOp>(loc, serializedVal, allocaVal, Value());
                     functionRegistry.call(rewriter, FunctionId::AggrHtBuilderMerge, ValueRange({builderPtr, hashed, plainMemrefKey, plainMemrefVal}));
                  }
               }
            }
            {
               OpBuilder::InsertionGuard insertionGuard(rewriter);
               scf::IfOp::ensureTerminator(newIfOp.elseRegion(), rewriter, loc);
               rewriter.setInsertionPointToStart(&newIfOp.elseRegion().front());
               Value serializedKey = currKey;// serializeForAggrHT(rewriter, typeConverter, builderPtr, currKey, aggrHTBuilderType.getKeyType(), functionRegistry);
               Value serializedVal = currVal;// serializeForAggrHT(rewriter, typeConverter, builderPtr, currVal, aggrHTBuilderType.getValType(), functionRegistry);
               rewriter.create<util::StoreOp>(loc, serializedKey, allocaKey, Value());
               rewriter.create<util::StoreOp>(loc, serializedVal, allocaVal, Value());
               functionRegistry.call(rewriter, FunctionId::AggrHtBuilderMerge, ValueRange({builderPtr, hashed, plainMemrefKey, plainMemrefVal}));
            }
            rewriter.replaceOp(op, mergeOpAdaptor.builder());
         }
      } else if (auto joinHtBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::JoinHTBuilderType>()) {
         Value v = mergeOpAdaptor.val();//serializeForJoinHT(rewriter, typeConverter, mergeOpAdaptor.builder(), mergeOpAdaptor.val(), mergeOp.val().getType(), functionRegistry);
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), v.getType()});
         auto valType = mergeOpAdaptor.val().getType().cast<mlir::TupleType>();

         auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, valType.getTypes(), mergeOpAdaptor.val())->getResults();
         Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), unPacked[0]);
         auto bucket = rewriter.create<mlir::util::PackOp>(rewriter.getUnknownLoc(), tupleType, mlir::ValueRange({hashed, v}));

         Value ptr = functionRegistry.call(rewriter, FunctionId::JoinHtBuilderMerge, ValueRange({mergeOpAdaptor.builder()}))[0];
         Value typedPtr = rewriter.create<util::GenericMemrefCastOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), tupleType, llvm::Optional<int64_t>()), ptr);
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), bucket, typedPtr, Value());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      } else if (auto joinHtBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::MarkableJoinHTBuilderType>()) {
         Value v = mergeOpAdaptor.val();//serializeForMJoinHT(rewriter, typeConverter, mergeOpAdaptor.builder(), mergeOpAdaptor.val(), mergeOp.val().getType(), functionRegistry);
         auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), v.getType()});
         auto valType = mergeOpAdaptor.val().getType().cast<mlir::TupleType>();

         auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, valType.getTypes(), mergeOpAdaptor.val())->getResults();
         Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), unPacked[0]);
         auto bucket = rewriter.create<mlir::util::PackOp>(rewriter.getUnknownLoc(), tupleType, mlir::ValueRange({hashed, v}));

         Value ptr = functionRegistry.call(rewriter, FunctionId::MJoinHtBuilderMerge, ValueRange({mergeOpAdaptor.builder()}))[0];
         Value typedPtr = rewriter.create<util::GenericMemrefCastOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), tupleType, llvm::Optional<int64_t>()), ptr);
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), bucket, typedPtr, Value());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      }

      return success();
   }
};

class CreateAggrHTBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateAggrHTBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateAggrHTBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;

      static size_t id = 0;
      auto createOp = cast<mlir::db::CreateAggrHTBuilder>(op);

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto ptrType = mlir::util::GenericMemrefType::get(getContext(), IntegerType::get(getContext(), 8), llvm::Optional<int64_t>());
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
         FuncOp compareFunc;
         {
            OpBuilder::InsertionGuard insertionGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            compareFunc = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({ptrType, ptrType}), TypeRange(mlir::db::BoolType::get(rewriter.getContext()))));
            auto* funcBody = new Block;
            funcBody->addArguments(TypeRange({ptrType, ptrType}));
            compareFunc.body().push_back(funcBody);
            rewriter.setInsertionPointToStart(funcBody);
            Value left = funcBody->getArgument(0);
            Value right = funcBody->getArgument(1);

            Value genericMemrefLeft = rewriter.create<util::GenericMemrefCastOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), keyType, llvm::Optional<int64_t>()), left);
            Value genericMemrefRight = rewriter.create<util::GenericMemrefCastOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), keyType, llvm::Optional<int64_t>()), right);
            Value tupleLeft = rewriter.create<util::LoadOp>(createOp.getLoc(), keyType, genericMemrefLeft, Value());
            Value tupleRight = rewriter.create<util::LoadOp>(createOp.getLoc(), keyType, genericMemrefRight, Value());
            Value res = rewriter.create<mlir::CallOp>(createOp->getLoc(), rawCompareFunc, ValueRange({tupleLeft, tupleRight})).getResult(0);
            rewriter.create<mlir::ReturnOp>(createOp->getLoc(), res);
         }
         FuncOp funcOp;
         {
            OpBuilder::InsertionGuard insertionGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            funcOp = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_update" + std::to_string(id++), rewriter.getFunctionType(TypeRange({ptrType, ptrType}), TypeRange()));
            auto* funcBody = new Block;
            funcBody->addArguments(TypeRange({ptrType, ptrType}));
            funcOp.body().push_back(funcBody);
            rewriter.setInsertionPointToStart(funcBody);
            Value left = funcBody->getArgument(0);
            Value right = funcBody->getArgument(1);

            Value genericMemrefLeft = rewriter.create<util::GenericMemrefCastOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), aggrType, llvm::Optional<int64_t>()), left);
            Value genericMemrefRight = rewriter.create<util::GenericMemrefCastOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), valType, llvm::Optional<int64_t>()), right);
            Value tupleLeft = rewriter.create<util::LoadOp>(createOp.getLoc(), aggrType, genericMemrefLeft, Value());
            Value tupleRight = rewriter.create<util::LoadOp>(createOp.getLoc(), valType, genericMemrefRight, Value());
            Value res = rewriter.create<mlir::CallOp>(createOp->getLoc(), rawUpdateFunc, ValueRange({tupleLeft, tupleRight})).getResult(0);
            Value castedVal = rewriter.create<mlir::db::TypeCastOp>(rewriter.getUnknownLoc(), typeConverter->convertType(aggrType), res);
            rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), castedVal, genericMemrefLeft, Value());
            rewriter.create<mlir::ReturnOp>(rewriter.getUnknownLoc());
         }

         Value updateFunctionPointer = rewriter.create<mlir::ConstantOp>(createOp->getLoc(), funcOp.type(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.sym_name())));
         Value compareFunctionPointer = rewriter.create<mlir::ConstantOp>(createOp->getLoc(), compareFunc.type(), SymbolRefAttr::get(rewriter.getStringAttr(compareFunc.sym_name())));
         Value keySize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), keyType);
         Value valSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), valType);
         Value aggrSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), aggrType);
         Value combinedSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), TupleType::get(getContext(), {keyType, aggrType}));

         Value allocaInitial;
         {
            OpBuilder::InsertionGuard insertionGuard(rewriter);
            auto func = op->getParentOfType<mlir::FuncOp>();
            rewriter.setInsertionPointToStart(&func.getBody().front());
            allocaInitial = rewriter.create<mlir::util::AllocaOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), aggrType, llvm::Optional<int64_t>()), Value());
         }
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), createOp.initial(), allocaInitial, Value());
         Value plainMemref = rewriter.create<mlir::util::GenericMemrefCastOp>(createOp->getLoc(), ptrType, allocaInitial);

         Value builder = functionRegistry.call(rewriter, FunctionId::AggrHtBuilderCreate, {keySize, valSize, aggrSize, combinedSize, compareFunctionPointer, updateFunctionPointer, plainMemref})[0];
         mlir::Value compareFuncRef = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rawCompareFunc.getType(), SymbolRefAttr::get(rewriter.getStringAttr(rawCompareFunc.sym_name())));
         mlir::Value updateFuncRef = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rawUpdateFunc.getType(), SymbolRefAttr::get(rewriter.getStringAttr(rawUpdateFunc.sym_name())));
         rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, TupleType::get(rewriter.getContext(), {ptrType, rawCompareFunc.getType(), rawUpdateFunc.getType()}), ValueRange{builder, compareFuncRef, updateFuncRef});

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
            auto ptrType = mlir::util::GenericMemrefType::get(getContext(), IntegerType::get(getContext(), 8), llvm::Optional<int64_t>());

            auto updateFnType = FunctionType::get(getContext(), {aggrHTBuilderType.getAggrType(), aggrHTBuilderType.getValType()}, {aggrHTBuilderType.getAggrType()});
            auto compareFnType = FunctionType::get(getContext(), {aggrHTBuilderType.getKeyType(), aggrHTBuilderType.getKeyType()}, {mlir::db::BoolType::get(getContext())});
            auto builderUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), TypeRange{ptrType, typeConverter->convertType(compareFnType), typeConverter->convertType(updateFnType)}, buildAdaptor.builder())->getResults();
            auto builderPtr = builderUnpacked[0];
            Value aggrHashtable = functionRegistry.call(rewriter, FunctionId::AggrHtBuilderBuild, builderPtr)[0];
            rewriter.replaceOp(op, aggrHashtable);
         }
      } else if (auto joinHtBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::JoinHTBuilderType>()) {
         Value vector = functionRegistry.call(rewriter, FunctionId::JoinHtBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, vector);
      } else if (auto joinHtBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::MarkableJoinHTBuilderType>()) {
         Value vector = functionRegistry.call(rewriter, FunctionId::MJoinHtBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, vector);
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
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateJoinHtBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateJoinHTBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto createOp = mlir::cast<mlir::db::CreateJoinHTBuilder>(op);
      auto builderType = createOp.builder().getType().cast<mlir::db::JoinHTBuilderType>();
      auto entryType = mlir::TupleType::get(rewriter.getContext(), {builderType.getKeyType(), builderType.getValType()});

      auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
      Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), tupleType);
      Value vectorBuilder = functionRegistry.call(rewriter, FunctionId::JoinHtBuilderCreate, {elementSize})[0];
      rewriter.replaceOp(op, vectorBuilder);
      return success();
   }
};
class CreateMarkableJoinHtBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateMarkableJoinHtBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateMarkableJoinHTBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto createOp = mlir::cast<mlir::db::CreateMarkableJoinHTBuilder>(op);
      auto builderType = createOp.builder().getType().cast<mlir::db::MarkableJoinHTBuilderType>();
      auto entryType = mlir::TupleType::get(rewriter.getContext(), {builderType.getKeyType(), builderType.getValType()});

      auto tupleType = mlir::TupleType::get(rewriter.getContext(), {rewriter.getIndexType(), entryType});
      Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), tupleType);
      Value vectorBuilder = functionRegistry.call(rewriter, FunctionId::MJoinHtBuilderCreate, {elementSize})[0];
      rewriter.replaceOp(op, vectorBuilder);
      return success();
   }
};
} // namespace
void mlir::db::populateBuilderToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateTableBuilderLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateVectorBuilderLowering>(typeConverter, patterns.getContext());
   patterns.insert<CreateAggrHTBuilderLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<BuilderMergeLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<BuilderBuildLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateJoinHtBuilderLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<CreateMarkableJoinHtBuilderLowering>(functionRegistry, typeConverter, patterns.getContext());
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
      return mlir::TupleType::get(patterns.getContext(),{IndexType::get(patterns.getContext()),mlir::util::GenericMemrefType::get(patterns.getContext(), typeConverter.convertType(vectorBuilderType.getElementType()), -1)});
   });
   typeConverter.addConversion([&](mlir::db::AggrHTBuilderType aggrHtBuilderType) {
      auto updateFnType = FunctionType::get(patterns.getContext(), {aggrHtBuilderType.getAggrType(), aggrHtBuilderType.getValType()}, {aggrHtBuilderType.getAggrType()});
      auto compareFnType = FunctionType::get(patterns.getContext(), {aggrHtBuilderType.getKeyType(), aggrHtBuilderType.getKeyType()}, {mlir::db::BoolType::get(patterns.getContext())});

      if (aggrHtBuilderType.getKeyType().getTypes().empty()) {
         return (Type) TupleType::get(patterns.getContext(), {typeConverter.convertType(aggrHtBuilderType.getAggrType()), typeConverter.convertType(updateFnType)});
      } else {
         auto ptrType = mlir::util::GenericMemrefType::get(patterns.getContext(), IntegerType::get(patterns.getContext(), 8), llvm::Optional<int64_t>());
         return (Type) TupleType::get(patterns.getContext(), {ptrType, typeConverter.convertType(compareFnType), typeConverter.convertType(updateFnType)});
      }
   });
   typeConverter.addConversion([&](mlir::db::JoinHTBuilderType joinHtBuilderType) {
      return mlir::util::GenericMemrefType::get(patterns.getContext(), IntegerType::get(patterns.getContext(), 8), llvm::Optional<int64_t>());
   });
   typeConverter.addConversion([&](mlir::db::MarkableJoinHTBuilderType joinHtBuilderType) {
      return mlir::util::GenericMemrefType::get(patterns.getContext(), IntegerType::get(patterns.getContext(), 8), llvm::Optional<int64_t>());
   });
}
