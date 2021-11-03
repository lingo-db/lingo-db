#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"

#include "mlir/Conversion/DBToArrowStd/BitUtil.h"
#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
using namespace mlir;

class WhileIterator {
   protected:
   mlir::TypeConverter* typeConverter;

   public:
   void setTypeConverter(TypeConverter* typeConverter) {
      WhileIterator::typeConverter = typeConverter;
   }
   virtual Type iteratorType(OpBuilder& builder) = 0;
   virtual void init(OpBuilder& builder) = 0;
   virtual Value iterator(OpBuilder& builder) = 0;
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) = 0;
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) = 0;
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) = 0;
   virtual void iteratorFree(OpBuilder& builder, Value iterator) = 0;
   virtual ~WhileIterator() {}
};
class ForIterator {
   protected:
   mlir::TypeConverter* typeConverter;

   public:
   virtual void init(OpBuilder& builder) = 0;
   virtual Value lower(OpBuilder& builder) = 0;
   virtual Value upper(OpBuilder& builder) = 0;
   virtual Value step(OpBuilder& builder) = 0;
   virtual Value getElement(OpBuilder& builder, Value index) = 0;
   virtual void destroyElement(OpBuilder& builder, Value elem) = 0;
   virtual void down(OpBuilder& builder) = 0;
   virtual ~ForIterator() {}
   void setTypeConverter(TypeConverter* typeConverter) {
      ForIterator::typeConverter = typeConverter;
   }
};
class TableIterator : public WhileIterator {
   Value tableInfo;
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   TableIterator(Value tableInfo, db::codegen::FunctionRegistry& functionRegistry) : tableInfo(tableInfo), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) override {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      return mlir::util::RefType::get(builder.getContext(),IntegerType::get(builder.getContext(), 8),llvm::Optional<int64_t>());
   }

   virtual Value iterator(OpBuilder& builder) override {
      Value tablePtr = builder.create<util::GetTupleOp>(builder.getUnknownLoc(), mlir::util::RefType::get(builder.getContext(),IntegerType::get(builder.getContext(), 8),llvm::Optional<int64_t>()), tableInfo, 0);
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorInit, tablePtr)[0];
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorNext, iterator)[0];
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      Value currElementPtr = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorCurr, iterator)[0];
      Value currElement = builder.create<mlir::util::SetTupleOp>(builder.getUnknownLoc(), typeConverter->convertType(tableInfo.getType()), tableInfo, currElementPtr, 0);
      return currElement;
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value rawValue = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorValid, iterator)[0];
      Value dbValue = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorFree, iterator);
   }
};

class JoinHtIterator : public WhileIterator {
   Value iteratorInfo;
   Value initialPos;
   Value vec;
   Type elementType;
   public:
   JoinHtIterator(Value tableInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : iteratorInfo(tableInfo), elementType(elementType) {
   }
   virtual void init(OpBuilder& builder) override {
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), typeConverter->convertType(iteratorInfo.getType()).cast<TupleType>().getTypes(), iteratorInfo);
      initialPos=unpacked.getResult(0);
      vec=unpacked.getResult(1);
   }
   virtual Type iteratorType(OpBuilder& builder) override {

      return builder.getIndexType();
   }

   virtual Value iterator(OpBuilder& builder) override {
      return initialPos;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), vec.getType().cast<mlir::util::RefType>().getElementType(), vec, iterator);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), loaded);

      return unpacked.getResult(0);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), vec.getType().cast<mlir::util::RefType>().getElementType(), vec, iterator);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), loaded);

      return unpacked.getResult(1);
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value maxValue = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0xFFFFFFFFFFFFFFFF);
      Value rawValue = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),arith::CmpIPredicate::ne,iterator,maxValue);
      Value dbValue = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
   }
};


class MarkableJoinHtIterator : public ForIterator {
   Value hashTable;
   Type elementType;
   Value values;
   Value len;
   Value trueMemref;
   Type memrefType;
   Value typeSize;
   using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;

   public:
   MarkableJoinHtIterator(Value hashTable, Type elementType) : hashTable(hashTable), elementType(elementType) {
   }
   virtual void init(OpBuilder& builder) {
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), typeConverter->convertType(hashTable.getType()).cast<TupleType>().getTypes(), hashTable);
      values = unpacked.getResult(0);
      len = unpacked.getResult(1);
      memrefType = MemRefType::get({-1}, i8Type);
      trueMemref=builder.create<util::ToMemrefOp>(builder.getUnknownLoc(), memrefType, values);
      typeSize = builder.create<util::SizeOfOp>(builder.getUnknownLoc(), builder.getIndexType(), mlir::TypeAttr::get(values.getType().cast<mlir::util::RefType>().getElementType()));
   }
   virtual Value upper(OpBuilder& builder) {
      return len;
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), values.getType().cast<mlir::util::RefType>().getElementType(), values, index);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), loaded);

      Value loadedValue=unpacked.getResult(2);
      Value offset=builder.create<arith::MulIOp>(builder.getUnknownLoc(),typeSize,index);
      Value ref=builder.create<memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get(   {},i8Type  ), trueMemref,offset,ValueRange{});

      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), mlir::ValueRange{loadedValue,ref});
      return loaded;
   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) {
   }
   virtual void down(OpBuilder& builder) {
   }
};
class AggrHtIterator : public ForIterator {
   Value hashTable;
   Type keyType;
   Type aggrType;
   Value values;
   Value len;

   public:
   AggrHtIterator(Value hashTable, Type keyType,Type aggrType) : hashTable(hashTable), keyType(keyType),aggrType(aggrType) {
   }
   virtual void init(OpBuilder& builder) {
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), typeConverter->convertType(hashTable.getType()).cast<TupleType>().getTypes(), hashTable);
      values = unpacked.getResult(1);
      len = unpacked.getResult(0);
   }
   virtual Value upper(OpBuilder& builder) {
      return len;
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), values.getType().cast<mlir::util::RefType>().getElementType(), values, index);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), loaded);
      return unpacked.getResult(2);

   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) {
   }
   virtual void down(OpBuilder& builder) {
   }
};
class MarkableJoinHtLookupIterator : public WhileIterator {
   Value iteratorInfo;
   Value initialPos;
   Value vec;
   Type tupleType;
   Value trueMemref;
   Value typeSize;
   Type memrefType;

   public:
   MarkableJoinHtLookupIterator(Value tableInfo, Type elementType) : iteratorInfo(tableInfo){
      tupleType=elementType.cast<mlir::TupleType>().getType(0);

   }
   virtual void init(OpBuilder& builder) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), typeConverter->convertType(iteratorInfo.getType()).cast<TupleType>().getTypes(), iteratorInfo);
      initialPos=unpacked.getResult(0);
      vec=unpacked.getResult(1);
      memrefType = MemRefType::get({-1}, i8Type);
      trueMemref=builder.create<util::ToMemrefOp>(builder.getUnknownLoc(), memrefType, unpacked.getResult(1));
      typeSize = builder.create<util::SizeOfOp>(builder.getUnknownLoc(), builder.getIndexType(), mlir::TypeAttr::get(vec.getType().cast<mlir::util::RefType>().getElementType()));

   }
   virtual Type iteratorType(OpBuilder& builder) override {
      return builder.getIndexType();
   }

   virtual Value iterator(OpBuilder& builder) override {
      return initialPos;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), vec.getType().cast<mlir::util::RefType>().getElementType(), vec, iterator);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), loaded);

      return unpacked.getResult(1);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), vec.getType().cast<mlir::util::RefType>().getElementType(), vec, iterator);
      auto unpacked = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), loaded);

      Value loadedValue=unpacked.getResult(2);
      //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
      Value offset=builder.create<arith::MulIOp>(builder.getUnknownLoc(),typeSize,iterator);
      //::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides
      Value ref=builder.create<memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get(   {},i8Type  ), trueMemref,offset,ValueRange{});

      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), mlir::ValueRange{loadedValue,ref});
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value maxValue = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0xFFFFFFFFFFFFFFFF);
      Value rawValue = builder.create<arith::CmpIOp>(builder.getUnknownLoc(),arith::CmpIPredicate::ne,iterator,maxValue);
      Value dbValue = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
   }
};

class RangeIterator : public WhileIterator {
   Value rangeTuple;
   Value upper;
   Value step;
   mlir::db::DBType elementType;

   public:
   RangeIterator(mlir::db::DBType elementType, Value rangeTuple) : rangeTuple(rangeTuple), elementType(elementType) {
   }
   virtual void init(OpBuilder& builder) override {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      return elementType;
   }
   virtual Value iterator(OpBuilder& builder) override {
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({elementType, elementType, elementType}), rangeTuple);
      Value lower = unpacked.getResult(0);
      upper = unpacked.getResult(1);
      step = unpacked.getResult(2);
      return lower;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      return builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), elementType, iterator, step);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      return iterator;
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      return builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::DBCmpPredicate::lt, iterator, upper);
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
   }
};
class TableRowIterator : public ForIterator {
   Value tableChunkInfo;
   Value chunk;
   Type elementType;
   db::codegen::FunctionRegistry& functionRegistry;
   Type getValueBufferType(mlir::TypeConverter& typeConverter, OpBuilder& builder, mlir::db::DBType type) {
      if (type.isa<mlir::db::StringType>()) {
         return builder.getI32Type();
      }
      if (type.isa<mlir::db::DecimalType>()) {
         return builder.getIntegerType(128);
      }
      return typeConverter.convertType(type.getBaseType());
   }
   struct Column {
      mlir::db::DBType type;
      Type stdType;
      Value offset;
      Value nullMultiplier;
      Value nullBitmap;
      Value values;
      Value varLenBuffer;
   };
   std::vector<Column> columnInfo;

   public:
   TableRowIterator(Value tableChunkInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : tableChunkInfo(tableChunkInfo), elementType(elementType), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) {
      auto indexType = IndexType::get(builder.getContext());
      auto tableChunkInfoType = typeConverter->convertType(tableChunkInfo.getType()).cast<TupleType>();
      auto columnTypes = elementType.dyn_cast_or_null<TupleType>().getTypes();
      auto unpackOp = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tableChunkInfoType.getTypes(), tableChunkInfo);
      chunk = unpackOp.getResult(0);

      Value const0 = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 0));
      Value const1 = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 1));
      Value const2 = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 2));

      size_t columnIdx = 0;
      for (auto columnType : columnTypes) {
         auto dbtype = columnType.dyn_cast_or_null<mlir::db::DBType>();
         Value columnId = unpackOp.getResult(1 + columnIdx);
         Value offset = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnOffset, mlir::ValueRange({chunk, columnId}))[0];
         Value bitmapBuffer{};
         auto convertedType = getValueBufferType(*typeConverter, builder, dbtype);
         auto typeSize = builder.create<util::SizeOfOp>(builder.getUnknownLoc(), indexType, mlir::TypeAttr::get(convertedType));
         Value valueBuffer0 = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const1}))[0];
         Value valueBuffer;
         if (!dbtype.isa<mlir::db::BoolType>()) {
            Value byteOffset = builder.create<arith::MulIOp>(builder.getUnknownLoc(), indexType, offset, typeSize);
            Value byteSize = builder.create<util::DimOp>(builder.getUnknownLoc(),indexType, valueBuffer0);
            Value asMemRef=builder.create<util::ToMemrefOp>(builder.getUnknownLoc(),MemRefType::get({-1},builder.getIntegerType(8)),valueBuffer0);
            Value view = builder.create<memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get({-1}, builder.getIntegerType(8)), asMemRef, byteOffset, ValueRange({byteSize}));
            valueBuffer =builder.create<mlir::util::ToGenericMemrefOp>(builder.getUnknownLoc(),mlir::util::RefType::get(builder.getContext(),convertedType,-1),view);

         } else {
            valueBuffer = valueBuffer0;
         }
         Value varLenBuffer{};
         Value nullMultiplier;
         if (dbtype.isNullable()) {
            bitmapBuffer = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const0}))[0];
            Value bitmapSize = builder.create<util::DimOp>(builder.getUnknownLoc(), indexType,bitmapBuffer);
            Value emptyBitmap = builder.create<arith::CmpIOp>(builder.getUnknownLoc(), arith::CmpIPredicate::eq, const0, bitmapSize);
            nullMultiplier = builder.create<mlir::SelectOp>(builder.getUnknownLoc(), emptyBitmap, const0, const1);
         }
         if (dbtype.isVarLen()) {
            varLenBuffer = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const2}))[0];
            Value asMemRef=builder.create<util::ToMemrefOp>(builder.getUnknownLoc(),MemRefType::get({-1},builder.getIntegerType(8)),varLenBuffer);
            varLenBuffer=asMemRef;
         }
         columnInfo.push_back({dbtype, convertedType, offset, nullMultiplier, bitmapBuffer, valueBuffer, varLenBuffer});
         columnIdx++;
      }
   }
   virtual Value upper(OpBuilder& builder) {
      return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkNumRows, mlir::ValueRange({chunk}))[0];
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      auto indexType = IndexType::get(builder.getContext());
      Value const1 = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 1));
      std::vector<Type> types;
      std::vector<Value> values;
      for (auto column : columnInfo) {
         types.push_back(column.type);
         Value val;
         if (column.type.isa<db::StringType>()) {
            Value pos1 = builder.create<util::LoadOp>(builder.getUnknownLoc(),column.stdType, column.values, index);
            Value ip1 = builder.create<arith::AddIOp>(builder.getUnknownLoc(), indexType, index, const1);
            Value pos2 = builder.create<util::LoadOp>(builder.getUnknownLoc(),column.stdType, column.values, ip1);
            Value len = builder.create<arith::SubIOp>(builder.getUnknownLoc(), builder.getI32Type(), pos2, pos1);
            Value pos1AsIndex = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), pos1, indexType);
            Value lenAsIndex = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), len, indexType);
            Value view = builder.create<mlir::memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get({-1}, builder.getIntegerType(8)), column.varLenBuffer, pos1AsIndex, mlir::ValueRange({lenAsIndex}));
            val =builder.create<mlir::util::ToGenericMemrefOp>(builder.getUnknownLoc(),mlir::util::RefType::get(builder.getContext(),IntegerType::get(builder.getContext(), 8),-1),view);
         } else if (column.type.isa<db::BoolType>()) {
            Value realPos = builder.create<arith::AddIOp>(builder.getUnknownLoc(), indexType, column.offset, index);
            val = mlir::db::codegen::BitUtil::getBit(builder, column.values, realPos);
         } else if (auto decimalType = column.type.dyn_cast_or_null<db::DecimalType>()) {
            val = builder.create<util::LoadOp>(builder.getUnknownLoc(),column.stdType, column.values, index);
            if (typeConverter->convertType(decimalType.getBaseType()).cast<mlir::IntegerType>().getWidth() != 128) {
               auto converted = builder.create<arith::TruncIOp>(builder.getUnknownLoc(), typeConverter->convertType(decimalType.getBaseType()), val);
               val = converted;
            }
         } else if (column.stdType.isa<mlir::IntegerType>() || column.stdType.isa<mlir::FloatType>()) {
            val = builder.create<util::LoadOp>(builder.getUnknownLoc(),column.stdType, column.values, index);
         } else {
            assert(val && "unhandled type!!");
         }
         if (column.type.isNullable()) {
            Value realPos = builder.create<arith::AddIOp>(builder.getUnknownLoc(), indexType, column.offset, index);
            realPos = builder.create<arith::MulIOp>(builder.getUnknownLoc(), indexType, column.nullMultiplier, realPos);
            Value isnull = mlir::db::codegen::BitUtil::getBit(builder, column.nullBitmap, realPos, true);
            val = builder.create<mlir::db::CombineNullOp>(builder.getUnknownLoc(), column.type, val, isnull);
         }
         values.push_back(val);
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), ValueRange(values));
   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) {
   }
   virtual void down(OpBuilder& builder) {
   }
};
class VectorIterator : public ForIterator {
   Value vector;
   Type elementType;
   Value values;
   Value len;
   using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;

   public:
   VectorIterator(mlir::db::codegen::FunctionRegistry& functionRegistry, Value vector, Type elementType) : vector(vector), elementType(elementType) {
   }
   virtual void init(OpBuilder& builder) {
      Type typedPtrType = util::RefType::get(builder.getContext(), elementType, -1);
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getIndexType(),builder.getIndexType(), typedPtrType}), vector);

      values=unpacked.getResult(2);
      len=unpacked.getResult(0);
   }
   virtual Value upper(OpBuilder& builder) {
      return len;
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), elementType, values, index);
      return loaded;
   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) {
   }
   virtual void down(OpBuilder& builder) {
   }
};
class ValueOnlyAggrHTIterator : public ForIterator {
   Value ht;
   Type valType;

   public:
   ValueOnlyAggrHTIterator(db::codegen::FunctionRegistry& functionRegistry, Value ht, Type valType) : ht(ht), valType(valType) {
   }
   virtual void init(OpBuilder& builder) override{
   }
   virtual Value upper(OpBuilder& builder) override{
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual Value getElement(OpBuilder& builder, Value index) override{
      Value undefTuple = builder.create<mlir::util::UndefTupleOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext()));
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), {undefTuple.getType(), typeConverter->convertType(valType)}), ValueRange({undefTuple, ht}));
   }
   virtual Value lower(OpBuilder& builder) override{
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) override {
      return builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) override{
   }
   virtual void down(OpBuilder& builder) override{
   }
};
class WhileIteratorIterationImpl : public mlir::db::CollectionIterationImpl {
   std::unique_ptr<WhileIterator> iterator;

   public:
   WhileIteratorIterationImpl(std::unique_ptr<WhileIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
      auto insertionPoint = builder.saveInsertionPoint();

      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      Type iteratorType = iterator->iteratorType(builder);
      Value initialIterator = iterator->iterator(builder);
      std::vector<Type> results = {iteratorType};
      std::vector<Value> iterValues = {initialIterator};
      for (auto iterArg : iterArgs) {
         results.push_back(iterArg.getType());
         iterValues.push_back(iterArg);
      }
      auto whileOp = builder.create<mlir::db::WhileOp>(builder.getUnknownLoc(), results, iterValues);
      Block* before = new Block;
      Block* after = new Block;
      whileOp.before().push_back(before);
      whileOp.after().push_back(after);
      for (auto t : results) {
         before->addArgument(t);
         after->addArgument(t);
      }

      builder.setInsertionPointToStart(&whileOp.before().front());
      auto arg1 = whileOp.before().front().getArgument(0);
      Value condition = iterator->iteratorValid(builder, arg1);
      if (flag) {
         Value flagValue = builder.create<mlir::db::GetFlag>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), flag);
         Value shouldContinue = builder.create<mlir::db::NotOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), flagValue);
         Value anded = builder.create<mlir::db::AndOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), ValueRange({condition, shouldContinue}));
         condition = anded;
      }
      builder.create<mlir::db::ConditionOp>(builder.getUnknownLoc(), condition, whileOp.before().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.after().front());
      auto arg2 = whileOp.after().front().getArgument(0);
      Value currElement = iterator->iteratorGetCurrentElement(builder, arg2);
      Value nextIterator = iterator->iteratorNext(builder, arg2);
      auto terminator = builder.create<mlir::db::YieldOp>(builder.getUnknownLoc());
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      std::vector<Value> bodyParams = {currElement};
      auto additionalArgs = whileOp.after().front().getArguments().drop_front();
      bodyParams.insert(bodyParams.end(), additionalArgs.begin(), additionalArgs.end());
      auto returnValues = bodyBuilder(bodyParams, builder);
      returnValues.insert(returnValues.begin(), nextIterator);
      builder.setInsertionPoint(terminator);
      builder.create<mlir::db::YieldOp>(builder.getUnknownLoc(), returnValues);
      builder.eraseOp(terminator);
      Value finalIterator = whileOp.getResult(0);
      builder.restoreInsertionPoint(insertionPoint);
      iterator->iteratorFree(builder, finalIterator);
      auto loopResultValues = whileOp.results().drop_front();
      return std::vector<Value>(loopResultValues.begin(), loopResultValues.end());
   }
};
class ForIteratorIterationImpl : public mlir::db::CollectionIterationImpl {
   std::unique_ptr<ForIterator> iterator;

   public:
   ForIteratorIterationImpl(std::unique_ptr<ForIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
      if (flag) {
         return implementLoopCondition(iterArgs, flag, typeConverter, builder, bodyBuilder);
      } else {
         return implementLoopSimple(iterArgs, typeConverter, builder, bodyBuilder);
      }
   }
   std::vector<Value> implementLoopSimple(const ValueRange& iterArgs, TypeConverter& typeConverter, ConversionPatternRewriter& builder, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();
      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      auto forOp = builder.create<scf::ForOp>(builder.getUnknownLoc(), iterator->lower(builder), iterator->upper(builder), iterator->step(builder), iterArgs.size() ? iterArgs : llvm::None);
      if (iterArgs.size()) {
         builder.setInsertionPointToStart(forOp.getBody());
         builder.create<scf::YieldOp>(builder.getUnknownLoc());
      }
      Operation* terminator = forOp.getBody()->getTerminator();
      builder.setInsertionPointToStart(forOp.getBody());
      auto element = iterator->getElement(builder, forOp.getInductionVar());
      std::vector<Value> bodyArguments = {element};

      bodyArguments.insert(bodyArguments.end(), forOp.getRegionIterArgs().begin(), forOp.getRegionIterArgs().end());
      auto results = bodyBuilder(bodyArguments, builder);
      iterator->destroyElement(builder, element);
      if (iterArgs.size()) {
         builder.create<scf::YieldOp>(builder.getUnknownLoc(), results);
         builder.eraseOp(terminator);
      }
      builder.restoreInsertionPoint(insertionPoint);
      iterator->down(builder);
      return std::vector<Value>(forOp.results().begin(), forOp.results().end());
   }
   std::vector<Value> implementLoopCondition(const ValueRange& iterArgs, Value flag, TypeConverter& typeConverter, ConversionPatternRewriter& builder, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();

      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      Type iteratorType = builder.getIndexType();
      Value initialIterator = iterator->lower(builder);
      std::vector<Type> results = {iteratorType};
      std::vector<Value> iterValues = {initialIterator};
      for (auto iterArg : iterArgs) {
         results.push_back(iterArg.getType());
         iterValues.push_back(iterArg);
      }
      Value upper = iterator->upper(builder);
      Value step = iterator->step(builder);
      auto whileOp = builder.create<mlir::db::WhileOp>(builder.getUnknownLoc(), results, iterValues);
      Block* before = new Block;
      Block* after = new Block;

      whileOp.before().push_back(before);
      whileOp.after().push_back(after);
      for (auto t : results) {
         before->addArgument(t);
         after->addArgument(t);
      }

      builder.setInsertionPointToStart(&whileOp.before().front());
      auto arg1 = whileOp.before().front().getArgument(0);
      Value condition = builder.create<arith::CmpIOp>(builder.getUnknownLoc(), arith::CmpIPredicate::ult, arg1, upper);
      Value dbcondition = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), condition);
      if (flag) {
         Value flagValue = builder.create<mlir::db::GetFlag>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), flag);
         Value shouldContinue = builder.create<mlir::db::NotOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), flagValue);
         Value anded = builder.create<mlir::db::AndOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), ValueRange({dbcondition, shouldContinue}));
         dbcondition = anded;
      }
      builder.create<mlir::db::ConditionOp>(builder.getUnknownLoc(), dbcondition, whileOp.before().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.after().front());
      auto arg2 = whileOp.after().front().getArgument(0);
      Value currElement = iterator->getElement(builder, arg2);
      Value nextIterator = builder.create<arith::AddIOp>(builder.getUnknownLoc(), builder.getIndexType(), arg2, step);
      auto terminator = builder.create<mlir::db::YieldOp>(builder.getUnknownLoc());
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      std::vector<Value> bodyParams = {currElement};
      auto additionalArgs = whileOp.after().front().getArguments().drop_front();
      bodyParams.insert(bodyParams.end(), additionalArgs.begin(), additionalArgs.end());
      auto returnValues = bodyBuilder(bodyParams, builder);
      returnValues.insert(returnValues.begin(), nextIterator);
      builder.setInsertionPoint(terminator);
      builder.create<mlir::db::YieldOp>(builder.getUnknownLoc(), returnValues);
      builder.eraseOp(terminator);
      builder.restoreInsertionPoint(insertionPoint);
      auto loopResultValues = whileOp.results().drop_front();
      return std::vector<Value>(loopResultValues.begin(), loopResultValues.end());
   }
};
std::unique_ptr<mlir::db::CollectionIterationImpl> mlir::db::CollectionIterationImpl::getImpl(Type collectionType, Value collection, mlir::db::codegen::FunctionRegistry& functionRegistry) {
   if (auto generic = collectionType.dyn_cast_or_null<mlir::db::GenericIterableType>()) {
      if (generic.getIteratorName() == "table_chunk_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<TableIterator>(collection, functionRegistry));
      } else if (generic.getIteratorName() == "table_row_iterator") {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<TableRowIterator>(collection, generic.getElementType(), functionRegistry));
      } else if (generic.getIteratorName() == "join_ht_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<JoinHtIterator>(collection, generic.getElementType(), functionRegistry));
      } else if (generic.getIteratorName() == "mjoin_ht_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<MarkableJoinHtLookupIterator>(collection, generic.getElementType()));
      }
   } else if (auto range = collectionType.dyn_cast_or_null<mlir::db::RangeType>()) {
      return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<RangeIterator>(range.getElementType().cast<DBType>(), collection));
   } else if (auto vector = collectionType.dyn_cast_or_null<mlir::db::VectorType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<VectorIterator>(functionRegistry, collection, vector.getElementType()));
   } else if (auto aggrHt = collectionType.dyn_cast_or_null<mlir::db::AggregationHashtableType>()) {
      if (aggrHt.getKeyType().getTypes().empty()) {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<ValueOnlyAggrHTIterator>(functionRegistry, collection, aggrHt.getValType()));
      } else {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<AggrHtIterator>(collection, aggrHt.getKeyType(), aggrHt.getValType()));
      }
   } else if (auto joinHt = collectionType.dyn_cast_or_null<mlir::db::MarkableJoinHashtableType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<MarkableJoinHtIterator>(collection, joinHt.getElementType()));
   }
   return std::unique_ptr<mlir::db::CollectionIterationImpl>();
}