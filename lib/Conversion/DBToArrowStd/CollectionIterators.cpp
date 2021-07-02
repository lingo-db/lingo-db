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
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);
      return ptrType;
   }

   virtual Value iterator(OpBuilder& builder) override {
      Value tablePtr = builder.create<util::GetTupleOp>(builder.getUnknownLoc(), MemRefType::get({}, IntegerType::get(builder.getContext(), 8)), tableInfo, 0);
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
   Type elementType;
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   JoinHtIterator(Value tableInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : iteratorInfo(tableInfo), elementType(elementType), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) override {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);

      return ptrType;
   }

   virtual Value iterator(OpBuilder& builder) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);

      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getIndexType(), ptrType}), iteratorInfo);
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtIteratorInit, mlir::ValueRange({unpacked.getResult(1), unpacked.getResult(0)}))[0];
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtIteratorNext, iterator)[0];
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      Type typedPtrType = util::GenericMemrefType::get(builder.getContext(), elementType, llvm::Optional<int64_t>());
      Value currElementPtr = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtIteratorCurr, iterator)[0];
      mlir::Value data = builder.create<util::ToGenericMemrefOp>(builder.getUnknownLoc(), typedPtrType, currElementPtr);

      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), elementType, data, Value());
      return loaded;
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value rawValue = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtIteratorValid, iterator)[0];
      Value dbValue = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::JoinHtIteratorFree, iterator);
   }
};
class MarkableJoinHtIterator : public WhileIterator {
   Value iteratorInfo;
   Type tupleType;
   Type markerType;
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   MarkableJoinHtIterator(Value tableInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : iteratorInfo(tableInfo), functionRegistry(functionRegistry) {
      tupleType=elementType.cast<mlir::TupleType>().getType(0);
      markerType=elementType.cast<mlir::TupleType>().getType(0);

   }
   virtual void init(OpBuilder& builder) override {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);

      return ptrType;
   }

   virtual Value iterator(OpBuilder& builder) override {
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtIteratorInit, mlir::ValueRange({iteratorInfo}))[0];
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtIteratorNext, iterator)[0];
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      auto ptrType = MemRefType::get({}, i8Type);

      Type typedPtrType = util::GenericMemrefType::get(builder.getContext(), tupleType, llvm::Optional<int64_t>());
      Value pair = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtIteratorCurr, iterator)[0];
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({ptrType, ptrType}), pair);
      mlir::Value data = builder.create<util::ToGenericMemrefOp>(builder.getUnknownLoc(), typedPtrType, unpacked.getResult(0));

      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), tupleType, data, Value());
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), {tupleType,ptrType}), mlir::ValueRange{loaded,unpacked.getResult(1)});
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value rawValue = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtIteratorValid, iterator)[0];
      Value dbValue = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtIteratorFree, iterator);
   }
};
class MarkableJoinHtLookupIterator : public WhileIterator {
   Value iteratorInfo;
   Type tupleType;
   Type markerType;
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   MarkableJoinHtLookupIterator(Value tableInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : iteratorInfo(tableInfo), functionRegistry(functionRegistry) {
      tupleType=elementType.cast<mlir::TupleType>().getType(0);
      markerType=elementType.cast<mlir::TupleType>().getType(0);

   }
   virtual void init(OpBuilder& builder) override {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);

      return ptrType;
   }

   virtual Value iterator(OpBuilder& builder) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);

      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getIndexType(), ptrType}), iteratorInfo);
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtLookupIteratorInit, mlir::ValueRange({unpacked.getResult(1), unpacked.getResult(0)}))[0];
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtLookupIteratorNext, iterator)[0];
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      auto ptrType = MemRefType::get({}, i8Type);

      Type typedPtrType = util::GenericMemrefType::get(builder.getContext(), tupleType, llvm::Optional<int64_t>());
      Value pair = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtLookupIteratorCurr, iterator)[0];
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({ptrType, ptrType}), pair);
      mlir::Value data = builder.create<util::ToGenericMemrefOp>(builder.getUnknownLoc(), typedPtrType, unpacked.getResult(0));

      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), tupleType, data, Value());
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), {tupleType,ptrType}), mlir::ValueRange{loaded,unpacked.getResult(1)});
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value rawValue = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtLookupIteratorValid, iterator)[0];
      Value dbValue = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::MJoinHtLookupIteratorFree, iterator);
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

      Value const0 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 0));
      Value const1 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 1));
      Value const2 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 2));

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
            Value byteOffset = builder.create<mlir::MulIOp>(builder.getUnknownLoc(), indexType, offset, typeSize);
            Value byteSize = builder.create<memref::DimOp>(builder.getUnknownLoc(), valueBuffer0, 0);
            Value typedSize = builder.create<mlir::UnsignedDivIOp>(builder.getUnknownLoc(), indexType, byteSize, typeSize);
            valueBuffer = builder.create<memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get({-1}, convertedType), valueBuffer0, byteOffset, ValueRange({typedSize}));
         } else {
            valueBuffer = valueBuffer0;
         }
         Value varLenBuffer{};
         Value nullMultiplier;
         if (dbtype.isNullable()) {
            bitmapBuffer = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const0}))[0];
            Value bitmapSize = builder.create<memref::DimOp>(builder.getUnknownLoc(), bitmapBuffer, 0);
            Value emptyBitmap = builder.create<mlir::CmpIOp>(builder.getUnknownLoc(), mlir::CmpIPredicate::eq, const0, bitmapSize);
            nullMultiplier = builder.create<mlir::SelectOp>(builder.getUnknownLoc(), emptyBitmap, const0, const1);
         }
         if (dbtype.isVarLen()) {
            varLenBuffer = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const2}))[0];
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
      Value const1 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 1));
      std::vector<Type> types;
      std::vector<Value> values;
      for (auto column : columnInfo) {
         types.push_back(column.type);
         Value val;
         if (column.type.isa<db::StringType>()) {
            Value pos1 = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({index}));
            Value ip1 = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), indexType, index, const1);
            Value pos2 = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({ip1}));
            Value len = builder.create<mlir::SubIOp>(builder.getUnknownLoc(), builder.getI32Type(), pos2, pos1);
            Value pos1AsIndex = builder.create<IndexCastOp>(builder.getUnknownLoc(), pos1, indexType);
            Value lenAsIndex = builder.create<IndexCastOp>(builder.getUnknownLoc(), len, indexType);
            val = builder.create<mlir::memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get({-1}, builder.getIntegerType(8)), column.varLenBuffer, pos1AsIndex, mlir::ValueRange({lenAsIndex}));
         } else if (column.type.isa<db::BoolType>()) {
            Value realPos = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), indexType, column.offset, index);
            val = mlir::db::codegen::BitUtil::getBit(builder, column.values, realPos);
         } else if (auto decimalType = column.type.dyn_cast_or_null<db::DecimalType>()) {
            val = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({index}));
            if (typeConverter->convertType(decimalType.getBaseType()).cast<mlir::IntegerType>().getWidth() != 128) {
               auto converted = builder.create<mlir::TruncateIOp>(builder.getUnknownLoc(), typeConverter->convertType(decimalType.getBaseType()), val);
               val = converted;
            }
         } else if (column.stdType.isa<mlir::IntegerType>() || column.stdType.isa<mlir::FloatType>()) {
            val = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({index}));
         } else {
            assert(val && "unhandled type!!");
         }
         if (column.type.isNullable()) {
            Value realPos = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), indexType, column.offset, index);
            realPos = builder.create<mlir::MulIOp>(builder.getUnknownLoc(), indexType, column.nullMultiplier, realPos);
            Value isnull = mlir::db::codegen::BitUtil::getBit(builder, column.nullBitmap, realPos, true);
            val = builder.create<mlir::db::CombineNullOp>(builder.getUnknownLoc(), column.type, val, isnull);
         }
         values.push_back(val);
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), ValueRange(values));
   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
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
   Value rawValues;
   Type ptrType;
   Type typedPtrType;
   mlir::db::codegen::FunctionRegistry& functionRegistry;
   using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;

   public:
   VectorIterator(mlir::db::codegen::FunctionRegistry& functionRegistry, Value vector, Type elementType) : vector(vector), elementType(elementType), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) {
      Type typedPtrType = util::GenericMemrefType::get(builder.getContext(), elementType, -1);
      values = functionRegistry.call(builder, FunctionId::VectorGetValues, {vector})[0];
      values = builder.create<util::ToGenericMemrefOp>(builder.getUnknownLoc(), typedPtrType, values);
   }
   virtual Value upper(OpBuilder& builder) {
      return builder.create<util::DimOp>(builder.getUnknownLoc(), builder.getIndexType(), values);
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), elementType, values, index);
      return loaded;
   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) {
   }
   virtual void down(OpBuilder& builder) {
   }
};
class TopKIterator : public ForIterator {
   Value topK;
   Type elementType;
   Type ptrType;
   Type typedPtrType;
   mlir::db::codegen::FunctionRegistry& functionRegistry;
   using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;

   public:
   TopKIterator(mlir::db::codegen::FunctionRegistry& functionRegistry, Value topK, Type elementType) : topK(topK), elementType(elementType), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) {
      typedPtrType = util::GenericMemrefType::get(builder.getContext(), elementType, llvm::Optional<int64_t>());
   }
   virtual Value upper(OpBuilder& builder) {
      return functionRegistry.call(builder, FunctionId::TopKEntries, {topK})[0];
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      Value rawPtr = functionRegistry.call(builder, FunctionId::TopKGetEntry, {topK, index})[0];
      Value ptr = builder.create<util::ToGenericMemrefOp>(builder.getUnknownLoc(), typedPtrType, rawPtr);

      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), elementType, ptr, Value());
      return loaded;
   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) {
   }
   virtual void down(OpBuilder& builder) {
   }
};
class AggrHtIterator : public WhileIterator {
   Value iteratorInfo;
   Type keyType;
   Type valType;
   Type elementType;

   db::codegen::FunctionRegistry& functionRegistry;

   public:
   AggrHtIterator(Value tableInfo, Type keyType, Type valType, db::codegen::FunctionRegistry& functionRegistry) : iteratorInfo(tableInfo), keyType(keyType), valType(valType), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) override {
      elementType = TupleType::get(builder.getContext(), {keyType, valType});
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);

      return ptrType;
   }

   virtual Value iterator(OpBuilder& builder) override {
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::AggrHtIteratorInit, mlir::ValueRange({iteratorInfo}))[0];
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::AggrHtIteratorNext, iterator)[0];
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      Type typedPtrType = util::GenericMemrefType::get(builder.getContext(), elementType, llvm::Optional<int64_t>());
      Value currElementPtr = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::AggrHtIteratorCurr, iterator)[0];
      mlir::Value data = builder.create<util::ToGenericMemrefOp>(builder.getUnknownLoc(), typedPtrType, currElementPtr);

      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), elementType, data, Value());
      return loaded;
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value rawValue = functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::AggrHtIteratorValid, iterator)[0];
      Value dbValue = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::AggrHtIteratorFree, iterator);
   }
};

class ValueOnlyAggrHTIterator : public ForIterator {
   Value ht;
   Type valType;

   public:
   ValueOnlyAggrHTIterator(db::codegen::FunctionRegistry& functionRegistry, Value ht, Type valType) : ht(ht), valType(valType) {
   }
   virtual void init(OpBuilder& builder) {
   }
   virtual Value upper(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      Value undefTuple = builder.create<mlir::util::UndefTupleOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext()));
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), {undefTuple.getType(), typeConverter->convertType(valType)}), ValueRange({undefTuple, ht}));
   }
   virtual Value lower(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual void destroyElement(OpBuilder& builder, Value elem) {
   }
   virtual void down(OpBuilder& builder) {
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
      Value condition = builder.create<mlir::CmpIOp>(builder.getUnknownLoc(), mlir::CmpIPredicate::ult, arg1, upper);
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
      Value nextIterator = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), builder.getIndexType(), arg2, step);
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
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<MarkableJoinHtLookupIterator>(collection, generic.getElementType(), functionRegistry));
      }
   } else if (auto range = collectionType.dyn_cast_or_null<mlir::db::RangeType>()) {
      return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<RangeIterator>(range.getElementType().cast<DBType>(), collection));
   } else if (auto vector = collectionType.dyn_cast_or_null<mlir::db::VectorType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<VectorIterator>(functionRegistry, collection, vector.getElementType()));
   } else if (auto aggrHt = collectionType.dyn_cast_or_null<mlir::db::AggregationHashtableType>()) {
      if (aggrHt.getKeyType().getTypes().empty()) {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<ValueOnlyAggrHTIterator>(functionRegistry, collection, aggrHt.getValType()));
      } else {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<AggrHtIterator>(collection, aggrHt.getKeyType(), aggrHt.getValType(), functionRegistry));
      }
   } else if (auto joinHt = collectionType.dyn_cast_or_null<mlir::db::MarkableJoinHashtableType>()) {
      return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<MarkableJoinHtIterator>(collection, joinHt.getElementType(), functionRegistry));
   }  else if (auto topK = collectionType.dyn_cast_or_null<mlir::db::TopKType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<TopKIterator>(functionRegistry, collection, topK.getElementType()));
   }
   return std::unique_ptr<mlir::db::CollectionIterationImpl>();
}