#include "mlir/Dialect/DB/IR/DBCollectionType.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"

#include "mlir/Conversion/DBToArrowStd/BitUtil.h"
#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
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
      return functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorValid, iterator)[0];
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorFree, iterator);
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
   Type serializedElementType;
   Value values;
   Value rawValues;
   Type ptrType;
   Type typedPtrType;
   Type serializedType(OpBuilder& builder, Type type) {
      if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
         std::vector<Type> types;
         for (size_t i = 0; i < originalTupleType.size(); i++) {
            Type t = serializedType(builder, originalTupleType.getType(i));
            types.push_back(t);
         }
         return TupleType::get(builder.getContext(), types);
      } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
         if (stringType.isNullable()) {
            return TupleType::get(builder.getContext(), TypeRange({builder.getI1Type(), builder.getI64Type(), builder.getI64Type()}));
         } else {
            return TupleType::get(builder.getContext(), TypeRange({builder.getI64Type(), builder.getI64Type()}));
         }
      } else {
         return typeConverter->convertType(type);
      }
   }
   Value deserialize(OpBuilder& builder, Value rawValues, Value element, Type type) {
      if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
         auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
         std::vector<Value> serializedValues;
         std::vector<Type> types;
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
         for (size_t i = 0; i < tupleType.size(); i++) {
            Value currVal = unPackOp.getResult(i);
            Value serialized = deserialize(builder, rawValues, currVal, originalTupleType.getType(i));
            serializedValues.push_back(serialized);
            types.push_back(serialized.getType());
         }
         return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
      } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
         Value pos, len, isNull;
         if (stringType.isNullable()) {
            auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), builder.getIndexType(), builder.getIndexType()}), element);
            isNull = unPackOp.getResult(0);
            pos = unPackOp.getResult(1);
            len = unPackOp.getResult(2);
         } else {
            auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getIndexType(), builder.getIndexType()}), element);
            pos = unPackOp.getResult(0);
            len = unPackOp.getResult(1);
         }
         Value val = builder.create<mlir::memref::ViewOp>(builder.getUnknownLoc(), MemRefType::get({-1}, builder.getIntegerType(8)), rawValues, pos, mlir::ValueRange({len}));
         if (isNull) {
            val = builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), {builder.getI1Type(), val.getType()}), ValueRange({isNull, val}));
         }
         return val;
      } else {
         return element;
      }
   }

   public:
   VectorIterator(Value vector, Type elementType) : vector(vector), elementType(elementType){
   }
   virtual void init(OpBuilder& builder) {
      serializedElementType = serializedType(builder, elementType);
      Type ptrType = MemRefType::get({-1}, builder.getIntegerType(8));
      Type typedPtrType = util::GenericMemrefType::get(builder.getContext(), serializedElementType, -1);
      auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), TypeRange({ptrType, ptrType}), vector);
      values = unpacked.getResult(0);
      values = builder.create<util::ToGenericMemrefOp>(builder.getUnknownLoc(), typedPtrType, values);

      rawValues = unpacked.getResult(1);
   }
   virtual Value upper(OpBuilder& builder) {
      return builder.create<util::DimOp>(builder.getUnknownLoc(), builder.getIndexType(), values);
   }
   virtual Value getElement(OpBuilder& builder, Value index) {
      Value loaded = builder.create<util::LoadOp>(builder.getUnknownLoc(), serializedElementType, values, index);
      Value deserialized = deserialize(builder, rawValues, loaded, elementType);
      return deserialized;
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
   virtual std::vector<Value> implementLoop(mlir::ValueRange iterArgs, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
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
      builder.create<mlir::db::ConditionOp>(builder.getUnknownLoc(), iterator->iteratorValid(builder, arg1), whileOp.before().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.after().front());
      auto arg2 = whileOp.after().front().getArgument(0);
      Value currElement = iterator->iteratorGetCurrentElement(builder, arg2);
      Value nextIterator = iterator->iteratorNext(builder, arg2);
      auto terminator = builder.create<mlir::db::YieldOp>(builder.getUnknownLoc());
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      std::vector<Value> bodyParams = {currElement};
      auto additionalArgs = whileOp.after().front().getArguments().drop_front();
      bodyParams.insert(bodyParams.end(), additionalArgs.begin(), additionalArgs.end());
      std::vector<Value> returnValues = bodyBuilder(bodyParams, builder);
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
   virtual std::vector<Value> implementLoop(mlir::ValueRange iterArgs, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
      auto insertionPoint = builder.saveInsertionPoint();
      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      auto forOp = builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), iterator->lower(builder), iterator->upper(builder), iterator->step(builder), iterArgs.size() ? iterArgs : llvm::None);
      if (iterArgs.size()) {
         builder.setInsertionPointToStart(forOp.getBody());
         builder.create<scf::YieldOp>(builder.getUnknownLoc());
      }
      Operation* terminator = forOp.getBody()->getTerminator();
      builder.setInsertionPointToStart(forOp.getBody());
      auto element = iterator->getElement(builder, forOp.getInductionVar());
      std::vector<Value> bodyArguments = {element};

      bodyArguments.insert(bodyArguments.end(), forOp.getRegionIterArgs().begin(), forOp.getRegionIterArgs().end());
      std::vector<Value> results = bodyBuilder(bodyArguments, builder);
      iterator->destroyElement(builder, element);
      if (iterArgs.size()) {
         builder.create<scf::YieldOp>(builder.getUnknownLoc(), results);
         builder.eraseOp(terminator);
      }
      builder.restoreInsertionPoint(insertionPoint);
      iterator->down(builder);
      return std::vector<Value>(forOp.results().begin(), forOp.results().end());
   }
};
std::unique_ptr<mlir::db::CollectionIterationImpl> mlir::db::CollectionIterationImpl::getImpl(Type collectionType, Value collection, mlir::db::codegen::FunctionRegistry& functionRegistry) {
   if (auto generic = collectionType.dyn_cast_or_null<mlir::db::GenericIterableType>()) {
      if (generic.getIteratorName() == "table_chunk_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<TableIterator>(collection, functionRegistry));
      } else if (generic.getIteratorName() == "table_row_iterator") {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<TableRowIterator>(collection, generic.getElementType(), functionRegistry));
      }
   } else if (auto range = collectionType.dyn_cast_or_null<mlir::db::RangeType>()) {
      return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<RangeIterator>(range.getElementType().cast<DBType>(), collection));
   } else if (auto vector = collectionType.dyn_cast_or_null<mlir::db::VectorType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<VectorIterator>(collection, vector.getElementType()));
   }
   return std::unique_ptr<mlir::db::CollectionIterationImpl>();
}