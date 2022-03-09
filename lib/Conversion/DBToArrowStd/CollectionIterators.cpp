#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"

#include "mlir/Conversion/DBToArrowStd/BitUtil.h"
#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
using namespace mlir;

class WhileIterator {
   protected:
   mlir::TypeConverter* typeConverter;
   MLIRContext* context;
   mlir::Location loc;
   WhileIterator(mlir::MLIRContext* context) : context(context), loc(mlir::UnknownLoc::get(context)) {}

   public:
   void setTypeConverter(TypeConverter* typeConverter) {
      WhileIterator::typeConverter = typeConverter;
   }
   void setLoc(mlir::Location loc) {
      this->loc = loc;
   }
   virtual Type iteratorType(OpBuilder& builder) = 0;
   virtual void init(OpBuilder& builder){};
   virtual Value iterator(OpBuilder& builder) = 0;
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) = 0;
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) = 0;
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) = 0;
   virtual void iteratorFree(OpBuilder& builder, Value iterator){};
   virtual ~WhileIterator() {}
};
class ForIterator {
   protected:
   mlir::TypeConverter* typeConverter;
   MLIRContext* context;
   mlir::Location loc;
   ForIterator(mlir::MLIRContext* context) : context(context), loc(mlir::UnknownLoc::get(context)) {}

   public:
   void setLoc(mlir::Location loc) {
      this->loc = loc;
   }
   virtual void init(OpBuilder& builder){};
   virtual Value lower(OpBuilder& builder) {
      return builder.create<arith::ConstantIndexOp>(loc, 0);
   }

   virtual Value upper(OpBuilder& builder) = 0;
   virtual Value step(OpBuilder& builder) {
      return builder.create<arith::ConstantIndexOp>(loc, 1);
   }
   virtual Value getElement(OpBuilder& builder, Value index) = 0;
   virtual void destroyElement(OpBuilder& builder, Value elem){};
   virtual void down(OpBuilder& builder){};
   virtual ~ForIterator() {}
   void setTypeConverter(TypeConverter* typeConverter) {
      ForIterator::typeConverter = typeConverter;
   }
};
class TableIterator : public WhileIterator {
   Value tableInfo;
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   TableIterator(Value tableInfo, db::codegen::FunctionRegistry& functionRegistry) : WhileIterator(tableInfo.getContext()), tableInfo(tableInfo), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) override {
      auto convertedType = typeConverter->convertType(tableInfo.getType()).cast<mlir::util::RefType>();
      tableInfo = builder.create<mlir::util::LoadOp>(loc, convertedType.getElementType(), tableInfo);
   }

   virtual Type iteratorType(OpBuilder& builder) override {
      return mlir::util::RefType::get(builder.getContext(), IntegerType::get(builder.getContext(), 8), llvm::Optional<int64_t>());
   }

   virtual Value iterator(OpBuilder& builder) override {
      Value tablePtr = builder.create<util::GetTupleOp>(loc, mlir::util::RefType::get(builder.getContext(), IntegerType::get(builder.getContext(), 8), llvm::Optional<int64_t>()), tableInfo, 0);
      return functionRegistry.call(builder, loc, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorInit, tablePtr)[0];
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      return functionRegistry.call(builder, loc, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorNext, iterator)[0];
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      Value currElementPtr = functionRegistry.call(builder, loc, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorCurr, iterator)[0];
      Value currElement = builder.create<mlir::util::SetTupleOp>(loc, tableInfo.getType(), tableInfo, currElementPtr, 0);
      return currElement;
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      return functionRegistry.call(builder, loc, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorValid, iterator)[0];
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, loc, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorFree, iterator);
   }
};

class JoinHtLookupIterator : public WhileIterator {
   Value iteratorInfo;
   Type ptrType;
   Value hash;
   Value ptr;
   TupleType tupleType;

   public:
   JoinHtLookupIterator(Value tableInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : WhileIterator(tableInfo.getContext()), iteratorInfo(tableInfo) {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      tupleType = typeConverter->convertType(iteratorInfo.getType()).cast<mlir::TupleType>();
      ptrType = tupleType.getType(0);
      return ptrType;
   }

   virtual Value iterator(OpBuilder& builder) override {
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, tupleType.getTypes(), iteratorInfo);
      ptr = unpacked.getResult(0);
      hash = unpacked.getResult(1);
      return ptr;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      auto i8PtrType = mlir::util::RefType::get(builder.getContext(), builder.getI8Type());
      Value nextPtr = builder.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(builder.getContext(), i8PtrType), iterator, 0);
      mlir::Value next = builder.create<mlir::util::LoadOp>(loc, i8PtrType, nextPtr);
      next = builder.create<util::GenericMemrefCastOp>(loc, ptrType, next);
      return builder.create<mlir::util::FilterTaggedPtr>(loc, next.getType(), next, hash);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      auto payloadType = iterator.getType().cast<mlir::util::RefType>().getElementType().cast<TupleType>().getType(1);
      auto payloadPtrType = mlir::util::RefType::get(builder.getContext(), payloadType);
      Value payloadPtr = builder.create<util::TupleElementPtrOp>(loc, payloadPtrType, iterator, 1);
      return builder.create<mlir::util::LoadOp>(loc, payloadType, payloadPtr);
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      return builder.create<mlir::util::IsRefValidOp>(loc, builder.getI1Type(), iterator);
   }
};

class JoinHtIterator : public ForIterator {
   Value hashTable;
   Value values;
   Value len;

   public:
   JoinHtIterator(Value hashTable) : ForIterator(hashTable.getContext()), hashTable(hashTable) {
   }
   virtual void init(OpBuilder& builder) override {
      auto loaded = builder.create<util::LoadOp>(loc, typeConverter->convertType(hashTable.getType()).cast<mlir::util::RefType>().getElementType(), hashTable, Value());
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      values = unpacked.getResult(0);
      len = unpacked.getResult(1);
   }
   virtual Value upper(OpBuilder& builder) override {
      return len;
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value loaded = builder.create<util::LoadOp>(loc, values.getType().cast<mlir::util::RefType>().getElementType(), values, index);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);

      Value loadedValue = unpacked.getResult(1);
      return loadedValue;
   }
};
class AggrHtIterator : public ForIterator {
   Value hashTable;
   Value values;
   Value len;

   public:
   AggrHtIterator(Value hashTable) : ForIterator(hashTable.getContext()), hashTable(hashTable) {
   }
   virtual void init(OpBuilder& builder) override {
      auto elemType = typeConverter->convertType(hashTable.getType()).cast<mlir::util::RefType>().getElementType();
      auto loaded = builder.create<mlir::util::LoadOp>(loc, elemType, hashTable, Value());
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, elemType.cast<mlir::TupleType>().getTypes(), loaded);
      values = unpacked.getResult(2);
      len = unpacked.getResult(0);
   }
   virtual Value upper(OpBuilder& builder) override { return len; }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value loaded = builder.create<util::LoadOp>(loc, values.getType().cast<mlir::util::RefType>().getElementType(), values, index);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      return unpacked.getResult(2);
   }
};
class JoinHtModifyLookupIterator : public WhileIterator {
   Value iteratorInfo;
   Type ptrType;
   Value hash;
   Value ptr;
   TupleType tupleType;

   public:
   JoinHtModifyLookupIterator(Value tableInfo, Type elementType) : WhileIterator(tableInfo.getContext()), iteratorInfo(tableInfo) {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      tupleType = typeConverter->convertType(iteratorInfo.getType()).cast<mlir::TupleType>();
      ptrType = tupleType.getType(0);
      return ptrType;
   }
   virtual Value iterator(OpBuilder& builder) override {
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, tupleType.getTypes(), iteratorInfo);
      ptr = unpacked.getResult(0);
      hash = unpacked.getResult(1);
      return ptr;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      auto i8PtrType = mlir::util::RefType::get(builder.getContext(), builder.getI8Type());
      Value nextPtr = builder.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(builder.getContext(), i8PtrType), iterator, 0);
      mlir::Value next = builder.create<mlir::util::LoadOp>(loc, i8PtrType, nextPtr);
      next = builder.create<util::GenericMemrefCastOp>(loc, ptrType, next);
      return builder.create<mlir::util::FilterTaggedPtr>(loc, ptr.getType(), next, hash);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      auto bucketType = iterator.getType().cast<mlir::util::RefType>().getElementType();
      auto elemType = bucketType.cast<TupleType>().getTypes()[1];

      auto valType = elemType.cast<TupleType>().getTypes()[1];

      Value elemAddress = builder.create<util::TupleElementPtrOp>(loc, util::RefType::get(builder.getContext(), elemType, Optional<int64_t>()), iterator, 1);

      Value loadedValue = builder.create<util::LoadOp>(loc, elemType, elemAddress);

      Value valAddress = builder.create<util::TupleElementPtrOp>(loc, util::RefType::get(builder.getContext(), valType, Optional<int64_t>()), elemAddress, 1);

      return builder.create<mlir::util::PackOp>(loc, mlir::ValueRange{loadedValue, valAddress});
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      return builder.create<mlir::util::IsRefValidOp>(loc, builder.getI1Type(), iterator);
   }
};

class TableRowIterator : public ForIterator {
   Value tableChunkInfo;
   Value chunk;
   Type elementType;
   db::codegen::FunctionRegistry& functionRegistry;
   Type getValueBufferType(mlir::TypeConverter& typeConverter, OpBuilder& builder, Type type) {
      if (type.isa<mlir::db::StringType>()) {
         return builder.getI32Type();
      }
      if (type.isa<mlir::db::DecimalType>()) {
         return builder.getIntegerType(128);
      }
      return typeConverter.convertType(type);
   }
   struct Column {
      Type completeType;
      Type baseType;
      bool isNullable;
      Type stdType;
      Value offset;
      Value nullMultiplier;
      Value nullBitmap;
      Value values;
      Value varLenBuffer;
   };
   std::vector<Column> columnInfo;

   public:
   TableRowIterator(Value tableChunkInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : ForIterator(tableChunkInfo.getContext()), tableChunkInfo(tableChunkInfo), elementType(elementType), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) override {
      auto indexType = IndexType::get(builder.getContext());
      auto tableChunkInfoType = typeConverter->convertType(tableChunkInfo.getType()).cast<TupleType>();
      auto columnTypes = elementType.dyn_cast_or_null<TupleType>().getTypes();
      auto unpackOp = builder.create<util::UnPackOp>(loc, tableChunkInfoType.getTypes(), tableChunkInfo);
      chunk = unpackOp.getResult(0);

      Value const0 = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value const1 = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value const2 = builder.create<arith::ConstantIndexOp>(loc, 2);

      size_t columnIdx = 0;
      for (auto columnType : columnTypes) {
         auto nullableType = columnType.dyn_cast_or_null<mlir::db::NullableType>();
         bool isNullable = !!nullableType;
         Type completeType = columnType;
         mlir::Type baseType = isNullable ? nullableType.getType() : completeType;
         Value columnId = unpackOp.getResult(1 + columnIdx);
         Value offset;
         if (isIntegerType(baseType, 1) || isNullable) {
            offset = functionRegistry.call(builder, loc, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnOffset, mlir::ValueRange({chunk, columnId}))[0];
         }
         Value bitmapBuffer{};
         auto convertedType = getValueBufferType(*typeConverter, builder, baseType);
         Value valueBuffer;
         if (!isIntegerType(baseType, 1)) {
            valueBuffer = functionRegistry.call(builder, loc, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const1}))[0];
            valueBuffer = builder.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, convertedType, llvm::Optional<int64_t>()), valueBuffer);
         } else {
            valueBuffer = functionRegistry.call(builder, loc, db::codegen::FunctionRegistry::FunctionId::TableChunkGetRawColumnBuffer, mlir::ValueRange({chunk, columnId, const1}))[0];
         }
         Value varLenBuffer{};
         Value nullMultiplier;
         if (isNullable) {
            bitmapBuffer = functionRegistry.call(builder, loc, db::codegen::FunctionRegistry::FunctionId::TableChunkGetRawColumnBuffer, mlir::ValueRange({chunk, columnId, const0}))[0];
            Value bitmapSize = builder.create<util::DimOp>(loc, indexType, bitmapBuffer);
            Value emptyBitmap = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, const0, bitmapSize);
            nullMultiplier = builder.create<mlir::arith::SelectOp>(loc, emptyBitmap, const0, const1);
         }
         if (baseType.isa<mlir::db::StringType>()) {
            varLenBuffer = functionRegistry.call(builder, loc, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const2}))[0];
         }
         columnInfo.push_back({completeType, baseType, isNullable, convertedType, offset, nullMultiplier, bitmapBuffer, valueBuffer, varLenBuffer});
         columnIdx++;
      }
   }
   virtual Value upper(OpBuilder& builder) override {
      return functionRegistry.call(builder, loc, db::codegen::FunctionRegistry::FunctionId::TableChunkNumRows, mlir::ValueRange({chunk}))[0];
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      auto indexType = IndexType::get(builder.getContext());
      Value const1 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 1));
      std::vector<Type> types;
      std::vector<Value> values;
      for (auto column : columnInfo) {
         types.push_back(column.completeType);
         Value val;
         if (column.baseType.isa<db::StringType>()) {
            Value pos1 = builder.create<util::LoadOp>(loc, column.stdType, column.values, index);
            pos1.getDefiningOp()->setAttr("nosideffect", builder.getUnitAttr());

            Value ip1 = builder.create<arith::AddIOp>(loc, indexType, index, const1);
            Value pos2 = builder.create<util::LoadOp>(loc, column.stdType, column.values, ip1);
            pos2.getDefiningOp()->setAttr("nosideffect", builder.getUnitAttr());
            Value len = builder.create<arith::SubIOp>(loc, builder.getI32Type(), pos2, pos1);
            Value pos1AsIndex = builder.create<arith::IndexCastOp>(loc, indexType, pos1);
            Value ptr = builder.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, builder.getI8Type(), llvm::Optional<int64_t>()), column.varLenBuffer, pos1AsIndex);
            val = builder.create<mlir::util::CreateVarLen>(loc, mlir::util::VarLen32Type::get(builder.getContext()), ptr, len);
         } else if (isIntegerType(column.baseType, 1)) {
            Value realPos = builder.create<arith::AddIOp>(loc, indexType, column.offset, index);
            val = mlir::db::codegen::BitUtil::getBit(builder, loc, column.values, realPos);
         } else if (auto decimalType = column.baseType.dyn_cast_or_null<db::DecimalType>()) {
            val = builder.create<util::LoadOp>(loc, column.stdType, column.values, index);
            val.getDefiningOp()->setAttr("nosideffect", builder.getUnitAttr());
            if (typeConverter->convertType(decimalType).cast<mlir::IntegerType>().getWidth() != 128) {
               auto converted = builder.create<arith::TruncIOp>(loc, typeConverter->convertType(decimalType), val);
               val = converted;
            }
         } else if (column.stdType.isa<mlir::IntegerType>() || column.stdType.isa<mlir::FloatType>()) {
            val = builder.create<util::LoadOp>(loc, column.stdType, column.values, index);
            val.getDefiningOp()->setAttr("nosideffect", builder.getUnitAttr());
         } else {
            assert(val && "unhandled type!!");
         }
         if (column.isNullable) {
            Value realPos = builder.create<arith::AddIOp>(loc, indexType, column.offset, index);
            realPos = builder.create<arith::MulIOp>(loc, indexType, column.nullMultiplier, realPos);
            Value isnull = mlir::db::codegen::BitUtil::getBit(builder, loc, column.nullBitmap, realPos, true);
            val = builder.create<mlir::db::CombineNullOp>(loc, column.completeType, val, isnull);
         }
         values.push_back(val);
      }
      return builder.create<mlir::util::PackOp>(loc, TupleType::get(builder.getContext(), types), ValueRange(values));
   }
};
class VectorIterator : public ForIterator {
   Value vector;
   Type elementType;
   Value values;
   Value len;
   using FunctionId = mlir::db::codegen::FunctionRegistry::FunctionId;

   public:
   VectorIterator(mlir::db::codegen::FunctionRegistry& functionRegistry, Value vector, Type elementType) : ForIterator(vector.getContext()), vector(vector), elementType(elementType) {
   }
   virtual void init(OpBuilder& builder) override {
      Type typedPtrType = util::RefType::get(builder.getContext(), elementType, -1);
      auto loaded = builder.create<util::LoadOp>(loc, mlir::TupleType::get(builder.getContext(), TypeRange({builder.getIndexType(), builder.getIndexType(), typedPtrType})), vector, Value());
      auto unpacked = builder.create<util::UnPackOp>(loc, loaded);

      values = unpacked.getResult(2);
      len = unpacked.getResult(0);
   }
   virtual Value upper(OpBuilder& builder) override {
      return len;
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value loaded = builder.create<util::LoadOp>(loc, elementType, values, index);
      return loaded;
   }
};
class ValueOnlyAggrHTIterator : public ForIterator {
   Value ht;
   Type valType;

   public:
   ValueOnlyAggrHTIterator(db::codegen::FunctionRegistry& functionRegistry, Value ht, Type valType) : ForIterator(ht.getContext()), ht(ht), valType(valType) {
   }
   virtual Value upper(OpBuilder& builder) override {
      return builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value undefTuple = builder.create<mlir::util::UndefTupleOp>(loc, TupleType::get(builder.getContext()));
      return builder.create<mlir::util::PackOp>(loc, TupleType::get(builder.getContext(), {undefTuple.getType(), typeConverter->convertType(valType)}), ValueRange({undefTuple, ht}));
   }
};

static std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder) {
   for (size_t i = 0; i < values.size(); i++) {
      values[i] = builder.getRemappedValue(values[i]);
   }
   return values;
}

class WhileIteratorIterationImpl : public mlir::db::CollectionIterationImpl {
   std::unique_ptr<WhileIterator> iterator;

   public:
   WhileIteratorIterationImpl(std::unique_ptr<WhileIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::Location loc, mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(std::function<Value(OpBuilder&)>, ValueRange, OpBuilder)> bodyBuilder) override {
      auto insertionPoint = builder.saveInsertionPoint();

      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      iterator->setLoc(loc);
      Type iteratorType = iterator->iteratorType(builder);
      Value initialIterator = iterator->iterator(builder);
      std::vector<Type> results = {typeConverter.convertType(iteratorType)};
      std::vector<Value> iterValues = {builder.getRemappedValue(initialIterator)};
      for (auto iterArg : iterArgs) {
         results.push_back(typeConverter.convertType(iterArg.getType()));
         iterValues.push_back(builder.getRemappedValue(iterArg));
      }
      auto whileOp = builder.create<mlir::scf::WhileOp>(loc, results, iterValues);
      Block* before = new Block;
      Block* after = new Block;
      whileOp.getBefore().push_back(before);
      whileOp.getAfter().push_back(after);
      for (auto t : results) {
         before->addArgument(t, loc);
         after->addArgument(t, loc);
      }

      builder.setInsertionPointToStart(&whileOp.getBefore().front());
      auto arg1 = whileOp.getBefore().front().getArgument(0);
      Value condition = iterator->iteratorValid(builder, arg1);
      if (flag) {
         Value flagValue = builder.create<mlir::db::GetFlag>(loc, builder.getI1Type(), flag);
         Value shouldContinue = builder.create<mlir::db::NotOp>(loc, builder.getI1Type(), flagValue);
         Value anded = builder.create<mlir::db::AndOp>(loc, builder.getI1Type(), ValueRange({condition, shouldContinue}));
         condition = anded;
      }
      builder.create<mlir::scf::ConditionOp>(loc, builder.getRemappedValue(condition), whileOp.getBefore().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.getAfter().front());
      auto arg2 = whileOp.getAfter().front().getArgument(0);
      auto terminator = builder.create<mlir::db::YieldOp>(loc);
      builder.setInsertionPoint(terminator);
      std::vector<Value> bodyParams = {};
      auto additionalArgs = whileOp.getAfter().front().getArguments().drop_front();
      bodyParams.insert(bodyParams.end(), additionalArgs.begin(), additionalArgs.end());
      auto returnValues = bodyBuilder([&](mlir::OpBuilder& b) { return iterator->iteratorGetCurrentElement(b, arg2); }, bodyParams, builder);
      builder.setInsertionPoint(terminator);
      Value nextIterator = iterator->iteratorNext(builder, arg2);
      returnValues.insert(returnValues.begin(), nextIterator);
      builder.create<mlir::scf::YieldOp>(loc, remap(returnValues, builder));
      builder.eraseOp(terminator);
      Value finalIterator = whileOp.getResult(0);
      builder.restoreInsertionPoint(insertionPoint);
      iterator->iteratorFree(builder, finalIterator);
      auto loopResultValues = whileOp.getResults().drop_front();
      return std::vector<Value>(loopResultValues.begin(), loopResultValues.end());
   }
};
class ForIteratorIterationImpl : public mlir::db::CollectionIterationImpl {
   std::unique_ptr<ForIterator> iterator;

   public:
   ForIteratorIterationImpl(std::unique_ptr<ForIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::Location loc, mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(std::function<Value(OpBuilder&)>, ValueRange, OpBuilder)> bodyBuilder) override {
      if (flag) {
         return implementLoopCondition(loc, iterArgs, flag, typeConverter, builder, bodyBuilder);
      } else {
         return implementLoopSimple(loc, iterArgs, typeConverter, builder, bodyBuilder);
      }
   }
   std::vector<Value> implementLoopSimple(mlir::Location loc, const ValueRange& iterArgs, TypeConverter& typeConverter, ConversionPatternRewriter& builder, std::function<std::vector<Value>(std::function<Value(OpBuilder&)>, ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();
      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      iterator->setLoc(loc);
      auto forOp = builder.create<scf::ForOp>(loc, iterator->lower(builder), iterator->upper(builder), iterator->step(builder), iterArgs.size() ? iterArgs : llvm::None);
      if (iterArgs.size()) {
         builder.setInsertionPointToStart(forOp.getBody());
         builder.create<scf::YieldOp>(loc);
      }
      Operation* terminator = forOp.getBody()->getTerminator();
      builder.setInsertionPointToStart(forOp.getBody());
      std::vector<Value> bodyArguments = {};

      bodyArguments.insert(bodyArguments.end(), forOp.getRegionIterArgs().begin(), forOp.getRegionIterArgs().end());
      Value element;
      auto results = bodyBuilder([&](mlir::OpBuilder& b) { return element = iterator->getElement(b, forOp.getInductionVar()); }, bodyArguments, builder);
      iterator->destroyElement(builder, element);
      if (iterArgs.size()) {
         builder.create<scf::YieldOp>(loc, remap(results, builder));
         builder.eraseOp(terminator);
      }
      builder.restoreInsertionPoint(insertionPoint);
      iterator->down(builder);
      return std::vector<Value>(forOp.getResults().begin(), forOp.getResults().end());
   }
   std::vector<Value> implementLoopCondition(mlir::Location loc, const ValueRange& iterArgs, Value flag, TypeConverter& typeConverter, ConversionPatternRewriter& builder, std::function<std::vector<Value>(std::function<Value(OpBuilder&)>, ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();

      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      iterator->setLoc(loc);
      Type iteratorType = builder.getIndexType();
      Value initialIterator = iterator->lower(builder);
      std::vector<Type> results = {typeConverter.convertType(iteratorType)};
      std::vector<Value> iterValues = {builder.getRemappedValue(initialIterator)};
      for (auto iterArg : iterArgs) {
         results.push_back(typeConverter.convertType(iterArg.getType()));
         iterValues.push_back(builder.getRemappedValue(iterArg));
      }
      Value upper = iterator->upper(builder);
      Value step = iterator->step(builder);
      auto whileOp = builder.create<mlir::scf::WhileOp>(loc, results, iterValues);
      Block* before = new Block;
      Block* after = new Block;

      whileOp.getBefore().push_back(before);
      whileOp.getAfter().push_back(after);
      for (auto t : results) {
         before->addArgument(t, loc);
         after->addArgument(t, loc);
      }

      builder.setInsertionPointToStart(&whileOp.getBefore().front());
      auto arg1 = whileOp.getBefore().front().getArgument(0);
      Value condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, arg1, upper);
      if (flag) {
         Value flagValue = builder.create<mlir::db::GetFlag>(loc, builder.getI1Type(), flag);
         Value shouldContinue = builder.create<mlir::db::NotOp>(loc, builder.getI1Type(), flagValue);
         Value anded = builder.create<mlir::db::AndOp>(loc, builder.getI1Type(), ValueRange({condition, shouldContinue}));
         condition = anded;
      }
      builder.create<mlir::scf::ConditionOp>(loc, builder.getRemappedValue(condition), whileOp.getBefore().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.getAfter().front());
      auto arg2 = whileOp.getAfter().front().getArgument(0);
      Value nextIterator = builder.create<arith::AddIOp>(loc, builder.getIndexType(), arg2, step);
      auto terminator = builder.create<mlir::db::YieldOp>(loc);
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      std::vector<Value> bodyParams = {};
      auto additionalArgs = whileOp.getAfter().front().getArguments().drop_front();
      bodyParams.insert(bodyParams.end(), additionalArgs.begin(), additionalArgs.end());
      auto returnValues = bodyBuilder([&](mlir::OpBuilder& b) { return iterator->getElement(b, arg2); }, bodyParams, builder);
      returnValues.insert(returnValues.begin(), nextIterator);
      builder.setInsertionPoint(terminator);
      builder.create<mlir::scf::YieldOp>(loc, remap(returnValues, builder));
      builder.eraseOp(terminator);
      builder.restoreInsertionPoint(insertionPoint);
      auto loopResultValues = whileOp.getResults().drop_front();
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
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<JoinHtLookupIterator>(collection, generic.getElementType(), functionRegistry));
      } else if (generic.getIteratorName() == "join_ht_mod_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<JoinHtModifyLookupIterator>(collection, generic.getElementType()));
      }
   } else if (auto vector = collectionType.dyn_cast_or_null<mlir::db::VectorType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<VectorIterator>(functionRegistry, collection, vector.getElementType()));
   } else if (auto aggrHt = collectionType.dyn_cast_or_null<mlir::db::AggregationHashtableType>()) {
      if (aggrHt.getKeyType().getTypes().empty()) {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<ValueOnlyAggrHTIterator>(functionRegistry, collection, aggrHt.getValType()));
      } else {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<AggrHtIterator>(collection));
      }
   } else if (auto joinHt = collectionType.dyn_cast_or_null<mlir::db::JoinHashtableType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<JoinHtIterator>(collection));
   }
   return std::unique_ptr<mlir::db::CollectionIterationImpl>();
}