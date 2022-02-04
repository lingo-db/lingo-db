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
      Value currElement = builder.create<mlir::util::SetTupleOp>(loc, typeConverter->convertType(tableInfo.getType()), tableInfo, currElementPtr, 0);
      return currElement;
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value rawValue = functionRegistry.call(builder, loc, mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorValid, iterator)[0];
      Value dbValue = builder.create<mlir::db::TypeCastOp>(loc, mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      functionRegistry.call(builder, loc,mlir::db::codegen::FunctionRegistry::FunctionId::TableChunkIteratorFree, iterator);
   }
};

class JoinHtLookupIterator : public WhileIterator {
   Value iteratorInfo;
   Value initialPos;
   Value vec;

   public:
   JoinHtLookupIterator(Value tableInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : WhileIterator(tableInfo.getContext()), iteratorInfo(tableInfo) {
   }
   virtual void init(OpBuilder& builder) override {
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, typeConverter->convertType(iteratorInfo.getType()).cast<TupleType>().getTypes(), iteratorInfo);
      initialPos = unpacked.getResult(0);
      vec = unpacked.getResult(1);
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      return builder.getIndexType();
   }

   virtual Value iterator(OpBuilder& builder) override {
      return initialPos;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      Value loaded = builder.create<util::LoadOp>(loc, vec.getType().cast<mlir::util::RefType>().getElementType(), vec, iterator);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      return unpacked.getResult(0);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      Value loaded = builder.create<util::LoadOp>(loc, vec.getType().cast<mlir::util::RefType>().getElementType(), vec, iterator);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      return unpacked.getResult(1);
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value maxValue = builder.create<arith::ConstantIndexOp>(loc, 0xFFFFFFFFFFFFFFFF);
      Value rawValue = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, iterator, maxValue);
      Value dbValue = builder.create<mlir::db::TypeCastOp>(loc, mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
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
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, typeConverter->convertType(hashTable.getType()).cast<TupleType>().getTypes(), hashTable);
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
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, typeConverter->convertType(hashTable.getType()).cast<TupleType>().getTypes(), hashTable);
      values = unpacked.getResult(1);
      len = unpacked.getResult(0);
   }
   virtual Value upper(OpBuilder& builder) override {
      return len;
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value loaded = builder.create<util::LoadOp>(loc, values.getType().cast<mlir::util::RefType>().getElementType(), values, index);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      return unpacked.getResult(2);
   }
};
class JoinHtModifyLookupIterator : public WhileIterator {
   Value iteratorInfo;
   Value initialPos;
   Value vec;

   public:
   JoinHtModifyLookupIterator(Value tableInfo, Type elementType) : WhileIterator(tableInfo.getContext()), iteratorInfo(tableInfo) {
   }
   virtual void init(OpBuilder& builder) override {
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, typeConverter->convertType(iteratorInfo.getType()).cast<TupleType>().getTypes(), iteratorInfo);
      initialPos = unpacked.getResult(0);
      vec = unpacked.getResult(1);
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      return builder.getIndexType();
   }

   virtual Value iterator(OpBuilder& builder) override {
      return initialPos;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      Value loaded = builder.create<util::LoadOp>(loc, vec.getType().cast<mlir::util::RefType>().getElementType(), vec, iterator);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);

      return unpacked.getResult(0);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      auto bucketType = vec.getType().cast<mlir::util::RefType>().getElementType();
      auto elemType = bucketType.cast<TupleType>().getTypes()[1];

      auto valType = elemType.cast<TupleType>().getTypes()[1];
      Value offsetOne = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

      Value loaded = builder.create<util::LoadOp>(loc, bucketType, vec, iterator);
      Value bucketAddress = builder.create<util::ElementPtrOp>(loc, util::RefType::get(builder.getContext(), bucketType, Optional<int64_t>()), vec, iterator);
      Value elemAddress = builder.create<util::ElementPtrOp>(loc, util::RefType::get(builder.getContext(), elemType, Optional<int64_t>()), bucketAddress, offsetOne);

      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      Value loadedValue = unpacked.getResult(1);

      Value valAddress = builder.create<util::ElementPtrOp>(loc, util::RefType::get(builder.getContext(), valType, Optional<int64_t>()), elemAddress, offsetOne);

      return builder.create<mlir::util::PackOp>(loc, mlir::ValueRange{loadedValue, valAddress});
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      Value maxValue = builder.create<arith::ConstantIndexOp>(loc, 0xFFFFFFFFFFFFFFFF);
      Value rawValue = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, iterator, maxValue);
      Value dbValue = builder.create<mlir::db::TypeCastOp>(loc, mlir::db::BoolType::get(builder.getContext()), rawValue);
      return dbValue;
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
         auto dbtype = columnType.dyn_cast_or_null<mlir::db::DBType>();
         Value columnId = unpackOp.getResult(1 + columnIdx);
         Value offset;
         if(dbtype.isa<mlir::db::BoolType>()||dbtype.isNullable()){
            offset=functionRegistry.call(builder, loc, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnOffset, mlir::ValueRange({chunk, columnId}))[0];
         }
         Value bitmapBuffer{};
         auto convertedType = getValueBufferType(*typeConverter, builder, dbtype);
         Value valueBuffer;
         if (!dbtype.isa<mlir::db::BoolType>()) {
            valueBuffer = functionRegistry.call(builder, loc,db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const1}))[0];
            valueBuffer = builder.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context,convertedType,llvm::Optional<int64_t>()),valueBuffer);
         } else {
            valueBuffer = functionRegistry.call(builder, loc,db::codegen::FunctionRegistry::FunctionId::TableChunkGetRawColumnBuffer, mlir::ValueRange({chunk, columnId, const1}))[0];
         }
         Value varLenBuffer{};
         Value nullMultiplier;
         if (dbtype.isNullable()) {
            bitmapBuffer = functionRegistry.call(builder, loc,db::codegen::FunctionRegistry::FunctionId::TableChunkGetRawColumnBuffer, mlir::ValueRange({chunk, columnId, const0}))[0];
            Value bitmapSize = builder.create<util::DimOp>(loc, indexType, bitmapBuffer);
            Value emptyBitmap = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, const0, bitmapSize);
            nullMultiplier = builder.create<mlir::SelectOp>(loc, emptyBitmap, const0, const1);
         }
         if (dbtype.isVarLen()) {
            varLenBuffer = functionRegistry.call(builder, loc,db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const2}))[0];
         }
         columnInfo.push_back({dbtype, convertedType, offset, nullMultiplier, bitmapBuffer, valueBuffer, varLenBuffer});
         columnIdx++;
      }
   }
   virtual Value upper(OpBuilder& builder) override {
      return functionRegistry.call(builder,loc, db::codegen::FunctionRegistry::FunctionId::TableChunkNumRows, mlir::ValueRange({chunk}))[0];
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      auto indexType = IndexType::get(builder.getContext());
      Value const1 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 1));
      std::vector<Type> types;
      std::vector<Value> values;
      for (auto column : columnInfo) {
         types.push_back(column.type);
         Value val;
         if (column.type.isa<db::StringType>()) {
            Value pos1 = builder.create<util::LoadOp>(loc, column.stdType, column.values, index);
            Value ip1 = builder.create<arith::AddIOp>(loc, indexType, index, const1);
            Value pos2 = builder.create<util::LoadOp>(loc, column.stdType, column.values, ip1);
            Value len = builder.create<arith::SubIOp>(loc, builder.getI32Type(), pos2, pos1);
            Value pos1AsIndex = builder.create<arith::IndexCastOp>(loc, pos1, indexType);
            Value ptr=builder.create<util::ElementPtrOp>(loc,util::RefType::get(context,builder.getI8Type(),llvm::Optional<int64_t>()),column.varLenBuffer,pos1AsIndex);
            val = builder.create<mlir::util::CreateVarLen>(loc, mlir::util::VarLen32Type::get(builder.getContext()), ptr,len);
         } else if (column.type.isa<db::BoolType>()) {
            Value realPos = builder.create<arith::AddIOp>(loc, indexType, column.offset, index);
            val = mlir::db::codegen::BitUtil::getBit(builder, loc, column.values, realPos);
         } else if (auto decimalType = column.type.dyn_cast_or_null<db::DecimalType>()) {
            val = builder.create<util::LoadOp>(loc, column.stdType, column.values, index);
            if (typeConverter->convertType(decimalType.getBaseType()).cast<mlir::IntegerType>().getWidth() != 128) {
               auto converted = builder.create<arith::TruncIOp>(loc, typeConverter->convertType(decimalType.getBaseType()), val);
               val = converted;
            }
         } else if (column.stdType.isa<mlir::IntegerType>() || column.stdType.isa<mlir::FloatType>()) {
            val = builder.create<util::LoadOp>(loc, column.stdType, column.values, index);
         } else {
            assert(val && "unhandled type!!");
         }
         if (column.type.isNullable()) {
            Value realPos = builder.create<arith::AddIOp>(loc, indexType, column.offset, index);
            realPos = builder.create<arith::MulIOp>(loc, indexType, column.nullMultiplier, realPos);
            Value isnull = mlir::db::codegen::BitUtil::getBit(builder, loc, column.nullBitmap, realPos, true);
            val = builder.create<mlir::db::CombineNullOp>(loc, column.type, val, isnull);
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
      auto unpacked = builder.create<util::UnPackOp>(loc, TypeRange({builder.getIndexType(), builder.getIndexType(), typedPtrType}), vector);

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

static std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder){
   for(size_t i=0;i<values.size();i++){
      values[i]=builder.getRemappedValue(values[i]);
   }
   return values;
}
class WhileIteratorIterationImpl : public mlir::db::CollectionIterationImpl {
   std::unique_ptr<WhileIterator> iterator;

   public:
   WhileIteratorIterationImpl(std::unique_ptr<WhileIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::Location loc, mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
      auto insertionPoint = builder.saveInsertionPoint();

      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      iterator->setLoc(loc);
      Type iteratorType = iterator->iteratorType(builder);
      Value initialIterator = iterator->iterator(builder);
      std::vector<Type> results = {iteratorType};
      std::vector<Value> iterValues = {initialIterator};
      for (auto iterArg : iterArgs) {
         results.push_back(iterArg.getType());
         iterValues.push_back(iterArg);
      }
      auto whileOp = builder.create<mlir::db::WhileOp>(loc, results, iterValues);
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
         Value flagValue = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), flag);
         Value shouldContinue = builder.create<mlir::db::NotOp>(loc, mlir::db::BoolType::get(builder.getContext()), flagValue);
         Value anded = builder.create<mlir::db::AndOp>(loc, mlir::db::BoolType::get(builder.getContext()), ValueRange({condition, shouldContinue}));
         condition = anded;
      }
      builder.create<mlir::db::ConditionOp>(loc, condition, whileOp.before().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.after().front());
      auto arg2 = whileOp.after().front().getArgument(0);
      Value currElement = iterator->iteratorGetCurrentElement(builder, arg2);
      Value nextIterator = iterator->iteratorNext(builder, arg2);
      auto terminator = builder.create<mlir::db::YieldOp>(loc);
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      std::vector<Value> bodyParams = {currElement};
      auto additionalArgs = whileOp.after().front().getArguments().drop_front();
      bodyParams.insert(bodyParams.end(), additionalArgs.begin(), additionalArgs.end());
      auto returnValues = bodyBuilder(bodyParams, builder);
      returnValues.insert(returnValues.begin(), nextIterator);
      builder.setInsertionPoint(terminator);
      builder.create<mlir::db::YieldOp>(loc, remap(returnValues,builder));
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
   virtual std::vector<Value> implementLoop(mlir::Location loc, mlir::ValueRange iterArgs, Value flag, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
      if (flag) {
         return implementLoopCondition(loc, iterArgs, flag, typeConverter, builder, bodyBuilder);
      } else {
         return implementLoopSimple(loc, iterArgs, typeConverter, builder, bodyBuilder);
      }
   }
   std::vector<Value> implementLoopSimple(mlir::Location loc, const ValueRange& iterArgs, TypeConverter& typeConverter, ConversionPatternRewriter& builder, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
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
      auto element = iterator->getElement(builder, forOp.getInductionVar());
      std::vector<Value> bodyArguments = {element};

      bodyArguments.insert(bodyArguments.end(), forOp.getRegionIterArgs().begin(), forOp.getRegionIterArgs().end());
      auto results = bodyBuilder(bodyArguments, builder);
      iterator->destroyElement(builder, element);
      if (iterArgs.size()) {
         builder.create<scf::YieldOp>(loc, remap(results,builder));
         builder.eraseOp(terminator);
      }
      builder.restoreInsertionPoint(insertionPoint);
      iterator->down(builder);
      return std::vector<Value>(forOp.getResults().begin(), forOp.getResults().end());
   }
   std::vector<Value> implementLoopCondition(mlir::Location loc, const ValueRange& iterArgs, Value flag, TypeConverter& typeConverter, ConversionPatternRewriter& builder, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();

      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);
      iterator->setLoc(loc);
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
      auto whileOp = builder.create<mlir::db::WhileOp>(loc, results, iterValues);
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
      Value condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, arg1, upper);
      Value dbcondition = builder.create<mlir::db::TypeCastOp>(loc, mlir::db::BoolType::get(builder.getContext()), condition);
      if (flag) {
         Value flagValue = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), flag);
         Value shouldContinue = builder.create<mlir::db::NotOp>(loc, mlir::db::BoolType::get(builder.getContext()), flagValue);
         Value anded = builder.create<mlir::db::AndOp>(loc, mlir::db::BoolType::get(builder.getContext()), ValueRange({dbcondition, shouldContinue}));
         dbcondition = anded;
      }
      builder.create<mlir::db::ConditionOp>(loc, dbcondition, whileOp.before().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.after().front());
      auto arg2 = whileOp.after().front().getArgument(0);
      Value currElement = iterator->getElement(builder, arg2);
      Value nextIterator = builder.create<arith::AddIOp>(loc, builder.getIndexType(), arg2, step);
      auto terminator = builder.create<mlir::db::YieldOp>(loc);
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      std::vector<Value> bodyParams = {currElement};
      auto additionalArgs = whileOp.after().front().getArguments().drop_front();
      bodyParams.insert(bodyParams.end(), additionalArgs.begin(), additionalArgs.end());
      auto returnValues = bodyBuilder(bodyParams, builder);
      returnValues.insert(returnValues.begin(), nextIterator);
      builder.setInsertionPoint(terminator);
      builder.create<mlir::db::YieldOp>(loc, remap(returnValues,builder));
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