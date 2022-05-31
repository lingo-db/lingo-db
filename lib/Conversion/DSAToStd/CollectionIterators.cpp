#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"

#include "mlir/Conversion/DSAToStd/CollectionIteration.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include "runtime-defs/DataSourceIteration.h"
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
   mlir::Value len;
   ForIterator(mlir::MLIRContext* context) : context(context), loc(mlir::UnknownLoc::get(context)) {}

   public:
   void setLoc(mlir::Location loc) {
      this->loc = loc;
   }
   virtual void init(OpBuilder& builder){};
   virtual Value lower(OpBuilder& builder) {
      return builder.create<arith::ConstantIndexOp>(loc, 0);
   }

   virtual Value upper(OpBuilder& builder) {
      return len;
   }
   virtual Value step(OpBuilder& builder) {
      return builder.create<arith::ConstantIndexOp>(loc, 1);
   }
   virtual Value getElement(OpBuilder& builder, Value index) = 0;
   virtual ~ForIterator() {}
   void setTypeConverter(TypeConverter* typeConverter) {
      ForIterator::typeConverter = typeConverter;
   }
};
class TableIterator2 : public WhileIterator {
   Value tableInfo;
   mlir::dsa::RecordBatchType recordBatchType;

   public:
   TableIterator2(Value tableInfo, mlir::dsa::RecordBatchType recordBatchType) : WhileIterator(tableInfo.getContext()), tableInfo(tableInfo), recordBatchType(recordBatchType) {}

   virtual Type iteratorType(OpBuilder& builder) override {
      return mlir::util::RefType::get(builder.getContext(), IntegerType::get(builder.getContext(), 8));
   }

   virtual Value iterator(OpBuilder& builder) override {
      return tableInfo;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      rt::DataSourceIteration::next(builder, loc)({iterator});
      return tableInfo;
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      mlir::Value recordBatchInfoPtr;
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(&iterator.getParentRegion()->getParentOfType<mlir::func::FuncOp>().getBody().front());
         recordBatchInfoPtr = builder.create<mlir::util::AllocaOp>(loc, mlir::util::RefType::get(builder.getContext(), typeConverter->convertType(recordBatchType)), mlir::Value());
      }
      rt::DataSourceIteration::access(builder, loc)({iterator, recordBatchInfoPtr});
      return builder.create<mlir::util::LoadOp>(loc, recordBatchInfoPtr, mlir::Value());
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      return rt::DataSourceIteration::isValid(builder, loc)({iterator})[0];
   }
   virtual void iteratorFree(OpBuilder& builder, Value iterator) override {
      rt::DataSourceIteration::end(builder, loc)({iterator});
   }
};

class JoinHtLookupIterator : public WhileIterator {
   Value iteratorInfo;
   Type ptrType;
   Value hash;
   Value ptr;
   TupleType tupleType;
   bool modifiable;

   public:
   JoinHtLookupIterator(Value tableInfo, Type elementType, bool modifiable = false) : WhileIterator(tableInfo.getContext()), iteratorInfo(tableInfo), modifiable(modifiable) {
   }
   virtual Type iteratorType(OpBuilder& builder) override {
      tupleType = iteratorInfo.getType().cast<mlir::TupleType>();
      ptrType = tupleType.getType(0);
      return ptrType;
   }

   virtual Value iterator(OpBuilder& builder) override {
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, iteratorInfo);
      ptr = unpacked.getResult(0);
      hash = unpacked.getResult(1);
      return ptr;
   }
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) override {
      auto i8PtrType = mlir::util::RefType::get(builder.getContext(), builder.getI8Type());
      Value nextPtr = builder.create<util::TupleElementPtrOp>(loc, mlir::util::RefType::get(builder.getContext(), i8PtrType), iterator, 0);
      mlir::Value next = builder.create<mlir::util::LoadOp>(loc, nextPtr, mlir::Value());
      next = builder.create<util::GenericMemrefCastOp>(loc, ptrType, next);
      return builder.create<mlir::util::FilterTaggedPtr>(loc, next.getType(), next, hash);
   }
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) override {
      auto bucketType = iterator.getType().cast<mlir::util::RefType>().getElementType();
      auto elemType = bucketType.cast<TupleType>().getTypes()[1];
      auto valType = elemType.cast<TupleType>().getTypes()[1];
      Value elemAddress = builder.create<util::TupleElementPtrOp>(loc, util::RefType::get(builder.getContext(), elemType), iterator, 1);
      Value loadedValue = builder.create<util::LoadOp>(loc, elemType, elemAddress);
      if (modifiable) {
         Value valAddress = builder.create<util::TupleElementPtrOp>(loc, util::RefType::get(builder.getContext(), valType), elemAddress, 1);
         return builder.create<mlir::util::PackOp>(loc, mlir::ValueRange{loadedValue, valAddress});
      } else {
         return loadedValue;
      }
   }
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) override {
      return builder.create<mlir::util::IsRefValidOp>(loc, builder.getI1Type(), iterator);
   }
};

class JoinHtIterator : public ForIterator {
   Value hashTable;
   Value values;

   public:
   JoinHtIterator(Value hashTable) : ForIterator(hashTable.getContext()), hashTable(hashTable) {
   }
   virtual void init(OpBuilder& builder) override {
      auto loaded = builder.create<util::LoadOp>(loc, hashTable);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      values = unpacked.getResult(4);
      len = unpacked.getResult(2);
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value loaded = builder.create<util::LoadOp>(loc, values, index);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      return unpacked.getResult(1);
   }
};
class AggrHtIterator : public ForIterator {
   Value hashTable;
   Value values;

   public:
   AggrHtIterator(Value hashTable) : ForIterator(hashTable.getContext()), hashTable(hashTable) {
   }
   virtual void init(OpBuilder& builder) override {
      auto loaded = builder.create<mlir::util::LoadOp>(loc, hashTable);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      values = unpacked.getResult(3);
      len = unpacked.getResult(1);
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value loaded = builder.create<util::LoadOp>(loc, values, index);
      auto unpacked = builder.create<mlir::util::UnPackOp>(loc, loaded);
      return unpacked.getResult(2);
   }
};
class RecordBatchIterator : public ForIterator {
   mlir::Value recordBatch;
   mlir::dsa::RecordBatchType recordBatchType;

   public:
   RecordBatchIterator(Value recordBatch, Type recordBatchType) : ForIterator(recordBatch.getContext()), recordBatch(recordBatch), recordBatchType(recordBatchType.cast<mlir::dsa::RecordBatchType>()) {
   }
   virtual Value upper(OpBuilder& builder) override {
      return builder.create<mlir::util::GetTupleOp>(loc, builder.getIndexType(), recordBatch, 0);
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      return builder.create<mlir::util::PackOp>(loc, typeConverter->convertType(mlir::dsa::RecordType::get(builder.getContext(), recordBatchType.getRowType())), mlir::ValueRange({index, recordBatch}));
   }
};

class VectorIterator : public ForIterator {
   Value vector;
   Value values;

   public:
   VectorIterator(Value vector) : ForIterator(vector.getContext()), vector(vector) {
   }
   virtual void init(OpBuilder& builder) override {
      auto loaded = builder.create<util::LoadOp>(loc, vector, Value());
      auto unpacked = builder.create<util::UnPackOp>(loc, loaded);

      values = unpacked.getResult(2);
      len = unpacked.getResult(0);
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      return builder.create<util::LoadOp>(loc, values, index);
   }
};
class ValueOnlyAggrHTIterator : public ForIterator {
   Value ht;
   Type valType;

   public:
   ValueOnlyAggrHTIterator(Value ht, Type valType) : ForIterator(ht.getContext()), ht(ht), valType(valType) {
   }
   virtual Value upper(OpBuilder& builder) override {
      return builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1));
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      Value undefTuple = builder.create<mlir::util::UndefOp>(loc, TupleType::get(builder.getContext()));
      Value val = builder.create<mlir::util::LoadOp>(loc, ht);
      return builder.create<mlir::util::PackOp>(loc, ValueRange({undefTuple, val}));
   }
};

static std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder) {
   for (size_t i = 0; i < values.size(); i++) {
      values[i] = builder.getRemappedValue(values[i]);
   }
   return values;
}

class WhileIteratorIterationImpl : public mlir::dsa::CollectionIterationImpl {
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
         Value flagValue = builder.create<mlir::dsa::GetFlag>(loc, builder.getI1Type(), flag);
         Value falseValue = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 0));
         Value shouldContinue = builder.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, flagValue, falseValue);
         Value anded = builder.create<mlir::arith::AndIOp>(loc, builder.getI1Type(), ValueRange({condition, shouldContinue}));
         condition = anded;
      }
      builder.create<mlir::scf::ConditionOp>(loc, builder.getRemappedValue(condition), whileOp.getBefore().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.getAfter().front());
      auto arg2 = whileOp.getAfter().front().getArgument(0);
      auto terminator = builder.create<mlir::dsa::YieldOp>(loc);
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
class ForIteratorIterationImpl : public mlir::dsa::CollectionIterationImpl {
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
      if (iterArgs.size()) {
         builder.create<scf::YieldOp>(loc, remap(results, builder));
         builder.eraseOp(terminator);
      }
      builder.restoreInsertionPoint(insertionPoint);
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
         Value flagValue = builder.create<mlir::dsa::GetFlag>(loc, builder.getI1Type(), flag);
         Value falseValue = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 0));
         Value shouldContinue = builder.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, flagValue, falseValue);
         Value anded = builder.create<mlir::arith::AndIOp>(loc, builder.getI1Type(), ValueRange({condition, shouldContinue}));
         condition = anded;
      }
      builder.create<mlir::scf::ConditionOp>(loc, builder.getRemappedValue(condition), whileOp.getBefore().front().getArguments());
      builder.setInsertionPointToStart(&whileOp.getAfter().front());
      auto arg2 = whileOp.getAfter().front().getArgument(0);
      Value nextIterator = builder.create<arith::AddIOp>(loc, builder.getIndexType(), arg2, step);
      auto terminator = builder.create<mlir::dsa::YieldOp>(loc);
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
std::unique_ptr<mlir::dsa::CollectionIterationImpl> mlir::dsa::CollectionIterationImpl::getImpl(Type collectionType, Value loweredCollection) {
   if (auto generic = collectionType.dyn_cast_or_null<mlir::dsa::GenericIterableType>()) {
      if (generic.getIteratorName() == "table_chunk_iterator") {
         if (auto recordBatchType = generic.getElementType().dyn_cast_or_null<mlir::dsa::RecordBatchType>()) {
            return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<TableIterator2>(loweredCollection, recordBatchType));
         }
      } else if (generic.getIteratorName() == "join_ht_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<JoinHtLookupIterator>(loweredCollection, generic.getElementType(), false));
      } else if (generic.getIteratorName() == "join_ht_mod_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<JoinHtLookupIterator>(loweredCollection, generic.getElementType(), true));
      }
   } else if (auto vector = collectionType.dyn_cast_or_null<mlir::dsa::VectorType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<VectorIterator>(loweredCollection));
   } else if (auto aggrHt = collectionType.dyn_cast_or_null<mlir::dsa::AggregationHashtableType>()) {
      if (aggrHt.getKeyType().getTypes().empty()) {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<ValueOnlyAggrHTIterator>(loweredCollection, aggrHt.getValType()));
      } else {
         return std::make_unique<ForIteratorIterationImpl>(std::make_unique<AggrHtIterator>(loweredCollection));
      }
   } else if (auto joinHt = collectionType.dyn_cast_or_null<mlir::dsa::JoinHashtableType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<JoinHtIterator>(loweredCollection));
   } else if (auto recordBatch = collectionType.dyn_cast_or_null<mlir::dsa::RecordBatchType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<RecordBatchIterator>(loweredCollection, recordBatch));
   }
   return std::unique_ptr<mlir::dsa::CollectionIterationImpl>();
}
