#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/DSA/IR/DSATypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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

class BufferIterator : public ForIterator {
   Value buffer;
   Value values;

   public:
   BufferIterator(Value buffer) : ForIterator(buffer.getContext()), buffer(buffer) {
   }
   virtual void init(OpBuilder& builder) override {
      values = builder.create<util::BufferGetRef>(loc, buffer.getType().cast<mlir::util::BufferType>().getElementType(), buffer);
      len = builder.create<util::BufferGetLen>(loc, mlir::IndexType::get(context), buffer);
   }
   virtual Value getElement(OpBuilder& builder, Value index) override {
      return builder.create<util::ArrayElementPtrOp>(loc, mlir::util::RefType::get(buffer.getContext(), values.getType().cast<mlir::util::RefType>().getElementType()), values, index);
   }
};

static std::vector<Value> remap(std::vector<Value> values, ConversionPatternRewriter& builder) {
   for (size_t i = 0; i < values.size(); i++) {
      values[i] = builder.getRemappedValue(values[i]);
   }
   return values;
}
class ForIteratorIterationImpl : public mlir::dsa::CollectionIterationImpl {
   std::unique_ptr<ForIterator> iterator;

   public:
   ForIteratorIterationImpl(std::unique_ptr<ForIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::Location loc, mlir::ValueRange iterArgs, mlir::TypeConverter& typeConverter, ConversionPatternRewriter& builder, ModuleOp parentModule, std::function<std::vector<Value>(std::function<Value(OpBuilder&)>, ValueRange, OpBuilder)> bodyBuilder) override {
      return implementLoopSimple(loc, iterArgs, typeConverter, builder, bodyBuilder);
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
};
std::unique_ptr<mlir::dsa::CollectionIterationImpl> mlir::dsa::CollectionIterationImpl::getImpl(Type collectionType, Value loweredCollection) {
   if (auto vector = collectionType.dyn_cast_or_null<mlir::util::BufferType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<BufferIterator>(loweredCollection));
   } else if (auto recordBatch = collectionType.dyn_cast_or_null<mlir::dsa::RecordBatchType>()) {
      return std::make_unique<ForIteratorIterationImpl>(std::make_unique<RecordBatchIterator>(loweredCollection, recordBatch));
   }
   return std::unique_ptr<mlir::dsa::CollectionIterationImpl>();
}
