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
   ModuleOp parentModule;
   mlir::TypeConverter* typeConverter;

   public:
   void setParentModule(const ModuleOp& parentModule) {
      WhileIterator::parentModule = parentModule;
   }
   void setTypeConverter(TypeConverter* typeConverter) {
      WhileIterator::typeConverter = typeConverter;
   }

   public:
   virtual void init(OpBuilder& builder) = 0;
   virtual Value iterator(OpBuilder& builder) = 0;
   virtual Value iteratorNext(OpBuilder& builder, Value iterator) = 0;
   virtual Value iteratorGetCurrentElement(OpBuilder& builder, Value iterator) = 0;
   virtual Value iteratorValid(OpBuilder& builder, Value iterator) = 0;
   virtual void iteratorFree(OpBuilder& builder, Value iterator) = 0;
   virtual ~WhileIterator() {}
};
class TableIterator : public WhileIterator {
   Value tableInfo;
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   TableIterator(Value tableInfo, db::codegen::FunctionRegistry& functionRegistry) : tableInfo(tableInfo), functionRegistry(functionRegistry) {
   }
   virtual void init(OpBuilder& builder) override {
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
class WhileIteratorIterationImpl : public mlir::db::CollectionIterationImpl {
   std::unique_ptr<WhileIterator> iterator;

   public:
   WhileIteratorIterationImpl(std::unique_ptr<WhileIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::TypeRange iterArgTypes, mlir::TypeConverter& typeConverter, OpBuilder& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
      auto insertionPoint = builder.saveInsertionPoint();
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto ptrType = MemRefType::get({}, i8Type);
      iterator->setParentModule(parentModule);
      iterator->setTypeConverter(&typeConverter);
      iterator->init(builder);

      Value initialIterator = iterator->iterator(builder);
      auto whileOp = builder.create<mlir::scf::WhileOp>(builder.getUnknownLoc(), TypeRange({ptrType}), ValueRange({initialIterator}));
      Block* before = new Block;
      Block* after = new Block;
      whileOp.before().push_back(before);
      whileOp.after().push_back(after);
      before->addArgument(ptrType);
      after->addArgument(ptrType);

      builder.setInsertionPointToStart(&whileOp.before().front());
      auto arg1 = whileOp.before().front().getArgument(0);
      builder.create<mlir::scf::ConditionOp>(builder.getUnknownLoc(), iterator->iteratorValid(builder, arg1), arg1);
      builder.setInsertionPointToStart(&whileOp.after().front());
      auto arg2 = whileOp.after().front().getArgument(0);
      Value currElement = iterator->iteratorGetCurrentElement(builder, arg2);
      Value nextIterator = iterator->iteratorNext(builder, arg2);
      builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), nextIterator);
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      bodyBuilder(currElement, builder);
      Value finalIterator = whileOp.getResult(0);
      builder.restoreInsertionPoint(insertionPoint);
      iterator->iteratorFree(builder, finalIterator);
      return {};
   }
};
class TableRowIterator : public mlir::db::CollectionIterationImpl {
   Value tableChunkInfo;
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
      Value nullBitmap;
      Value values;
      Value varLenBuffer;
   };

   public:
   TableRowIterator(Value tableChunkInfo, Type elementType, db::codegen::FunctionRegistry& functionRegistry) : tableChunkInfo(tableChunkInfo), elementType(elementType), functionRegistry(functionRegistry) {
   }
   virtual std::vector<Value> implementLoop(mlir::TypeRange iterArgTypes, mlir::TypeConverter& typeConverter, OpBuilder& builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) override {
      auto insertionPoint = builder.saveInsertionPoint();
      auto indexType=IndexType::get(builder.getContext());
      std::vector<Column> columnInfo;
      auto tableChunkInfoType = typeConverter.convertType(tableChunkInfo.getType()).cast<TupleType>();
      auto columnTypes = elementType.dyn_cast_or_null<TupleType>().getTypes();
      auto unpackOp = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tableChunkInfoType.getTypes(), tableChunkInfo);
      Value chunk = unpackOp.getResult(0);

      Value numRows = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkNumRows, mlir::ValueRange({chunk}))[0];
      Value const0 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 0));
      Value const1 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 1));
      Value const2 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 2));

      size_t columnIdx = 0;
      for (auto columnType : columnTypes) {
         columnType.dump();
         auto dbtype = columnType.dyn_cast_or_null<mlir::db::DBType>();
         Value columnId = unpackOp.getResult(1 + columnIdx);
         Value offset = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnOffset, mlir::ValueRange({chunk, columnId}))[0];
         Value bitmapBuffer{};
         auto convertedType = getValueBufferType(typeConverter, builder, dbtype);
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
         if (dbtype.isNullable()) {
            bitmapBuffer = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const0}))[0];
         }
         if (dbtype.isVarLen()) {
            varLenBuffer = functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::TableChunkGetColumnBuffer, mlir::ValueRange({chunk, columnId, const2}))[0];
         }
         columnInfo.push_back({dbtype, convertedType, offset, bitmapBuffer, valueBuffer, varLenBuffer});
         columnIdx++;
      }

      auto forOp = builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), const0, numRows, const1);
      builder.setInsertionPointToStart(forOp.getBody());
      std::vector<Type> types;
      std::vector<Value> values;
      for (auto column : columnInfo) {
         types.push_back(column.type);
         Value val;
         if (column.type.isa<db::StringType>()) {
            Value pos1 = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({forOp.getInductionVar()}));
            Value ip1 = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), indexType, forOp.getInductionVar(), const1);
            Value pos2 = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({ip1}));
            Value len = builder.create<mlir::SubIOp>(builder.getUnknownLoc(), builder.getI32Type(), pos2, pos1);
            Value pos1AsIndex = builder.create<IndexCastOp>(builder.getUnknownLoc(), pos1, indexType);
            Value lenAsIndex = builder.create<IndexCastOp>(builder.getUnknownLoc(), len, indexType);
            val = builder.create<mlir::memref::SubViewOp>(builder.getUnknownLoc(), column.varLenBuffer, mlir::ValueRange({pos1AsIndex}), mlir::ValueRange({lenAsIndex}), mlir::ValueRange({const0}));
         } else if (column.type.isa<db::BoolType>()) {
            Value realPos = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), indexType, column.offset, forOp.getInductionVar());
            val = mlir::db::codegen::BitUtil::getBit(builder, column.values, realPos);
         } else if (column.stdType.isa<mlir::IntegerType>() || column.stdType.isa<mlir::FloatType>()) {
            val = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({forOp.getInductionVar()}));
         } else {
            assert(val && "unhandled type!!");
         }
         if (column.type.isNullable()) {
            Value realPos = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), indexType, column.offset, forOp.getInductionVar());
            Value isnull = mlir::db::codegen::BitUtil::getBit(builder, column.nullBitmap, realPos, true);
            val = builder.create<mlir::db::CombineNullOp>(builder.getUnknownLoc(), column.type, val, isnull);
            values.push_back(val);
         }
      }
      Value tuple = builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), ValueRange(values));
      bodyBuilder(tuple, builder);

      builder.restoreInsertionPoint(insertionPoint);
      return {};
   }
};
std::unique_ptr<mlir::db::CollectionIterationImpl> mlir::db::CollectionIterationImpl::getImpl(Type collectionType, Value collection, mlir::db::codegen::FunctionRegistry& functionRegistry) {
   if (auto generic = collectionType.dyn_cast_or_null<mlir::db::GenericIterableType>()) {
      if (generic.getIteratorName() == "table_chunk_iterator") {
         return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<TableIterator>(collection, functionRegistry));
      } else if (generic.getIteratorName() == "table_row_iterator") {
         return std::make_unique<TableRowIterator>(collection, generic.getElementType(),functionRegistry);
      }
   }
   return std::unique_ptr<mlir::db::CollectionIterationImpl>();
}