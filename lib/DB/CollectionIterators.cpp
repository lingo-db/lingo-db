#include "mlir/Dialect/DB/IR/DBCollectionType.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Dialect/util/UtilOps.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
using namespace mlir;
namespace {

//declare external function or return reference to already existing one
static mlir::FuncOp getOrInsertFn(mlir::OpBuilder& rewriter,
                                  ModuleOp module, const std::string& name, FunctionType fnType) {
   if (FuncOp funcOp = module.lookupSymbol<FuncOp>(name))
      return funcOp;
   PatternRewriter::InsertionGuard insertGuard(rewriter);
   rewriter.setInsertionPointToStart(module.getBody());
   FuncOp funcOp = rewriter.create<FuncOp>(module.getLoc(), name, fnType, rewriter.getStringAttr("private"));
   funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
   return funcOp;
}
} // end namespace
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
   virtual void init(OpBuilder builder) = 0;
   virtual Value iterator(OpBuilder builder) = 0;
   virtual Value iteratorNext(OpBuilder builder, Value iterator) = 0;
   virtual Value iteratorGetCurrentElement(OpBuilder builder, Value iterator) = 0;
   virtual Value iteratorValid(OpBuilder builder, Value iterator) = 0;
   virtual void iteratorFree(OpBuilder builder, Value iterator) = 0;
   virtual ~WhileIterator() {}
};
class TableIterator : public WhileIterator {
   Value tableInfo;
   Type ptrType;
   Type i1Type;

   public:
   TableIterator(Value tableInfo) : tableInfo(tableInfo) {
   }
   virtual void init(OpBuilder builder) {
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      ptrType = MemRefType::get({}, i8Type);
      i1Type = IntegerType::get(builder.getContext(), 1);
   }

   virtual Value iterator(OpBuilder builder) {
      auto iteratorInitFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_init", builder.getFunctionType({ptrType}, {ptrType}));

      Value tablePtr = builder.create<util::GetTupleOp>(builder.getUnknownLoc(), ptrType, tableInfo, 0);
      Value initialIterator = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorInitFn, mlir::ValueRange({tablePtr})).getResult(0);
      return initialIterator;
   }
   virtual Value iteratorNext(OpBuilder builder, Value iterator) {
      auto iteratorNextFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_next", builder.getFunctionType({ptrType}, {ptrType}));
      Value nextIterator = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorNextFn, mlir::ValueRange({iterator})).getResult(0);
      return nextIterator;
   }
   virtual Value iteratorGetCurrentElement(OpBuilder builder, Value iterator) {
      auto iteratorCurrFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_curr", builder.getFunctionType({ptrType}, {ptrType}));
      Value currElementPtr = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorCurrFn, mlir::ValueRange({iterator})).getResult(0);
      Value currElement = builder.create<mlir::util::SetTupleOp>(builder.getUnknownLoc(), typeConverter->convertType(tableInfo.getType()), tableInfo, currElementPtr, 0);
      return currElement;
   }
   virtual Value iteratorValid(OpBuilder builder, Value iterator) {
      auto iteratorValidFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_valid", builder.getFunctionType({ptrType}, {i1Type}));
      Value valid = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorValidFn, mlir::ValueRange({iterator})).getResult(0);
      return valid;
   }
   virtual void iteratorFree(OpBuilder builder, Value iterator) {
      auto iteratorFreeFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_free", builder.getFunctionType({ptrType}, {}));
      builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorFreeFn, mlir::ValueRange({iterator}));
   }
};
class BitUtil {
   public:
   static Value getBit(OpBuilder builder, Value bits, Value pos, bool negated = false) {
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto indexType = IndexType::get(builder.getContext());
      Value const3 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 3));
      Value const7 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 7));

      Value div8 = builder.create<UnsignedShiftRightOp>(builder.getUnknownLoc(), indexType, pos, const3);
      Value rem8 = builder.create<AndOp>(builder.getUnknownLoc(), indexType, pos, const7);
      Value loadedByte = builder.create<memref::LoadOp>(builder.getUnknownLoc(), bits, ValueRange({div8}));
      Value rem8AsByte = builder.create<IndexCastOp>(builder.getUnknownLoc(), rem8, i8Type);
      Value shifted = builder.create<UnsignedShiftRightOp>(builder.getUnknownLoc(), i8Type, loadedByte, rem8AsByte);
      Value res = builder.create<TruncateIOp>(builder.getUnknownLoc(), i1Type, shifted);
      if (negated) {
         Value constTrue = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), i1Type, builder.getIntegerAttr(i1Type, 1));
         res = builder.create<XOrOp>(builder.getUnknownLoc(), res, constTrue); //negate
      }
      return res;
   }
};
class WhileIteratorIterationImpl : public mlir::db::CollectionIterationImpl {
   std::unique_ptr<WhileIterator> iterator;

   public:
   WhileIteratorIterationImpl(std::unique_ptr<WhileIterator> iterator) : iterator(std::move(iterator)) {
   }
   virtual std::vector<Value> implementLoop(mlir::TypeRange iterArgTypes, mlir::TypeConverter& typeConverter, OpBuilder builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
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
   Type getValueBufferType(mlir::TypeConverter& typeConverter, OpBuilder builder, mlir::db::DBType type) {
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
   TableRowIterator(Value tableChunkInfo, Type elementType) : tableChunkInfo(tableChunkInfo), elementType(elementType) {
   }
   virtual std::vector<Value> implementLoop(mlir::TypeRange iterArgTypes, mlir::TypeConverter& typeConverter, OpBuilder builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      std::vector<Column> columnInfo;
      auto ptrType = MemRefType::get({}, i8Type);
      auto byteRange = MemRefType::get({-1}, i8Type);
      auto indexType = IndexType::get(builder.getContext());
      auto tableChunkInfoType = typeConverter.convertType(tableChunkInfo.getType()).cast<TupleType>();
      auto columnTypes = elementType.dyn_cast_or_null<TupleType>().getTypes();
      auto unpackOp = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tableChunkInfoType.getTypes(), tableChunkInfo);
      Value chunk = unpackOp.getResult(0);

      auto tableChunkNumRowsFn = getOrInsertFn(builder, parentModule, "table_chunk_num_rows", builder.getFunctionType({ptrType}, {indexType}));
      auto tableChunkColumnBufferFn = getOrInsertFn(builder, parentModule, "table_chunk_get_column_buffer", builder.getFunctionType({ptrType, indexType, indexType}, {byteRange}));
      auto tableChunkColumnOffsetFn = getOrInsertFn(builder, parentModule, "table_chunk_get_column_offset", builder.getFunctionType({ptrType, indexType}, {indexType}));

      Value numRows = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkNumRowsFn, mlir::ValueRange({chunk})).getResult(0);
      Value const0 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 0));
      Value const1 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 1));
      Value const2 = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), indexType, builder.getIntegerAttr(indexType, 2));

      size_t columnIdx = 0;
      for (auto columnType : columnTypes) {
         columnType.dump();
         auto dbtype = columnType.dyn_cast_or_null<mlir::db::DBType>();
         Value columnId = unpackOp.getResult(1 + columnIdx);
         Value offset = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkColumnOffsetFn, mlir::ValueRange({chunk, columnId})).getResult(0);
         Value bitmapBuffer{};
         auto convertedType = getValueBufferType(typeConverter, builder, dbtype);
         auto typeSize = builder.create<util::SizeOfOp>(builder.getUnknownLoc(), indexType, mlir::TypeAttr::get(convertedType));
         Value valueBuffer0 = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkColumnBufferFn, mlir::ValueRange({chunk, columnId, const1})).getResult(0);
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
            bitmapBuffer = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkColumnBufferFn, mlir::ValueRange({chunk, columnId, const0})).getResult(0);
         }
         if (dbtype.isVarLen()) {
            varLenBuffer = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkColumnBufferFn, mlir::ValueRange({chunk, columnId, const2})).getResult(0);
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
            val = BitUtil::getBit(builder, column.values, realPos);
         } else if (column.stdType.isa<mlir::IntegerType>() || column.stdType.isa<mlir::FloatType>()) {
            val = builder.create<memref::LoadOp>(builder.getUnknownLoc(), column.values, ValueRange({forOp.getInductionVar()}));
         } else {
            assert(val && "unhandled type!!");
         }
         if (column.type.isNullable()) {
            Value realPos = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), indexType, column.offset, forOp.getInductionVar());
            Value isnull = BitUtil::getBit(builder, column.nullBitmap, realPos, true);
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
std::unique_ptr<mlir::db::CollectionIterationImpl> mlir::db::GenericIterableType::getIterationImpl(Value collection) const {
   if (getIteratorName() == "table_chunk_iterator") {
      return std::make_unique<WhileIteratorIterationImpl>(std::make_unique<TableIterator>(collection));
   } else if (getIteratorName() == "table_row_iterator") {
      return std::make_unique<TableRowIterator>(collection, getElementType());
   }
   return std::unique_ptr<CollectionIterationImpl>();
}