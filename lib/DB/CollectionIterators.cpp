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
   return funcOp;
}
} // end namespace
class TableChunkIterator : public mlir::db::CollectionIterator {
   Value tableInfo;

   public:
   TableChunkIterator(Value tableInfo) : tableInfo(tableInfo) {
   }
   virtual std::vector<Value> implementLoop(mlir::TypeRange iterArgTypes, mlir::TypeConverter& typeConverter, OpBuilder builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      auto i1Type = IntegerType::get(builder.getContext(), 1);

      auto ptrType = MemRefType::get({}, i8Type);
      Value tablePtr = builder.create<util::GetTupleOp>(builder.getUnknownLoc(), ptrType, tableInfo, 0);

      auto iteratorInitFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_init", builder.getFunctionType({ptrType}, {ptrType}));
      auto iteratorValidFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_valid", builder.getFunctionType({ptrType}, {i1Type}));
      auto iteratorNextFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_next", builder.getFunctionType({ptrType}, {ptrType}));
      auto iteratorCurrFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_curr", builder.getFunctionType({ptrType}, {ptrType}));

      auto iteratorFreeFn = getOrInsertFn(builder, parentModule, "table_chunk_iterator_free", builder.getFunctionType({ptrType}, {}));

      Value initialIterator = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorInitFn, mlir::ValueRange({tablePtr})).getResult(0);
      auto whileOp = builder.create<mlir::scf::WhileOp>(builder.getUnknownLoc(), TypeRange({ptrType}), ValueRange({initialIterator}));
      Block* before = new Block;
      Block* after = new Block;
      whileOp.before().push_back(before);
      whileOp.after().push_back(after);
      before->addArgument(ptrType);
      after->addArgument(ptrType);

      builder.setInsertionPointToStart(&whileOp.before().front());
      auto arg1 = whileOp.before().front().getArgument(0);
      Value valid = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorValidFn, mlir::ValueRange({arg1})).getResult(0);
      builder.create<mlir::scf::ConditionOp>(builder.getUnknownLoc(), valid, arg1);
      builder.setInsertionPointToStart(&whileOp.after().front());
      auto arg2 = whileOp.after().front().getArgument(0);
      Value currElementPtr = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorCurrFn, mlir::ValueRange({arg2})).getResult(0);
      Value currElement = builder.create<mlir::util::SetTupleOp>(builder.getUnknownLoc(), typeConverter.convertType(tableInfo.getType()), tableInfo, currElementPtr, 0);
      Value nextIterator = builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorNextFn, mlir::ValueRange({arg2})).getResult(0);

      builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), nextIterator);
      builder.setInsertionPoint(nextIterator.getDefiningOp());
      bodyBuilder(currElement, builder);
      Value finalIterator = whileOp.getResult(0);
      builder.restoreInsertionPoint(insertionPoint);
      builder.create<mlir::CallOp>(builder.getUnknownLoc(), iteratorFreeFn, mlir::ValueRange({finalIterator}));
      return {};
   }
};
class TableRowIterator : public mlir::db::CollectionIterator {
   Value tableChunkInfo;
   Type elementType;

   public:
   TableRowIterator(Value tableChunkInfo, Type elementType) : tableChunkInfo(tableChunkInfo), elementType(elementType) {
   }
   virtual std::vector<Value> implementLoop(mlir::TypeRange iterArgTypes, mlir::TypeConverter& typeConverter, OpBuilder builder, ModuleOp parentModule, std::function<std::vector<Value>(ValueRange, OpBuilder)> bodyBuilder) {
      auto insertionPoint = builder.saveInsertionPoint();
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);
      std::vector<std::tuple<Value, Value, Value, Value, mlir::db::DBType>> columnInfo;
      auto ptrType = MemRefType::get({}, i8Type);
      auto indexType = IndexType::get(builder.getContext());

      auto ptr64Type = MemRefType::get({}, indexType);

      auto tableChunkInfoType = typeConverter.convertType(tableChunkInfo.getType()).cast<TupleType>();
      auto columnTypes = elementType.dyn_cast_or_null<TupleType>().getTypes();
      auto unpackOp = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tableChunkInfoType.getTypes(), tableChunkInfo);
      Value chunk = unpackOp.getResult(0);

      auto tableChunkNumRowsFn = getOrInsertFn(builder, parentModule, "table_chunk_num_rows", builder.getFunctionType({ptrType}, {indexType}));
      auto tableChunkColumnBufferFn = getOrInsertFn(builder, parentModule, "table_chunk_get_column_buffer", builder.getFunctionType({ptrType, indexType, indexType}, {ptrType}));
      auto tableChunkColumnOffsetFn = getOrInsertFn(builder, parentModule, "table_chunk_get_column_offset", builder.getFunctionType({ptrType, indexType}, {indexType}));
      auto tableColumnGetInt64 = getOrInsertFn(builder, parentModule, "table_column_get_int_64", builder.getFunctionType({ptrType, indexType, indexType}, {mlir::db::IntType::get(builder.getContext(), false, 64)}));
      auto tableColumnGetInt32 = getOrInsertFn(builder, parentModule, "table_column_get_int_32", builder.getFunctionType({ptrType, indexType, indexType}, {mlir::db::IntType::get(builder.getContext(), false, 32)}));
      auto tableColumnGetFloat64 = getOrInsertFn(builder, parentModule, "table_column_get_float_64", builder.getFunctionType({ptrType, indexType, indexType}, {mlir::db::FloatType::get(builder.getContext(), false, 64)}));
      auto tableColumnGetFloat32 = getOrInsertFn(builder, parentModule, "table_column_get_float_32", builder.getFunctionType({ptrType, indexType, indexType}, {mlir::db::FloatType::get(builder.getContext(), false, 32)}));
      auto tableColumnGetDecimal = getOrInsertFn(builder, parentModule, "table_column_get_decimal", builder.getFunctionType({ptrType, indexType, indexType}, {mlir::db::IntType::get(builder.getContext(), false, 128)}));
      auto tableColumnGetBinary = getOrInsertFn(builder, parentModule, "table_column_get_binary", builder.getFunctionType({ptrType, ptrType, indexType, indexType, ptr64Type}, {ptrType}));
      auto tableColumnIsNull = getOrInsertFn(builder, parentModule, "table_column_is_null", builder.getFunctionType({ptrType, indexType, indexType}, {i1Type}));
      auto tableColumnGetBool = getOrInsertFn(builder, parentModule, "table_column_get_bool", builder.getFunctionType({ptrType, indexType, indexType}, {mlir::db::BoolType::get(builder.getContext())}));

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
         Value valueBuffer = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkColumnBufferFn, mlir::ValueRange({chunk, columnId, const1})).getResult(0);
         Value varLenBuffer{};
         if (dbtype.isNullable()) {
            bitmapBuffer = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkColumnBufferFn, mlir::ValueRange({chunk, columnId, const0})).getResult(0);
         }
         if (dbtype.isVarLen()) {
            varLenBuffer = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableChunkColumnBufferFn, mlir::ValueRange({chunk, columnId, const2})).getResult(0);
         }
         columnInfo.push_back({offset, bitmapBuffer, valueBuffer, varLenBuffer, dbtype});
         columnIdx++;
      }

      auto forOp = builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), const0, numRows, const1);
      builder.setInsertionPointToStart(forOp.getBody());
      std::vector<Type> types;
      std::vector<Value> values;
      for (auto column : columnInfo) {
         auto [offset, bitmapBuffer, valueBuffer, varLenBuffer, dbtype] = column;
         types.push_back(dbtype);
         Value val;
         if (auto intType = dbtype.dyn_cast_or_null<db::IntType>()) {
            val = builder.create<mlir::CallOp>(builder.getUnknownLoc(), intType.getWidth() == 64 ? tableColumnGetInt64 : tableColumnGetInt32, mlir::ValueRange({valueBuffer, offset, forOp.getInductionVar()})).getResult(0);
         } else if (auto dateType = dbtype.dyn_cast_or_null<db::DateType>()) {
            val = builder.create<mlir::CallOp>(builder.getUnknownLoc(), dateType.getUnit() == mlir::db::DateUnitAttr::millisecond ? tableColumnGetInt64 : tableColumnGetInt32, mlir::ValueRange({valueBuffer, offset, forOp.getInductionVar()})).getResult(0);
         } else if (auto floatType = dbtype.dyn_cast_or_null<db::FloatType>()) {
            val = builder.create<mlir::CallOp>(builder.getUnknownLoc(), floatType.getWidth() == 64 ? tableColumnGetFloat64 : tableColumnGetFloat32, mlir::ValueRange({valueBuffer, offset, forOp.getInductionVar()})).getResult(0);
         } else if (dbtype.isa<db::StringType>()) {
            Value lenPtr = builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), ptr64Type);
            Value v = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableColumnGetBinary, mlir::ValueRange({valueBuffer, varLenBuffer, offset, forOp.getInductionVar(), lenPtr})).getResult(0);
            Value len = builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(), lenPtr);
            auto strDynamicType = MemRefType::get({-1}, IntegerType::get(builder.getContext(), 8));

            val = builder.create<mlir::memref::ReinterpretCastOp>(builder.getUnknownLoc(), strDynamicType, v, const0, len, const0);
         } else if (dbtype.isa<db::DecimalType>()) {
            val = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableColumnGetDecimal, mlir::ValueRange({valueBuffer, offset, forOp.getInductionVar()})).getResult(0);
         } else if (dbtype.isa<db::BoolType>()) {
            val = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableColumnGetBool, mlir::ValueRange({valueBuffer, offset, forOp.getInductionVar()})).getResult(0);
         }
         assert(val && "unhandled type!!");
         if (dbtype.isNullable()) {
            Value isnull = builder.create<mlir::CallOp>(builder.getUnknownLoc(), tableColumnIsNull, mlir::ValueRange({bitmapBuffer, offset, forOp.getInductionVar()})).getResult(0);
            val = builder.create<mlir::db::CombineNullOp>(builder.getUnknownLoc(), dbtype, val, isnull);
            values.push_back(val);
         }
      }
      Value tuple = builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), ValueRange(values));
      bodyBuilder(tuple, builder);

      builder.restoreInsertionPoint(insertionPoint);
      forOp->dump();

      return {};
   }
};
std::unique_ptr<mlir::db::CollectionIterator> mlir::db::GenericIterableType::getIterator(Value collection) const {
   if (getIteratorName() == "table_chunk_iterator") {
      return std::make_unique<TableChunkIterator>(collection);
   } else if (getIteratorName() == "table_row_iterator") {
      return std::make_unique<TableRowIterator>(collection, getElementType());
   }
   return std::unique_ptr<CollectionIterator>();
}