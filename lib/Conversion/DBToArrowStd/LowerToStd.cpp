#include "arrow/type_fwd.h"
#include "mlir-support/mlir-support.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/DBToArrowStd/CollectionIteration.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Conversion/DBToArrowStd/NullHandler.h"
#include "mlir/Conversion/DBToArrowStd/SerializationUtil.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <iostream>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {



class ForOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit ForOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::ForOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::ForOpAdaptor forOpAdaptor(operands);
      auto forOp = cast<mlir::db::ForOp>(op);
      auto argumentTypes = forOp.region().getArgumentTypes();
      auto collectionType = forOp.collection().getType().dyn_cast_or_null<mlir::db::CollectionType>();

      auto iterator = mlir::db::CollectionIterationImpl::getImpl(collectionType, forOp.collection(), functionRegistry);

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      std::vector<Value> results = iterator->implementLoop(forOpAdaptor.initArgs(), *typeConverter, rewriter, parentModule, [&](ValueRange values, OpBuilder builder) {
         auto yieldOp = cast<mlir::db::YieldOp>(forOp.getBody()->getTerminator());
         rewriter.mergeBlockBefore(forOp.getBody(), &*builder.getInsertionPoint(), values);
         std::vector<Value> results(yieldOp.results().begin(), yieldOp.results().end());
         rewriter.eraseOp(yieldOp);
         return results;
      });
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         llvm::dbgs() << forOp.region().getBlocks().size() << "\n";
         forOp.region().push_back(new Block());
         forOp.region().front().addArguments(argumentTypes);
         rewriter.setInsertionPointToStart(&forOp.region().front());
         rewriter.create<mlir::db::YieldOp>(forOp.getLoc());
      }

      rewriter.replaceOp(op, results);
      return success();
   }
};
class GetTableLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit GetTableLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::GetTable::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto getTableOp = cast<mlir::db::GetTable>(op);
      auto tableName = rewriter.create<mlir::db::ConstantOp>(rewriter.getUnknownLoc(), mlir::db::StringType::get(rewriter.getContext(), false), rewriter.getStringAttr(getTableOp.tablename()));
      auto tablePtr = functionRegistry.call(rewriter, db::codegen::FunctionRegistry::FunctionId::ExecutionContextGetTable, mlir::ValueRange({getTableOp.execution_context(), tableName}))[0];
      rewriter.replaceOp(getTableOp, tablePtr);
      return success();
   }
};
class TableScanLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit TableScanLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::TableScan::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::TableScanAdaptor adaptor(operands);
      auto tablescan = cast<mlir::db::TableScan>(op);
      std::vector<Type> types;
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto ptrType = MemRefType::get({}, i8Type);
      auto indexType = IndexType::get(rewriter.getContext());

      std::vector<Value> values;
      types.push_back(ptrType);
      auto tablePtr = adaptor.table();
      values.push_back(tablePtr);
      for (auto c : tablescan.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         types.push_back(indexType);
         auto columnName = rewriter.create<mlir::db::ConstantOp>(rewriter.getUnknownLoc(), mlir::db::StringType::get(rewriter.getContext(), false), stringAttr);
         auto columnId = functionRegistry.call(rewriter, db::codegen::FunctionRegistry::FunctionId::TableGetColumnId, mlir::ValueRange({tablePtr, columnName}))[0];
         values.push_back(columnId);
      }
      rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, mlir::TupleType::get(rewriter.getContext(), types), values);
      return success();
   }
};
static Value getArrowDataType(OpBuilder& builder, db::codegen::FunctionRegistry& functionRegistry, db::DBType type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;
   int typeConstant = 0;
   int param1 = 0;
   int param2 = 0;
   auto loc = builder.getUnknownLoc();
   if (auto intType = type.dyn_cast_or_null<mlir::db::IntType>()) {
      switch (intType.getWidth()) {
         case 8: typeConstant = arrow::Type::type::INT8; break;
         case 16: typeConstant = arrow::Type::type::INT16; break;
         case 32: typeConstant = arrow::Type::type::INT32; break;
         case 64: typeConstant = arrow::Type::type::INT64; break;
      }
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      typeConstant = arrow::Type::type::DECIMAL128;
      param1 = decimalType.getP();
      param2 = decimalType.getS();
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      typeConstant = arrow::Type::type::BOOL;
   } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
      switch (floatType.getWidth()) {
         case 16: typeConstant = arrow::Type::type::HALF_FLOAT; break;
         case 32: typeConstant = arrow::Type::type::FLOAT; break;
         case 64: typeConstant = arrow::Type::type::DOUBLE; break;
      }
   } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
      typeConstant = arrow::Type::type::STRING;
   } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
      if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
         typeConstant = arrow::Type::type::DATE32;
      } else {
         typeConstant = arrow::Type::type::DATE64;
      }
   }
   //TODO: also implement date types etc

   Value arrowTypeConstant = builder.create<mlir::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(typeConstant));
   Value arrowTypeParam1 = builder.create<mlir::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param1));
   Value arrowTypeParam2 = builder.create<mlir::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param2));

   Value arrowType = functionRegistry.call(builder, FunctionId::ArrowGetType2Param, ValueRange({arrowTypeConstant, arrowTypeParam1, arrowTypeParam2}))[0];
   return arrowType;
}
class CreateTableBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateTableBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateTableBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto createTB = cast<mlir::db::CreateTableBuilder>(op);
      auto loc = rewriter.getUnknownLoc();

      Value schema = functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaCreate, {})[0];
      TupleType rowType = createTB.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
      size_t i = 0;
      for (auto c : createTB.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         auto dbType = rowType.getType(i).cast<mlir::db::DBType>();
         auto arrowType = getArrowDataType(rewriter, functionRegistry, dbType);
         auto columnName = rewriter.create<mlir::db::ConstantOp>(loc, mlir::db::StringType::get(rewriter.getContext(), false), stringAttr);
         Value typeNullable = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), dbType.isNullable()));

         functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaAddField, ValueRange({schema, arrowType, typeNullable, columnName}));
         i += 1;
      }
      schema = functionRegistry.call(rewriter, FunctionId::ArrowTableSchemaBuild, schema)[0];
      Value tableBuilder = functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderCreate, schema)[0];
      rewriter.replaceOp(op, tableBuilder);
      return success();
   }
};
class CreateVectorBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateVectorBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateVectorBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      Value vectorBuilder = functionRegistry.call(rewriter, FunctionId::VectorBuilderCreate, {})[0];
      rewriter.replaceOp(op, vectorBuilder);
      return success();
   }
};

static db::codegen::FunctionRegistry::FunctionId getStoreFunc(db::codegen::FunctionRegistry& functionRegistry, db::DBType type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;
   if (auto intType = type.dyn_cast_or_null<mlir::db::IntType>()) {
      switch (intType.getWidth()) {
         case 8: return FunctionId::ArrowTableBuilderAddInt8;
         case 16: return FunctionId::ArrowTableBuilderAddInt16;
         case 32: return FunctionId::ArrowTableBuilderAddInt32;
         case 64: return FunctionId::ArrowTableBuilderAddInt64;
      }
   } else if (auto boolType = type.dyn_cast_or_null<mlir::db::BoolType>()) {
      return FunctionId::ArrowTableBuilderAddBool;
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      return FunctionId::ArrowTableBuilderAddDecimal;
   } else if (auto floatType = type.dyn_cast_or_null<mlir::db::FloatType>()) {
      switch (floatType.getWidth()) {
         case 32: return FunctionId ::ArrowTableBuilderAddFloat32;
         case 64: return FunctionId ::ArrowTableBuilderAddFloat64;
      }
   } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
      return FunctionId::ArrowTableBuilderAddBinary;
   } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
      if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
         return FunctionId::ArrowTableBuilderAddDate32;
      } else {
         return FunctionId::ArrowTableBuilderAddDate64;
      }
   }
   //TODO: implement other types too
   return FunctionId::ArrowTableBuilderAddInt32;
}
Value serializeForVector(OpBuilder& builder, TypeConverter* converter, Value vectorBuilder, Value element, Type type, db::codegen::FunctionRegistry& functionRegistry) {
   if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
      auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
      std::vector<Value> serializedValues;
      std::vector<Type> types;
      auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
      for (size_t i = 0; i < tupleType.size(); i++) {
         Value currVal = unPackOp.getResult(i);
         Value serialized = serializeForVector(builder, converter, vectorBuilder, currVal, originalTupleType.getType(i), functionRegistry);
         serializedValues.push_back(serialized);
         types.push_back(serialized.getType());
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
   } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
      if (stringType.isNullable()) {
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())}), element);
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::VectorBuilderAddNullableVarLen, ValueRange({vectorBuilder, unPackOp.getResult(0), unPackOp.getResult(1)}))[0];
      } else {
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::VectorBuilderAddVarLen, ValueRange({vectorBuilder, element}))[0];
      }
   } else {
      return element;
   }
}
Value serializeForAggrHT(OpBuilder& builder, TypeConverter* converter, Value vectorBuilder, Value element, Type type, db::codegen::FunctionRegistry& functionRegistry) {
   if (auto originalTupleType = type.dyn_cast_or_null<TupleType>()) {
      auto tupleType = element.getType().dyn_cast_or_null<TupleType>();
      std::vector<Value> serializedValues;
      std::vector<Type> types;
      auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), element);
      for (size_t i = 0; i < tupleType.size(); i++) {
         Value currVal = unPackOp.getResult(i);
         Value serialized = serializeForAggrHT(builder, converter, vectorBuilder, currVal, originalTupleType.getType(i), functionRegistry);
         serializedValues.push_back(serialized);
         types.push_back(serialized.getType());
      }
      return builder.create<mlir::util::PackOp>(builder.getUnknownLoc(), TupleType::get(builder.getContext(), types), serializedValues);
   } else if (auto stringType = type.dyn_cast_or_null<db::StringType>()) {
      if (stringType.isNullable()) {
         auto unPackOp = builder.create<mlir::util::UnPackOp>(builder.getUnknownLoc(), TypeRange({builder.getI1Type(), converter->convertType(stringType.getBaseType())}), element);
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::AggrHtBuilderAddNullableVarLen, ValueRange({vectorBuilder, unPackOp.getResult(0), unPackOp.getResult(1)}))[0];
      } else {
         return functionRegistry.call(builder, db::codegen::FunctionRegistry::FunctionId::AggrHtBuilderAddVarLen, ValueRange({vectorBuilder, element}))[0];
      }
   } else {
      return element;
   }
}

class BuilderMergeLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit BuilderMergeLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BuilderMerge::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::BuilderMergeAdaptor mergeOpAdaptor(operands);
      auto mergeOp = cast<mlir::db::BuilderMerge>(op);
      auto loc = rewriter.getUnknownLoc();
      if (auto tableBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         TupleType rowType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
         auto loweredTypes = mergeOpAdaptor.val().getType().cast<TupleType>().getTypes();
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, loweredTypes, mergeOpAdaptor.val());
         size_t i = 0;
         Value falseValue = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         for (auto v : unPackOp.vals()) {
            Value isNull;
            if (mergeOp.val().getType().cast<TupleType>().getType(i).cast<db::DBType>().isNullable()) {
               auto nullUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, v.getType().cast<TupleType>().getTypes(), v);
               isNull = nullUnpacked.getResult(0);
               v = nullUnpacked->getResult(1);
            } else {
               isNull = falseValue;
            }
            Value columnId = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            functionRegistry.call(rewriter, getStoreFunc(functionRegistry, rowType.getType(i).cast<mlir::db::DBType>()), ValueRange({mergeOpAdaptor.builder(), columnId, isNull, v}));
            i++;
         }
         functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderFinishRow, mergeOpAdaptor.builder());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      } else if (auto vectorBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         Value v = serializeForVector(rewriter, typeConverter, mergeOpAdaptor.builder(), mergeOpAdaptor.val(), mergeOp.val().getType(), functionRegistry);
         Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), v.getType());
         Value ptr = functionRegistry.call(rewriter, FunctionId::VectorBuilderMerge, ValueRange({mergeOpAdaptor.builder(), elementSize}))[0];
         Value typedPtr = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), v.getType(), llvm::Optional<int64_t>()), ptr);
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), v, typedPtr, Value());
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      } else if (auto aggrHTBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::AggrHTBuilderType>()) {
         auto ptrType = MemRefType::get({}, rewriter.getIntegerType(8));

         auto loweredTypes = mergeOpAdaptor.val().getType().cast<TupleType>().getTypes();

         auto unPacked = rewriter.create<mlir::util::UnPackOp>(loc, loweredTypes, mergeOpAdaptor.val())->getResults();
         Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.getIndexType(), unPacked[0]);

         Value serializedKey = serializeForAggrHT(rewriter, typeConverter, mergeOpAdaptor.builder(), unPacked[0], aggrHTBuilderType.getKeyType(), functionRegistry);
         Value serializedVal = serializeForAggrHT(rewriter, typeConverter, mergeOpAdaptor.builder(), unPacked[1], aggrHTBuilderType.getValType(), functionRegistry);

         Type keyMemrefType = util::GenericMemrefType::get(rewriter.getContext(), serializedKey.getType(), llvm::Optional<int64_t>());
         Type valMemrefType = util::GenericMemrefType::get(rewriter.getContext(), serializedVal.getType(), llvm::Optional<int64_t>());
         Value allocaKey = rewriter.create<mlir::util::AllocaOp>(loc, keyMemrefType, Value());
         Value allocaVal = rewriter.create<mlir::util::AllocaOp>(loc, valMemrefType, Value());
         rewriter.create<util::StoreOp>(loc, serializedKey, allocaKey, Value());
         rewriter.create<util::StoreOp>(loc, serializedVal, allocaVal, Value());
         Value plainMemrefKey = rewriter.create<mlir::util::ToMemrefOp>(loc, ptrType, allocaKey);
         Value plainMemrefVal = rewriter.create<mlir::util::ToMemrefOp>(loc, ptrType, allocaVal);
         functionRegistry.call(rewriter, FunctionId::AggrHtBuilderMerge, ValueRange({mergeOpAdaptor.builder(), hashed, plainMemrefKey, plainMemrefVal}));
         rewriter.replaceOp(op, mergeOpAdaptor.builder());
      }

      return success();
   }
};
class SortOpLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit SortOpLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::SortOp::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      static size_t id = 0;
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::SortOpAdaptor sortOpAdaptor(operands);
      auto loweredVectorType = sortOpAdaptor.toSort().getType();
      auto sortOp = cast<mlir::db::SortOp>(op);
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto ptrType = MemRefType::get({}, rewriter.getIntegerType(8));
      Type elementType = sortOp.toSort().getType().cast<mlir::db::VectorType>().getElementType();
      Type serializedType = mlir::db::codegen::SerializationUtil::serializedType(rewriter, *typeConverter, elementType);
      FuncOp funcOp;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto byteRangeType = MemRefType::get({-1}, rewriter.getIntegerType(8));
         funcOp = rewriter.create<FuncOp>(parentModule.getLoc(), "db_sort_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({byteRangeType, ptrType, ptrType}), TypeRange(mlir::db::BoolType::get(rewriter.getContext()))));
         funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({byteRangeType, ptrType, ptrType}));
         funcOp.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value varLenData = funcBody->getArgument(0);
         Value left = funcBody->getArgument(1);
         Value right = funcBody->getArgument(2);

         Value genericMemrefLeft = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedType, llvm::Optional<int64_t>()), left);
         Value genericMemrefRight = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedType, llvm::Optional<int64_t>()), right);
         Value serializedTupleLeft = rewriter.create<util::LoadOp>(sortOp.getLoc(), serializedType, genericMemrefLeft, Value());
         Value serializedTupleRight = rewriter.create<util::LoadOp>(sortOp.getLoc(), serializedType, genericMemrefRight, Value());
         Value tupleLeft = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleLeft, elementType);
         Value tupleRight = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleRight, elementType);
         auto terminator = rewriter.create<mlir::ReturnOp>(sortOp.getLoc());
         Block* sortLambda = &sortOp.region().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.mergeBlockBefore(sortLambda, terminator, {tupleLeft, tupleRight});
         mlir::db::YieldOp yieldOp = mlir::cast<mlir::db::YieldOp>(terminator->getPrevNode());
         Value x = yieldOp.results()[0];
         x.setType(rewriter.getI1Type()); //todo: bad hack ;)
         rewriter.create<mlir::ReturnOp>(sortOp.getLoc(), x);
         rewriter.eraseOp(sortLambdaTerminator);
         rewriter.eraseOp(terminator);
      }
      Value functionPointer = rewriter.create<mlir::ConstantOp>(sortOp->getLoc(), funcOp.type(), rewriter.getSymbolRefAttr(funcOp.sym_name()));
      Type vectorMemrefType = util::GenericMemrefType::get(rewriter.getContext(), loweredVectorType, llvm::Optional<int64_t>());
      Value allocaVec = rewriter.create<mlir::util::AllocaOp>(sortOp->getLoc(), vectorMemrefType, Value());
      Value allocaNewVec = rewriter.create<mlir::util::AllocaOp>(sortOp->getLoc(), vectorMemrefType, Value());
      rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), sortOpAdaptor.toSort(), allocaVec, Value());
      Value plainMemref = rewriter.create<mlir::util::ToMemrefOp>(sortOp->getLoc(), ptrType, allocaVec);
      Value plainMemrefNew = rewriter.create<mlir::util::ToMemrefOp>(sortOp->getLoc(), ptrType, allocaNewVec);
      Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), serializedType);
      functionRegistry.call(rewriter, FunctionId::SortVector, {plainMemref, elementSize, functionPointer, plainMemrefNew});
      Value newVector = rewriter.create<util::LoadOp>(sortOp.getLoc(), loweredVectorType, allocaNewVec, Value());
      rewriter.replaceOp(op, newVector);
      return success();
   }
};
class CreateAggrHTBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateAggrHTBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateAggrHTBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;

      static size_t id = 0;
      mlir::db::SortOpAdaptor createAdaptor(operands);
      auto createOp = cast<mlir::db::CreateAggrHTBuilder>(op);

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto ptrType = MemRefType::get({}, rewriter.getIntegerType(8));
      Type keyType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getKeyType();
      TupleType keyTupleType = keyType.cast<mlir::TupleType>();
      Type valType = createOp.builder().getType().cast<mlir::db::AggrHTBuilderType>().getValType();

      Type serializedValType = mlir::db::codegen::SerializationUtil::serializedType(rewriter, *typeConverter, valType);
      Type serializedKeyType = mlir::db::codegen::SerializationUtil::serializedType(rewriter, *typeConverter, keyType);
      FuncOp compareFunc;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto byteRangeType = MemRefType::get({-1}, rewriter.getIntegerType(8));
         compareFunc = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_compare" + std::to_string(id++), rewriter.getFunctionType(TypeRange({byteRangeType, ptrType, ptrType}), TypeRange(mlir::db::BoolType::get(rewriter.getContext()))));
         compareFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({byteRangeType, ptrType, ptrType}));
         compareFunc.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value varLenData = funcBody->getArgument(0);
         Value left = funcBody->getArgument(1);
         Value right = funcBody->getArgument(2);

         Value genericMemrefLeft = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedKeyType, llvm::Optional<int64_t>()), left);
         Value genericMemrefRight = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedKeyType, llvm::Optional<int64_t>()), right);
         Value serializedTupleLeft = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedKeyType, genericMemrefLeft, Value());
         Value serializedTupleRight = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedKeyType, genericMemrefRight, Value());
         Value tupleLeft = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleLeft, keyType);
         Value tupleRight = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleRight, keyType);
         Value equal = rewriter.create<mlir::db::ConstantOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext()), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
         auto leftUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), keyTupleType.getTypes(), tupleLeft);
         auto rightUnpacked = rewriter.create<mlir::util::UnPackOp>(rewriter.getUnknownLoc(), keyTupleType.getTypes(), tupleRight);
         for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
            auto compared = rewriter.create<mlir::db::CmpOp>(rewriter.getUnknownLoc(), mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));

            Value localEqual = rewriter.create<mlir::db::AndOp>(rewriter.getUnknownLoc(), mlir::db::BoolType::get(rewriter.getContext(), equal.getType().cast<mlir::db::BoolType>().getNullable() || compared.getType().cast<mlir::db::BoolType>().getNullable()), ValueRange({equal, compared}));
            equal = localEqual;
         }
         //equal.setType(rewriter.getI1Type()); //todo: bad hack ;)
         rewriter.create<mlir::ReturnOp>(createOp->getLoc(), equal);
      }
      FuncOp funcOp;
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         auto byteRangeType = MemRefType::get({-1}, rewriter.getIntegerType(8));
         funcOp = rewriter.create<FuncOp>(parentModule.getLoc(), "db_ht_aggr_builder_update" + std::to_string(id++), rewriter.getFunctionType(TypeRange({byteRangeType, ptrType, ptrType}), TypeRange()));
         funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
         auto* funcBody = new Block;
         funcBody->addArguments(TypeRange({byteRangeType, ptrType, ptrType}));
         funcOp.body().push_back(funcBody);
         rewriter.setInsertionPointToStart(funcBody);
         Value varLenData = funcBody->getArgument(0);
         Value left = funcBody->getArgument(1);
         Value right = funcBody->getArgument(2);

         Value genericMemrefLeft = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedValType, llvm::Optional<int64_t>()), left);
         Value genericMemrefRight = rewriter.create<util::ToGenericMemrefOp>(rewriter.getUnknownLoc(), util::GenericMemrefType::get(rewriter.getContext(), serializedValType, llvm::Optional<int64_t>()), right);
         Value serializedTupleLeft = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedValType, genericMemrefLeft, Value());
         Value serializedTupleRight = rewriter.create<util::LoadOp>(createOp.getLoc(), serializedValType, genericMemrefRight, Value());
         Value tupleLeft = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleLeft, valType);
         Value tupleRight = mlir::db::codegen::SerializationUtil::deserialize(rewriter, varLenData, serializedTupleRight, valType);
         auto terminator = rewriter.create<mlir::ReturnOp>(createOp.getLoc());
         Block* sortLambda = &createOp.region().front();
         auto* sortLambdaTerminator = sortLambda->getTerminator();
         rewriter.mergeBlockBefore(sortLambda, terminator, {tupleLeft, tupleRight});
         mlir::db::YieldOp yieldOp = mlir::cast<mlir::db::YieldOp>(terminator->getPrevNode());
         Value x = yieldOp.results()[0];
         rewriter.setInsertionPoint(terminator);
         //todo: serialize!!
         x.setType(serializedValType); //todo: hacky
         rewriter.create<util::StoreOp>(rewriter.getUnknownLoc(), x, genericMemrefLeft, Value());

         funcOp->dump();
         rewriter.eraseOp(sortLambdaTerminator);
      }
      {
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         llvm::dbgs() << createOp.region().getBlocks().size() << "\n";
         createOp.region().push_back(new Block());
         rewriter.setInsertionPointToStart(&createOp.region().front());
         rewriter.create<mlir::db::YieldOp>(createOp.getLoc());
      }

      Value updateFunctionPointer = rewriter.create<mlir::ConstantOp>(createOp->getLoc(), funcOp.type(), rewriter.getSymbolRefAttr(funcOp.sym_name()));
      Value compareFunctionPointer = rewriter.create<mlir::ConstantOp>(createOp->getLoc(), compareFunc.type(), rewriter.getSymbolRefAttr(compareFunc.sym_name()));
      Value keySize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), serializedKeyType);
      Value valSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), serializedValType);
      Value builder = functionRegistry.call(rewriter, FunctionId::AggrHtBuilderCreate, {keySize, valSize, compareFunctionPointer, updateFunctionPointer})[0];
      rewriter.replaceOp(op, builder);
      return success();
   }
};
class BuilderBuildLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit BuilderBuildLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::BuilderBuild::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::BuilderBuildAdaptor buildAdaptor(operands);
      auto buildOp = cast<mlir::db::BuilderBuild>(op);
      if (auto tableBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         Value table = functionRegistry.call(rewriter, FunctionId::ArrowTableBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, table);
      } else if (auto vectorBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::VectorBuilderType>()) {
         Value vector = functionRegistry.call(rewriter, FunctionId::VectorBuilderBuild, buildAdaptor.builder())[0];

         rewriter.replaceOp(op, vector);
      } else if (auto aggrHTBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::AggrHTBuilderType>()) {
         Value aggrHashtable = functionRegistry.call(rewriter, FunctionId::AggrHtBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, aggrHashtable);
      }

      return success();
   }
};
class HashLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;
   Value xorImpl(OpBuilder& builder, Value v, Value totalHash) const {
      return builder.create<mlir::XOrOp>(builder.getUnknownLoc(), v, totalHash);
   }
   Value hashImpl(OpBuilder& builder, Value v, Value totalHash) const {
      //todo: more checks:
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      if (auto intType = v.getType().dyn_cast_or_null<mlir::IntegerType>()) {
         switch (intType.getWidth()) {
            case 1: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashBool, v)[0]);
            case 8: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt8, v)[0]);
            case 16: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt16, v)[0]);
            case 32: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt32, v)[0]);
            case 64: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt64, v)[0]);
            case 128: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashInt128, v)[0]);
         }
      } else if (auto floatType = v.getType().dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 32: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashFloat32, v)[0]);
            case 64: return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashFloat64, v)[0]);
         }
      } else if (auto memrefType = v.getType().dyn_cast_or_null<mlir::MemRefType>()) {
         return xorImpl(builder, totalHash, functionRegistry.call(builder, FunctionId::HashBinary, v)[0]);
      } else if (auto tupleType = v.getType().dyn_cast_or_null<mlir::TupleType>()) {
         auto unpacked = builder.create<util::UnPackOp>(builder.getUnknownLoc(), tupleType.getTypes(), v);
         for (auto v : unpacked->getResults()) {
            totalHash = hashImpl(builder, v, totalHash);
         }
         return totalHash;
      }
      assert(false && "should not happen");
      return Value();
   }

   public:
   explicit HashLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::Hash::getOperationName(), 1, context), functionRegistry(functionRegistry) {}
   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::HashAdaptor hashAdaptor(operands);
      hashAdaptor.val().getType().dump();
      Value const0 = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
      rewriter.replaceOp(op, hashImpl(rewriter, hashAdaptor.val(), const0));
      return success();
   }
};
class CreateRangeLowering : public ConversionPattern {
   public:
   explicit CreateRangeLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateRange::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = rewriter.getUnknownLoc();
      auto createRangeOp = cast<mlir::db::CreateRange>(op);
      Type storageType = createRangeOp.range().getType().cast<mlir::db::RangeType>().getElementType();
      Value combined = rewriter.create<mlir::util::PackOp>(loc, TypeRange(TupleType::get(getContext(), {storageType, storageType, storageType})), ValueRange({createRangeOp.lower(), createRangeOp.upper(), createRangeOp.step()}));
      rewriter.replaceOp(op, combined);

      return success();
   }
};

static Type convertFunctionType(FunctionType type, TypeConverter& typeConverter) {
   TypeConverter::SignatureConversion result(type.getNumInputs());
   SmallVector<Type, 1> newResults;
   if (failed(typeConverter.convertSignatureArgs(type.getInputs(), result)) ||
       failed(typeConverter.convertTypes(type.getResults(), newResults))) {
      return Type();
   }

   auto newType = FunctionType::get(type.getContext(),
                                    result.getConvertedTypes(), newResults);
   return newType;
}
class FuncConstLowering : public ConversionPattern {
   public:
   explicit FuncConstLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::ConstantOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::ConstantOp constantOp = mlir::cast<mlir::ConstantOp>(op);
      if (auto type = constantOp.getType().dyn_cast_or_null<mlir::FunctionType>()) {
         // Convert the original function types.

         rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, convertFunctionType(type, *typeConverter), constantOp.value());
         return success();

      } else {
         return failure();
      }
   }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// dbToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct DBToStdLoweringPass
   : public PassWrapper<DBToStdLoweringPass, OperationPass<ModuleOp>> {
   DBToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect, util::UtilDialect, memref::MemRefDialect>();
   }
   void runOnOperation() final;
};
static TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      Type converted = typeConverter.convertType(t);
      converted = converted ? converted : t;
      types.push_back(converted);
   }
   return TupleType::get(tupleType.getContext(), TypeRange(types));
}
} // end anonymous namespace
static bool hasDBType(TypeRange types) {
   bool res = false;
   for (Type type : types) {
      if (type.isa<db::DBType>()) {
         res |= true;
      } else if (auto tupleType = type.dyn_cast_or_null<TupleType>()) {
         res |= hasDBType(tupleType.getTypes());
      } else if (auto genericMemrefType = type.dyn_cast_or_null<util::GenericMemrefType>()) {
         res |= hasDBType(genericMemrefType.getElementType());
      } else if (auto functionType = type.dyn_cast_or_null<mlir::FunctionType>()) {
         res |= hasDBType(functionType.getInputs()) ||
            hasDBType(functionType.getResults());
      } else if (type.isa<mlir::db::TableType>() || type.isa<mlir::db::VectorType>()) {
         res = true;
      }
   }
   return res;
}
void DBToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   mlir::db::codegen::FunctionRegistry functionRegistry(&getContext());
   functionRegistry.registerFunctions();

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<StandardOpsDialect>();
   target.addLegalDialect<memref::MemRefDialect>();

   target.addLegalDialect<scf::SCFDialect>();
   target.addLegalDialect<util::UtilDialect>();
   target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto isLegal = !hasDBType(op.getType().getInputs()) &&
         !hasDBType(op.getType().getResults());
      //op->dump();
      //llvm::dbgs() << "isLegal:" << isLegal << "\n";
      return isLegal;
   });
   target.addDynamicallyLegalOp<ConstantOp>([&](ConstantOp op) {
      if (auto functionType = op.getType().dyn_cast_or_null<mlir::FunctionType>()) {
         auto isLegal = !hasDBType(functionType.getInputs()) &&
            !hasDBType(functionType.getResults());
         return isLegal;
      } else {
         return true;
      }
   });
   target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(
      [](Operation* op) {
         auto isLegal = !hasDBType(op->getOperandTypes()) &&
            !hasDBType(op->getResultTypes());
         //op->dump();
         //llvm::dbgs() << "isLegal:" << isLegal << "\n";
         return isLegal;
      });
   target.addDynamicallyLegalOp<util::DimOp, util::SetTupleOp, util::GetTupleOp, util::UndefTupleOp, util::PackOp, util::UnPackOp, util::ToGenericMemrefOp, util::StoreOp, util::LoadOp, util::MemberRefOp, util::FromRawPointerOp, util::ToRawPointerOp, util::AllocOp, util::DeAllocOp, util::AllocaOp, util::AllocaOp>(
      [](Operation* op) {
         auto isLegal = !hasDBType(op->getOperandTypes()) &&
            !hasDBType(op->getResultTypes());

         return isLegal;
      });
   target.addDynamicallyLegalOp<util::SizeOfOp>(
      [](util::SizeOfOp op) {
         auto isLegal = !hasDBType(op.type());
         return isLegal;
      });

   //Add own types to LLVMTypeConverter
   TypeConverter typeConverter;
   typeConverter.addConversion([&](mlir::db::DBType type) {
      Type rawType = ::llvm::TypeSwitch<::mlir::db::DBType, mlir::Type>(type)
                        .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
                           return IntegerType::get(&getContext(), 1);
                        })
                        .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
                           if (t.getUnit() == mlir::db::DateUnitAttr::day) {
                              return IntegerType::get(&getContext(), 32);
                           } else {
                              return IntegerType::get(&getContext(), 64);
                           }
                        })
                        .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
                           if (t.getUnit() == mlir::db::TimeUnitAttr::second && t.getUnit() == mlir::db::TimeUnitAttr::millisecond) {
                              return IntegerType::get(&getContext(), 32);
                           } else {
                              return IntegerType::get(&getContext(), 64);
                           }
                        })
                        .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                           return IntegerType::get(&getContext(), 128);
                        })
                        .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
                           return IntegerType::get(&getContext(), t.getWidth());
                        })
                        .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
                           return IntegerType::get(&getContext(), t.getWidth());
                        })
                        .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
                           Type res;
                           if (t.getWidth() == 32) {
                              res = FloatType::getF32(&getContext());
                           } else if (t.getWidth() == 64) {
                              res = FloatType::getF64(&getContext());
                           }
                           return res;
                        })
                        .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
                           return MemRefType::get({-1}, IntegerType::get(&getContext(), 8));
                        })
                        .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
                           return IntegerType::get(&getContext(), 64);
                        })
                        .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
                           return IntegerType::get(&getContext(), 64);
                        })
                        .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
                           if (t.getUnit() == mlir::db::IntervalUnitAttr::daytime) {
                              return IntegerType::get(&getContext(), 64);
                           } else {
                              return IntegerType::get(&getContext(), 32);
                           }
                        })
                        .Default([](::mlir::Type) { return Type(); });
      if (type.isNullable()) {
         return (Type) TupleType::get(&getContext(), {IntegerType::get(&getContext(), 1), rawType});
      } else {
         return rawType;
      }
   });
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::FunctionType functionType) {
      return convertFunctionType(functionType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::FunctionType functionType) {
      return convertFunctionType(functionType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::db::TableType tableType) {
      return MemRefType::get({}, IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](mlir::db::TableBuilderType tableType) {
      return MemRefType::get({}, IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](mlir::db::VectorBuilderType vectorBuilderType) {
      return MemRefType::get({}, IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](mlir::db::AggrHTBuilderType aggrHtBuilderType) {
      auto ptrType = MemRefType::get({}, IntegerType::get(&getContext(), 8));
      return ptrType;
   });
   typeConverter.addConversion([&](mlir::db::AggregationHashtableType aggregationHashtableType) {
      auto ptrType = MemRefType::get({}, IntegerType::get(&getContext(), 8));
      return ptrType;
   });
   typeConverter.addConversion([&](mlir::db::VectorType vectorType) {
      auto ptrType = MemRefType::get({-1}, IntegerType::get(&getContext(), 8));
      return TupleType::get(&getContext(), {ptrType, ptrType});
   });
   typeConverter.addConversion([&](mlir::db::RangeType rangeType) {
      auto convertedType = typeConverter.convertType(rangeType.getElementType());
      return TupleType::get(&getContext(), {convertedType, convertedType, convertedType});
   });
   typeConverter.addConversion([&](mlir::db::GenericIterableType genericIterableType) {
      Type elementType = genericIterableType.getElementType();
      Type nestedElementType = elementType;
      if (auto nested = elementType.dyn_cast_or_null<mlir::db::GenericIterableType>()) {
         nestedElementType = nested.getElementType();
      }
      if (genericIterableType.getIteratorName() == "table_chunk_iterator") {
         std::vector<Type> types;
         auto i8Type = IntegerType::get(&getContext(), 8);
         auto ptrType = MemRefType::get({}, i8Type);
         auto indexType = IndexType::get(&getContext());
         types.push_back(ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) TupleType::get(&getContext(), types);
      } else if (genericIterableType.getIteratorName() == "table_row_iterator") {
         std::vector<Type> types;
         auto i8Type = IntegerType::get(&getContext(), 8);
         auto ptrType = MemRefType::get({}, i8Type);
         auto indexType = IndexType::get(&getContext());
         types.push_back(ptrType);
         if (auto tupleT = nestedElementType.dyn_cast_or_null<TupleType>()) {
            for (size_t i = 0; i < tupleT.getTypes().size(); i++) {
               types.push_back(indexType);
            }
         }
         return (Type) TupleType::get(&getContext(), types);
      }
      return Type();
   });
   typeConverter.addConversion([&](mlir::IntegerType iType) { return iType; });
   typeConverter.addConversion([&](mlir::IndexType iType) { return iType; });
   typeConverter.addConversion([&](mlir::FloatType fType) { return fType; });
   typeConverter.addConversion([&](mlir::MemRefType refType) { return refType; });

   typeConverter.addSourceMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::DBType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, IntegerType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, IntegerType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::GenericIterableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::TableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::TableBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::VectorBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::VectorBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::VectorType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::VectorType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, MemRefType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, MemRefType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, TupleType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, TupleType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::AggregationHashtableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::AggregationHashtableType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::AggrHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, db::AggrHTBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });

   OwningRewritePatternList patterns(&getContext());
   /*patterns.add<FunctionLikeSignatureConversion>(&getContext(), typeConverter);
   patterns.add<ForwardOperands<CallOp>,
                ForwardOperands<CallIndirectOp>,
                ForwardOperands<ReturnOp>>(typeConverter, &getContext());*/
   mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   mlir::db::populateScalarToStdPatterns(typeConverter, patterns);
   mlir::db::populateControlFlowToStdPatterns(typeConverter, patterns);
   mlir::db::populateRuntimeSpecificScalarToStdPatterns(functionRegistry, typeConverter, patterns);

   mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   // Add own Lowering Patterns

   patterns.insert<FuncConstLowering>(typeConverter, &getContext());


   patterns.insert<TableScanLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateTableBuilderLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateVectorBuilderLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateAggrHTBuilderLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<BuilderMergeLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<BuilderBuildLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<HashLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<GetTableLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<SortOpLowering>(functionRegistry, typeConverter, &getContext());

   patterns.insert<ForOpLowering>(functionRegistry, typeConverter, &getContext());
   patterns.insert<CreateRangeLowering>(functionRegistry, typeConverter, &getContext());

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
