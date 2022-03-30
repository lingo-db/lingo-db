#include "arrow/type_fwd.h"

#include "mlir/Conversion/DBToArrowStd/ArrowTypes.h"
#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
/*namespace {

static db::codegen::FunctionRegistry::FunctionId getStoreFunc(db::codegen::FunctionRegistry& functionRegistry, Type type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;
   if (isIntegerType(type, 1)) {
      return FunctionId::ArrowTableBuilderAddBool;
   } else if (auto intWidth = getIntegerWidth(type, false)) {
      switch (intWidth) {
         case 8: return FunctionId::ArrowTableBuilderAddInt8;
         case 16: return FunctionId::ArrowTableBuilderAddInt16;
         case 32: return FunctionId::ArrowTableBuilderAddInt32;
         case 64: return FunctionId::ArrowTableBuilderAddInt64;
      }
   } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      if (decimalType.getP() < 19) {
         return FunctionId::ArrowTableBuilderAddSmallDecimal;
      }
      return FunctionId::ArrowTableBuilderAddDecimal;
   } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
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
   } else if (type.isa<mlir::db::CharType>()) {
      return FunctionId ::ArrowTableBuilderAddFixedBinary;
   }
   //TODO: implement other types too
   return FunctionId::ArrowTableBuilderAddInt32;
}
class AddTableRowLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit AddTableRowLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::AddTableRow::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::AddTableRowAdaptor mergeOpAdaptor(operands);
      auto mergeOp = cast<mlir::db::AddTableRow>(op);
      auto loc = op->getLoc();
      if (auto tableBuilderType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         TupleType rowType = mergeOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
         auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, mergeOpAdaptor.row());
         size_t i = 0;
         Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

         for (auto v : unPackOp.vals()) {
            Value val = v;
            Value isNull;
            if (mergeOp.row().getType().cast<TupleType>().getType(i).isa<mlir::db::NullableType>()) {
               auto nullUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, val);
               isNull = nullUnpacked.getResult(0);
               val = nullUnpacked->getResult(1);
            } else {
               isNull = falseValue;
            }
            if (auto charType = mergeOp.row().getType().cast<TupleType>().getType(i).dyn_cast_or_null<mlir::db::CharType>()) {
               if (charType.getBytes() < 8) {
                  val = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), val);
               }
            }
            Value columnId = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(i));
            functionRegistry.call(rewriter, loc, getStoreFunc(functionRegistry, getBaseType(rowType.getType(i))), ValueRange({mergeOpAdaptor.builder(), columnId, isNull, val}));
            i++;
         }
         functionRegistry.call(rewriter, loc, FunctionId::ArrowTableBuilderFinishRow, mergeOpAdaptor.builder());
         rewriter.eraseOp(op);
      }
      return success();
   }
};


static Value getArrowDataType(OpBuilder& builder, Location loc, db::codegen::FunctionRegistry& functionRegistry, Type type) {
   using FunctionId = db::codegen::FunctionRegistry::FunctionId;

   auto [typeConstant, param1, param2] = db::codegen::convertTypeToArrow(type);
   //TODO: also implement date types etc

   Value arrowTypeConstant = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(typeConstant));
   Value arrowTypeParam1 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param1));
   Value arrowTypeParam2 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(param2));

   Value arrowType = functionRegistry.call(builder, loc, FunctionId::ArrowGetType2Param, ValueRange({arrowTypeConstant, arrowTypeParam1, arrowTypeParam2}))[0];
   return arrowType;
}
class FinalizeTableLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit FinalizeTableLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::FinalizeTable::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      mlir::db::FinalizeTableAdaptor buildAdaptor(operands);
      auto buildOp = cast<mlir::db::FinalizeTable>(op);
      if (auto tableBuilderType = buildOp.builder().getType().dyn_cast<mlir::db::TableBuilderType>()) {
         Value table = functionRegistry.call(rewriter, op->getLoc(), FunctionId::ArrowTableBuilderBuild, buildAdaptor.builder())[0];
         rewriter.replaceOp(op, table);
      }
      return success();
   }
};
class CreateTableBuilderLowering : public ConversionPattern {
   db::codegen::FunctionRegistry& functionRegistry;

   public:
   explicit CreateTableBuilderLowering(db::codegen::FunctionRegistry& functionRegistry, TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::CreateTableBuilder::getOperationName(), 1, context), functionRegistry(functionRegistry) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      using FunctionId = db::codegen::FunctionRegistry::FunctionId;
      auto createTB = cast<mlir::db::CreateTableBuilder>(op);
      auto loc = op->getLoc();

      Value schema = functionRegistry.call(rewriter, loc, FunctionId::ArrowTableSchemaCreate, {})[0];
      TupleType rowType = createTB.builder().getType().dyn_cast<mlir::db::TableBuilderType>().getRowType();
      size_t i = 0;
      for (auto c : createTB.columns()) {
         auto stringAttr = c.cast<StringAttr>();
         auto isNullable = rowType.getType(i).isa<mlir::db::NullableType>();
         auto arrowType = getArrowDataType(rewriter, op->getLoc(), functionRegistry, getBaseType(rowType.getType(i)));
         auto columnName = rewriter.create<mlir::util::CreateConstVarLen>(op->getLoc(), mlir::util::VarLen32Type::get(rewriter.getContext()), stringAttr);
         Value typeNullable = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), isNullable));

         functionRegistry.call(rewriter, loc, FunctionId::ArrowTableSchemaAddField, ValueRange({schema, arrowType, typeNullable, columnName}));
         i += 1;
      }
      schema = functionRegistry.call(rewriter, loc, FunctionId::ArrowTableSchemaBuild, schema)[0];
      Value tableBuilder = functionRegistry.call(rewriter, loc, FunctionId::ArrowTableBuilderCreate, schema)[0];
      rewriter.replaceOp(op, tableBuilder);
      return success();
   }
};

} // namespace
 */
void mlir::db::populateBuilderToStdPatterns(mlir::db::codegen::FunctionRegistry& functionRegistry, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   /*patterns.insert<CreateTableBuilderLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<AddTableRowLowering>(functionRegistry, typeConverter, patterns.getContext());
   patterns.insert<FinalizeTableLowering>(functionRegistry, typeConverter, patterns.getContext());
   typeConverter.addSourceMaterialization([&](OpBuilder&, db::TableBuilderType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addConversion([&](mlir::db::TableBuilderType tableType) {
      return mlir::util::RefType::get(patterns.getContext(), IntegerType::get(patterns.getContext(), 8));
   });*/
}
