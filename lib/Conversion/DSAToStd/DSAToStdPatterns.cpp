#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "runtime-defs/ArrowColumn.h"
#include "runtime-defs/ArrowTable.h"
#include "runtime-defs/Hashtable.h"
#include "runtime-defs/LazyJoinHashtable.h"
using namespace mlir;
namespace {
/*mlir::Value getExecutionContext(ConversionPatternRewriter& rewriter, mlir::Operation* op) {
   auto parentModule = op->getParentOfType<ModuleOp>();
   mlir::func::FuncOp funcOp = parentModule.lookupSymbol<mlir::func::FuncOp>("rt_get_execution_context");
   if (!funcOp) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      funcOp = rewriter.create<mlir::func::FuncOp>(op->getLoc(), "rt_get_execution_context", mlir::FunctionType::get(op->getContext(), {}, {mlir::util::RefType::get(op->getContext(), rewriter.getI8Type())}), rewriter.getStringAttr("private"), ArrayAttr{}, ArrayAttr{});
   }
   mlir::Value executionContext = rewriter.create<mlir::func::CallOp>(op->getLoc(), funcOp, mlir::ValueRange{}).getResult(0);
   return executionContext;
}*/

class CBAppendLowering : public OpConversionPattern<mlir::dsa::Append> {
   public:
   using OpConversionPattern<mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!appendOp.getDs().getType().isa<mlir::dsa::ColumnBuilderType>()) {
         return failure();
      }
      mlir::Type arrowType = appendOp.getVal().getType();
      Value builderVal = adaptor.getDs();
      Value val = adaptor.getVal();
      Value isValid = adaptor.getValid();
      auto loc = appendOp->getLoc();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      mlir::Type typeX = getBaseType(val.getType());
      if (arrowType.isIndex()) {
         rt::ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val});
      } else if (isIntegerType(arrowType, 1)) {
         rt::ArrowColumnBuilder::addBool(rewriter, loc)({builderVal, isValid, val});
      } else if (auto intType = arrowType.dyn_cast_or_null<IntegerType>()) {
         switch (intType.getWidth()) {
            case 8: rt::ArrowColumnBuilder::addInt8(rewriter, loc)({builderVal, isValid, val}); break;
            case 16: rt::ArrowColumnBuilder::addInt16(rewriter, loc)({builderVal, isValid, val}); break;
            case 32: rt::ArrowColumnBuilder::addInt32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val}); break;
            default: assert(false && "should not happen");
         }
      } else if (auto floatType = arrowType.dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 32: rt::ArrowColumnBuilder::addFloat32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::ArrowColumnBuilder::addFloat64(rewriter, loc)({builderVal, isValid, val}); break;
         }
      } else if (arrowType.isa<mlir::dsa::ArrowStringType>()) {
         rt::ArrowColumnBuilder::addBinary(rewriter, loc)({builderVal, isValid, val});
      } else if (auto fixedWidthType = arrowType.dyn_cast_or_null<mlir::dsa::ArrowFixedSizedBinaryType>()) {
         rt::ArrowColumnBuilder::addFixedSized(rewriter, loc)({builderVal, isValid, val});
      } else if (arrowType.isa<mlir::dsa::ArrowDecimalType>()) {
         rt::ArrowColumnBuilder::addDecimal(rewriter, loc)({builderVal, isValid, val});
      } else if (arrowType.isa<mlir::dsa::ArrowTimeStampType, mlir::dsa::ArrowDate64Type, mlir::dsa::ArrowDayTimeIntervalType>()) {
         rt::ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val});
      } else if (arrowType.isa<mlir::dsa::ArrowDate32Type, mlir::dsa::ArrowMonthIntervalType>()) {
         rt::ArrowColumnBuilder::addInt32(rewriter, loc)({builderVal, isValid, val});
      } else {
         return failure();
      }
      rewriter.eraseOp(appendOp);
      return success();
   }
};

class CreateColumnBuilderLowering : public OpConversionPattern<mlir::dsa::CreateDS> {
   std::string arrowDescrFromType(mlir::Type type) const {
      if (type.isIndex()) {
         return "int[64]";
      } else if (isIntegerType(type, 1)) {
         return "bool";
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         return "int[" + std::to_string(intWidth) + "]";
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         return "uint[" + std::to_string(uIntWidth) + "]";
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::dsa::ArrowDecimalType>()) {
         auto prec = std::min(decimalType.getP(), (int64_t) 38);
         return "decimal[" + std::to_string(prec) + "," + std::to_string(decimalType.getS()) + "]";
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         return "float[" + std::to_string(floatType.getWidth()) + "]";
      } else if (type.isa<mlir::dsa::ArrowStringType>()) { //todo: do we still need the strings?
         return "string";
      } else if (type.isa<mlir::dsa::ArrowDate32Type>()) {
         return "date[32]";
      } else if (type.isa<mlir::dsa::ArrowDate64Type>()) {
         return "date[64]";
      } else if (auto fixedSizedBinaryType = type.dyn_cast_or_null<mlir::dsa::ArrowFixedSizedBinaryType>()) {
         return "fixed_sized[" + std::to_string(fixedSizedBinaryType.getByteWidth()) + "]";
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::dsa::ArrowMonthIntervalType>()) {
         return "interval_months";
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::dsa::ArrowDayTimeIntervalType>()) {
         return "interval_daytime";
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::dsa::ArrowTimeStampType>()) {
         return "timestamp[" + std::to_string(static_cast<uint32_t>(timestampType.getUnit())) + "]";
      }
      assert(false);
      return "";
   }

   public:
   using OpConversionPattern<mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getDs().getType().isa<mlir::dsa::ColumnBuilderType>()) {
         return failure();
      }
      auto loc = createOp->getLoc();
      mlir::Value typeDescr = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(getContext()), arrowDescrFromType(createOp.getType().cast<mlir::dsa::ColumnBuilderType>().getType()));
      Value columnBuilder = rt::ArrowColumnBuilder::create(rewriter, loc)({typeDescr})[0];
      rewriter.replaceOp(createOp, columnBuilder);
      return success();
   }
};
class CreateTableLowering : public OpConversionPattern<mlir::dsa::CreateTable> {
   public:
   using OpConversionPattern<mlir::dsa::CreateTable>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::CreateTable createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = createOp->getLoc();
      mlir::Value table = rt::ArrowTable::createEmpty(rewriter, loc)({})[0];
      for (auto x : llvm::zip(createOp.getColumnNames(), adaptor.getColumns())) {
         auto name = std::get<0>(x).cast<StringAttr>().getValue();
         auto column = std::get<1>(x);
         mlir::Value columnName = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(getContext()), name);
         table = rt::ArrowTable::addColumn(rewriter, loc)({table, columnName, column})[0];
      }
      rewriter.replaceOp(createOp, table);
      return success();
   }
};

class ColumnnBuilderConcat : public OpConversionPattern<mlir::dsa::Concat> {
   public:
   using OpConversionPattern<mlir::dsa::Concat>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Concat op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!op.getLeft().getType().isa<mlir::dsa::ColumnBuilderType>() || !op.getLeft().getType().isa<mlir::dsa::ColumnBuilderType>()) {
         return failure();
      }
      rt::ArrowColumnBuilder::merge(rewriter, op->getLoc())({adaptor.getLeft(), adaptor.getRight()});
      rewriter.replaceOp(op, adaptor.getLeft());
      return success();
   }
};
class ColumnnBuilderFinish : public OpConversionPattern<mlir::dsa::FinishColumn> {
   public:
   using OpConversionPattern<mlir::dsa::FinishColumn>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::FinishColumn op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!op.getBuilder().getType().isa<mlir::dsa::ColumnBuilderType>()) {
         return failure();
      }
      auto column = rt::ArrowColumnBuilder::finish(rewriter, op->getLoc())({
         adaptor.getBuilder(),
      })[0];
      rewriter.replaceOp(op, column);
      return success();
   }
};
class ArrowTypeToLowering : public OpConversionPattern<mlir::dsa::ArrowTypeTo> {
   public:
   using OpConversionPattern<mlir::dsa::ArrowTypeTo>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::ArrowTypeTo op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto t = op.getType();
      auto loc = op.getLoc();
      mlir::Value physicalVal = adaptor.getArrowVal();
      auto physicalType = physicalVal.getType();
      auto arrowType = op.getArrowVal().getType();
      if (arrowType.isa<dsa::ArrowDate32Type, dsa::ArrowDate64Type, dsa::ArrowTimeStampType>()) {
         if (physicalType.getIntOrFloatBitWidth() < 64) {
            physicalVal = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), physicalVal);
         }
         size_t multiplier = 1;
         if (arrowType.isa<dsa::ArrowDate32Type>()) {
            multiplier = 86400000000000;
         } else if (arrowType.isa<dsa::ArrowDate64Type>()) {
            multiplier = 1000000;
         } else if (auto timeStampType = arrowType.dyn_cast_or_null<dsa::ArrowTimeStampType>()) {
            switch (timeStampType.getUnit()) {
               case mlir::dsa::TimeUnitAttr::second: multiplier = 1000000000; break;
               case mlir::dsa::TimeUnitAttr::millisecond: multiplier = 1000000; break;
               case mlir::dsa::TimeUnitAttr::microsecond: multiplier = 1000; break;
               default: multiplier = 1;
            }
         }
         if (multiplier != 1) {
            mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, multiplier, 64);
            physicalVal = rewriter.create<mlir::arith::MulIOp>(loc, physicalVal, multiplierConst);
         }
         rewriter.replaceOp(op, physicalVal);
         return success();
      } else if (physicalType == t) {
         rewriter.replaceOp(op, adaptor.getArrowVal());
         return success();
      } else if (auto decimalType = arrowType.dyn_cast_or_null<dsa::ArrowDecimalType>()) {
         if (t.getIntOrFloatBitWidth() != 128) {
            rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, op.getType(), adaptor.getArrowVal());
            return success();
         }
      }
      return failure();
   }
};
class ArrowTypeFromLowering : public OpConversionPattern<mlir::dsa::ArrowTypeFrom> {
   public:
   using OpConversionPattern<mlir::dsa::ArrowTypeFrom>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::ArrowTypeFrom op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto arrowType = op.getType();
      auto loc = op.getLoc();
      auto physicalType = typeConverter->convertType(arrowType);
      auto inputType = op.getVal().getType();
      mlir::Value physicalVal = adaptor.getVal();
      if (arrowType.isa<dsa::ArrowDate32Type, dsa::ArrowDate64Type, dsa::ArrowTimeStampType>()) {
         size_t multiplier = 1;
         if (arrowType.isa<dsa::ArrowDate32Type>()) {
            multiplier = 86400000000000;
         } else if (arrowType.isa<dsa::ArrowDate64Type>()) {
            multiplier = 1000000;
         } else if (auto timeStampType = arrowType.dyn_cast_or_null<dsa::ArrowTimeStampType>()) {
            switch (timeStampType.getUnit()) {
               case mlir::dsa::TimeUnitAttr::second: multiplier = 1000000000; break;
               case mlir::dsa::TimeUnitAttr::millisecond: multiplier = 1000000; break;
               case mlir::dsa::TimeUnitAttr::microsecond: multiplier = 1000; break;
               default: multiplier = 1;
            }
         }
         if (multiplier != 1) {
            mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, multiplier, 64);
            physicalVal = rewriter.create<mlir::arith::DivSIOp>(loc, physicalVal, multiplierConst);
         }
         if (physicalType != rewriter.getI64Type()) {
            physicalVal = rewriter.create<mlir::arith::TruncIOp>(loc, physicalType, physicalVal);
         }
         rewriter.replaceOp(op, physicalVal);
         return success();
      } else if (physicalType == inputType) {
         rewriter.replaceOp(op, physicalVal);
         return success();
      } else if (auto decimalType = arrowType.dyn_cast_or_null<dsa::ArrowDecimalType>()) {
         if (inputType.getIntOrFloatBitWidth() != 128) {
            rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, physicalType, physicalVal);
            return success();
         }
      }
      return failure();
   }
};
} // end namespace
namespace mlir::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<ColumnnBuilderConcat, CBAppendLowering, ColumnnBuilderFinish, CreateColumnBuilderLowering, CreateTableLowering, ArrowTypeToLowering, ArrowTypeFromLowering>(typeConverter, patterns.getContext());
}
} // end namespace mlir::dsa
