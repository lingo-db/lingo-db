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
mlir::Value arrowTypeFrom(mlir::OpBuilder rewriter, mlir::Location loc, mlir::Type arrowType, mlir::Type physicalType, mlir::Type inputType, mlir::Value physicalVal) {
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
      return physicalVal;
   } else if (physicalType == inputType) {
      return physicalVal;
   } else if (auto decimalType = arrowType.dyn_cast_or_null<dsa::ArrowDecimalType>()) {
      if (inputType.getIntOrFloatBitWidth() != 128) {
         return rewriter.create<arith::ExtSIOp>(loc, physicalType, physicalVal);
      }
   }
   return mlir::Value();
}
mlir::Value arrowTypeTo(mlir::OpBuilder rewriter, mlir::Location loc, mlir::Type arrowType, mlir::Type physicalType, mlir::Type t, mlir::Value physicalVal) {
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
      return physicalVal;
   } else if (physicalType == t) {
      return physicalVal;
   } else if (auto decimalType = arrowType.dyn_cast_or_null<dsa::ArrowDecimalType>()) {
      if (t.getIntOrFloatBitWidth() != 128) {
         return rewriter.create<arith::TruncIOp>(loc, t, physicalVal);
      }
   }
   return mlir::Value();
}
class CBAppendLowering : public OpConversionPattern<mlir::dsa::Append> {
   mlir::LogicalResult append(ConversionPatternRewriter& rewriter, Location loc, mlir::Type arrowType, Value builderVal, Value isValid, mlir::Value val) const {
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
         return mlir::failure();
      }
      return mlir::success();
   }

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
      if (auto arrowListType = arrowType.dyn_cast_or_null<mlir::dsa::ArrowListType>()) {
         rt::ArrowColumnBuilder::addList(rewriter, loc)({builderVal, isValid});
         auto childBuilder = rt::ArrowColumnBuilder::getChildBuilder(rewriter, loc)({builderVal})[0];
         auto forOp = rewriter.create<mlir::dsa::ForOp>(loc, mlir::TypeRange{}, adaptor.getVal(), mlir::ValueRange{});
         mlir::Block* block = new mlir::Block;
         block->addArgument(adaptor.getVal().getType().cast<mlir::util::BufferType>().getElementType(), loc);
         forOp.getBodyRegion().push_back(block);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(block);
            mlir::Value loaded = rewriter.create<mlir::util::LoadOp>(loc, forOp.getInductionVar());
            //mlir::Value arrowValue = rewriter.create<mlir::dsa::ArrowTypeFrom>(loc, arrowListType.getType(), loaded);
            auto arrowType = arrowListType.getType();
            auto physicalType = typeConverter->convertType(arrowType);
            auto inputType = loaded.getType();
            mlir::Value arrowValue = arrowTypeFrom(rewriter, loc, arrowType, physicalType, inputType, loaded);
            if (append(rewriter, loc, arrowListType.getType(), childBuilder, isValid, arrowValue).failed()) {
               return failure();
            }
            rewriter.create<mlir::dsa::YieldOp>(loc);
         }
      } else {
         if (append(rewriter, loc, arrowType, builderVal, isValid, val).failed()) {
            return failure();
         }
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
      } else if (auto listType = type.dyn_cast_or_null<mlir::dsa::ArrowListType>()) {
         return "list[" + arrowDescrFromType(listType.getType()) + "]";
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

      mlir::Value res = arrowTypeTo(rewriter, loc, arrowType, physicalType, t, physicalVal);
      if (res) {
         rewriter.replaceOp(op, res);
         return success();
      } else {
         return failure();
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
      mlir::Value res = arrowTypeFrom(rewriter, loc, arrowType, physicalType, inputType, physicalVal);
      if (res) {
         rewriter.replaceOp(op, res);
         return success();
      } else {
         return failure();
      }
   }
};

std::vector<mlir::Type> getColumnInfoTypes(mlir::Type t, mlir::Type converted) {
   std::vector<mlir::Type> types;
   auto *context = t.getContext();
   auto indexType = IndexType::get(context);
   auto i8ptrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));
   mlir::Type valueType = converted;
   if (t.isa<mlir::dsa::ArrowStringType,mlir::dsa::ArrowListType>()) {
      valueType = mlir::IntegerType::get(context, 32);
   } else if (t == mlir::IntegerType::get(context, 1)) {
      valueType = mlir::IntegerType::get(context, 8);
   }

   types.push_back(indexType);
   types.push_back(indexType);
   types.push_back(i8ptrType);
   types.push_back(mlir::util::RefType::get(context, valueType));
   types.push_back(i8ptrType);
   types.push_back(i8ptrType);
   return types;
}
class AtLowering : public OpConversionPattern<mlir::dsa::At> {
   public:
   using OpConversionPattern<mlir::dsa::At>::OpConversionPattern;
   static Value getBit(OpBuilder builder, Location loc, Value bits, Value pos) {
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto indexType = IndexType::get(builder.getContext());
      Value const3 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 3));
      Value const7 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 7));
      Value const1Byte = builder.create<arith::ConstantOp>(loc, i8Type, builder.getIntegerAttr(i8Type, 1));

      Value div8 = builder.create<arith::ShRUIOp>(loc, indexType, pos, const3);
      Value rem8 = builder.create<arith::AndIOp>(loc, indexType, pos, const7);
      Value loadedByte = builder.create<mlir::util::LoadOp>(loc, i8Type, bits, div8);
      Value rem8AsByte = builder.create<arith::IndexCastOp>(loc, i8Type, rem8);
      Value shifted = builder.create<arith::ShRUIOp>(loc, i8Type, loadedByte, rem8AsByte);
      Value res1 = shifted;

      Value anded = builder.create<arith::AndIOp>(loc, i8Type, res1, const1Byte);
      Value res = builder.create<arith::CmpIOp>(loc, i1Type, mlir::arith::CmpIPredicate::eq, anded, const1Byte);
      return res;
   }
   Value loadValue(OpBuilder rewriter, mlir::Location loc, mlir::Type baseType, mlir::Value originalValueBuffer,
                   mlir::Value valueBuffer,
                   mlir::Value validityBuffer,
                   mlir::Value varLenBuffer,
                   mlir::Value childPtr,
                   mlir::Value nullMultiplier, mlir::Value columnOffset, mlir::Value index) const {
      auto* context = rewriter.getContext();
      auto indexType = rewriter.getIndexType();
      Value val;
      if (baseType.isa<mlir::dsa::ArrowStringType>()) {
         Value pos1 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, index);
         pos1.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value const1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         Value ip1 = rewriter.create<arith::AddIOp>(loc, indexType, index, const1);
         Value pos2 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, ip1);
         pos2.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value len = rewriter.create<arith::SubIOp>(loc, rewriter.getI32Type(), pos2, pos1);
         Value pos1AsIndex = rewriter.create<arith::IndexCastOp>(loc, indexType, pos1);
         Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), varLenBuffer, pos1AsIndex);
         val = rewriter.create<mlir::util::CreateVarLen>(loc, mlir::util::VarLen32Type::get(rewriter.getContext()), ptr, len);
      } else if (isIntegerType(baseType, 1)) {
         Value realPos = rewriter.create<arith::AddIOp>(loc, indexType, columnOffset, index);
         val = getBit(rewriter, loc, originalValueBuffer, realPos);
      } else if (baseType.isIntOrIndexOrFloat()) {
         //for integers and floats: just load the value
         val = rewriter.create<util::LoadOp>(loc, baseType, valueBuffer, index);
         val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
      } else if (baseType.isa<dsa::ArrowDate32Type, dsa::ArrowDate64Type, dsa::ArrowMonthIntervalType, dsa::ArrowDayTimeIntervalType, dsa::ArrowTimeStampType, dsa::ArrowDecimalType>()) {
         //dates, timestamps, etc are also just integers
         val = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(baseType), valueBuffer, index);
         val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
      } else if (auto fixedSizeType = baseType.dyn_cast<mlir::dsa::ArrowFixedSizedBinaryType>()) {
         auto convertedType = typeConverter->convertType(baseType);
         auto numBytes = fixedSizeType.getByteWidth();
         auto bits = numBytes * 8;
         if (bits == convertedType.getIntOrFloatBitWidth()) {
            //simple case: matches length of
            val = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(baseType), valueBuffer, index);
            val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         } else {
            Value factor = rewriter.create<mlir::arith::ConstantIndexOp>(loc, numBytes);
            Value pos = rewriter.create<arith::AddIOp>(loc, columnOffset, index);
            pos = rewriter.create<arith::MulIOp>(loc, pos, factor);
            Value valBuffer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI8Type()), originalValueBuffer);
            Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos);
            auto combine = [loc](mlir::IntegerType largerType, mlir::Value first, mlir::Value second, size_t shiftAmount, mlir::OpBuilder& rewriter) -> mlir::Value {
               mlir::Value ext1 = rewriter.create<arith::ExtUIOp>(loc, largerType, first);
               mlir::Value ext2 = rewriter.create<arith::ExtUIOp>(loc, largerType, second);
               mlir::Value shiftAmountConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, shiftAmount, largerType);
               mlir::Value shifted = rewriter.create<mlir::arith::ShLIOp>(loc, ext2, shiftAmountConst);
               return rewriter.create<mlir::arith::OrIOp>(loc, ext1, shifted);
            };
            if (bits == 24) {
               mlir::Value i16Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI16Type()), ptr);
               mlir::Value i16Val = rewriter.create<util::LoadOp>(loc, i16Ptr);
               Value const2 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 2);
               Value pos2 = rewriter.create<arith::AddIOp>(loc, pos, const2);
               Value i8Ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos2);
               mlir::Value i8Val = rewriter.create<util::LoadOp>(loc, i8Ptr);
               val = combine(rewriter.getI32Type(), i16Val, i8Val, 16, rewriter);

               i16Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
               i8Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());

            } else if (bits == 40) {
               mlir::Value i32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr);
               mlir::Value i32Val = rewriter.create<util::LoadOp>(loc, i32Ptr);
               Value const4 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 4);
               Value pos2 = rewriter.create<arith::AddIOp>(loc, pos, const4);
               Value i8Ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos2);
               mlir::Value i8Val = rewriter.create<util::LoadOp>(loc, i8Ptr);
               val = combine(rewriter.getI64Type(), i32Val, i8Val, 32, rewriter);

               i32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
               i8Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
            } else if (bits == 48) {
               mlir::Value i32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr);
               mlir::Value i32Val = rewriter.create<util::LoadOp>(loc, i32Ptr);
               Value const4 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 4);
               Value pos2 = rewriter.create<arith::AddIOp>(loc, pos, const4);
               Value i8Ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos2);
               mlir::Value i16Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI16Type()), i8Ptr);

               mlir::Value i16Val = rewriter.create<util::LoadOp>(loc, i16Ptr);
               val = combine(rewriter.getI64Type(), i32Val, i16Val, 32, rewriter);

               i32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
               i16Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
            } else if (bits == 56) {
               mlir::Value i32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr);
               mlir::Value i32Val = rewriter.create<util::LoadOp>(loc, i32Ptr);
               Value const3 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 3);
               Value const8i32 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 8, 32);

               Value pos3 = rewriter.create<arith::AddIOp>(loc, pos, const3);
               Value ptr3 = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), valBuffer, pos3);
               mlir::Value secondI32Ptr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, rewriter.getI32Type()), ptr3);

               mlir::Value secondI32Val = rewriter.create<util::LoadOp>(loc, secondI32Ptr);
               mlir::Value noDuplicate = rewriter.create<mlir::arith::ShRUIOp>(loc, secondI32Val, const8i32);
               val = combine(rewriter.getI64Type(), i32Val, noDuplicate, 32, rewriter);

               i32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
               secondI32Val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
            }
         }
      } else if (auto listType = baseType.dyn_cast_or_null<mlir::dsa::ArrowListType>()) {
         auto bufferType = mlir::util::BufferType::get(context, typeConverter->convertType(listType.getType()));
         auto columnInfoType = rewriter.getTupleType(getColumnInfoTypes(listType.getType(), typeConverter->convertType(listType.getType())));
         auto castedRef = rewriter.create<mlir::util::GenericMemrefCastOp>(loc, mlir::util::RefType::get(columnInfoType), childPtr);
         auto loaded = rewriter.create<mlir::util::LoadOp>(loc, castedRef);
         auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, loaded);
         mlir::Value childColumnOffset = unpacked.getResult(0);
         mlir::Value childNullMultiplier = unpacked.getResult(1);
         mlir::Value childValidityBuffer = unpacked.getResult(2);
         mlir::Value childOriginalValueBuffer = unpacked.getResult(3);
         mlir::Value childValueBuffer = rewriter.create<mlir::util::ArrayElementPtrOp>(loc, childOriginalValueBuffer.getType(), childOriginalValueBuffer, childColumnOffset); // pointer to the column

         mlir::Value childVarLenBuffer = unpacked.getResult(4);
         mlir::Value childChildPtr = unpacked.getResult(5);
         Value pos1 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, index);
         Value const1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         Value ip1 = rewriter.create<arith::AddIOp>(loc, indexType, index, const1);
         Value pos2 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, ip1);
         auto start = rewriter.create<arith::IndexCastOp>(loc, indexType, pos1);
         auto end = rewriter.create<arith::IndexCastOp>(loc, indexType, pos2);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         Value len = rewriter.create<arith::SubIOp>(loc, end, start);

         auto allocated = rewriter.create<mlir::util::AllocOp>(loc, bufferType.getElementType(), len);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(allocated->getBlock()->getTerminator());
            rewriter.create<mlir::util::DeAllocOp>(loc, allocated);
         }

         rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{}, [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange iters) {
            mlir::Value loadedValue = loadValue(rewriter, loc, listType.getType(), childOriginalValueBuffer, childValueBuffer, childValidityBuffer, childVarLenBuffer, childChildPtr, childNullMultiplier, childColumnOffset, iv);
            loadedValue = arrowTypeTo(rewriter, loc, listType.getType(), typeConverter->convertType(listType.getType()), bufferType.getT(), loadedValue);
            assert(!!loadedValue);
            mlir::Value pos=rewriter.create<mlir::arith::SubIOp>(loc, iv, start);
            rewriter.create<mlir::util::StoreOp>(loc, loadedValue, allocated, pos);
            builder.create<mlir::scf::YieldOp>(loc);
         });
         val = rewriter.create<mlir::util::BufferCreateOp>(loc, bufferType, allocated, len);
      } else {
         assert(val && "unhandled type!!");
      }
      return val;
   }

   public:
   LogicalResult matchAndRewrite(mlir::dsa::At atOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = atOp->getLoc();
      auto baseType = getBaseType(atOp.getType(0));
      mlir::Value index;
      mlir::Value columnOffset;
      auto indexType = rewriter.getIndexType();
      mlir::Value originalValueBuffer;
      mlir::Value valueBuffer;
      mlir::Value validityBuffer;
      mlir::Value varLenBuffer;
      mlir::Value nullMultiplier;
      mlir::Value childPtr;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         if (auto* definingOp = adaptor.getCollection().getDefiningOp()) {
            rewriter.setInsertionPointAfter(definingOp);
         }
         auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.getCollection());
         index = unpacked.getResult(0);
         auto info = unpacked.getResult(1);
         size_t column = atOp.getPos();
         size_t baseOffset = 1 + column * 6; // each column contains the following 5 values
         // columnOffset: Offset where the values for the column begins in the originalValueBuffer
         columnOffset = rewriter.create<mlir::util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset);
         // nullMultiplier: necessary to compute the position of validity bit in validityBuffer
         nullMultiplier = rewriter.create<mlir::util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset + 1);
         // validityBuffer: pointer to bytes to encode invalid values (e.g. null)
         validityBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, info.getType().cast<TupleType>().getType(baseOffset + 2), info, baseOffset + 2);
         // originalValueBuffer: pointer to the location of the row in memory
         originalValueBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, info.getType().cast<TupleType>().getType(baseOffset + 3), info, baseOffset + 3);
         valueBuffer = rewriter.create<mlir::util::ArrayElementPtrOp>(loc, originalValueBuffer.getType(), originalValueBuffer, columnOffset); // pointer to the column
         // varLenBuffer: pointer to variable sized data store
         varLenBuffer = rewriter.create<mlir::util::GetTupleOp>(loc, info.getType().cast<TupleType>().getType(baseOffset + 4), info, baseOffset + 4);
         childPtr = rewriter.create<mlir::util::GetTupleOp>(loc, info.getType().cast<TupleType>().getType(baseOffset + 5), info, baseOffset + 5);
      }
      Value val = loadValue(rewriter, loc, baseType, originalValueBuffer, valueBuffer, validityBuffer, varLenBuffer, childPtr, nullMultiplier, columnOffset, index);

      if (atOp->getNumResults() == 2) {
         Value realPos = rewriter.create<arith::AddIOp>(loc, indexType, columnOffset, index);
         realPos = rewriter.create<arith::MulIOp>(loc, indexType, nullMultiplier, realPos);
         Value isValid = getBit(rewriter, loc, validityBuffer, realPos);
         rewriter.replaceOp(atOp, mlir::ValueRange{val, isValid});
      } else {
         rewriter.replaceOp(atOp, val);
      }
      return success();
   }
};
} // end namespace
namespace mlir::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   auto *context = patterns.getContext();
   auto indexType = IndexType::get(context);
   auto i8ptrType = mlir::util::RefType::get(context, IntegerType::get(context, 8));

   typeConverter.addConversion([context, i8ptrType, indexType, &typeConverter](mlir::dsa::RecordBatchType recordBatchType) {
      std::vector<Type> types;
      types.push_back(indexType);
      if (auto tupleT = recordBatchType.getRowType().dyn_cast_or_null<TupleType>()) {
         for (auto t : tupleT.getTypes()) {
            auto converted = typeConverter.convertType(t);
            auto columnInfoTypes = getColumnInfoTypes(t, converted);
            types.insert(types.end(), columnInfoTypes.begin(), columnInfoTypes.end());
         }
      }
      return (Type) TupleType::get(context, types);
   });
   typeConverter.addConversion([context, &typeConverter, indexType](mlir::dsa::RecordType recordType) {
      return (Type) TupleType::get(context, {indexType, typeConverter.convertType(mlir::dsa::RecordBatchType::get(context, recordType.getRowType()))});
   });
   patterns.insert<ColumnnBuilderConcat, CBAppendLowering, ColumnnBuilderFinish, CreateColumnBuilderLowering, CreateTableLowering, ArrowTypeToLowering, ArrowTypeFromLowering, AtLowering>(typeConverter, patterns.getContext());
}
} // end namespace mlir::dsa
