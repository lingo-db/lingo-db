#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/runtime/ArrowColumn.h"
#include "lingodb/compiler/runtime/ArrowTable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {
using namespace lingodb::compiler::dialect;
using namespace lingodb::compiler::runtime;
mlir::Value arrowTypeFrom(mlir::OpBuilder rewriter, mlir::Location loc, mlir::Type arrowType, mlir::Type physicalType, mlir::Type inputType, mlir::Value physicalVal) {
   if (mlir::isa<dsa::ArrowDate32Type, dsa::ArrowDate64Type, dsa::ArrowTimeStampType>(arrowType)) {
      size_t multiplier = 1;
      if (mlir::isa<dsa::ArrowDate32Type>(arrowType)) {
         multiplier = 86400000000000;
      } else if (mlir::isa<dsa::ArrowDate64Type>(arrowType)) {
         multiplier = 1000000;
      } else if (auto timeStampType = mlir::dyn_cast_or_null<dsa::ArrowTimeStampType>(arrowType)) {
         switch (timeStampType.getUnit()) {
            case dsa::TimeUnitAttr::second: multiplier = 1000000000; break;
            case dsa::TimeUnitAttr::millisecond: multiplier = 1000000; break;
            case dsa::TimeUnitAttr::microsecond: multiplier = 1000; break;
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
   } else if (auto decimalType = mlir::dyn_cast_or_null<dsa::ArrowDecimalType>(arrowType)) {
      if (inputType.getIntOrFloatBitWidth() != 128) {
         return rewriter.create<arith::ExtSIOp>(loc, physicalType, physicalVal);
      }
   }
   return mlir::Value();
}
mlir::Value arrowTypeTo(mlir::OpBuilder rewriter, mlir::Location loc, mlir::Type arrowType, mlir::Type physicalType, mlir::Type t, mlir::Value physicalVal) {
   if (mlir::isa<dsa::ArrowDate32Type, dsa::ArrowDate64Type, dsa::ArrowTimeStampType>(arrowType)) {
      if (physicalType.getIntOrFloatBitWidth() < 64) {
         physicalVal = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), physicalVal);
      }
      size_t multiplier = 1;
      if (mlir::isa<dsa::ArrowDate32Type>(arrowType)) {
         multiplier = 86400000000000;
      } else if (mlir::isa<dsa::ArrowDate64Type>(arrowType)) {
         multiplier = 1000000;
      } else if (auto timeStampType = mlir::dyn_cast_or_null<dsa::ArrowTimeStampType>(arrowType)) {
         switch (timeStampType.getUnit()) {
            case dsa::TimeUnitAttr::second: multiplier = 1000000000; break;
            case dsa::TimeUnitAttr::millisecond: multiplier = 1000000; break;
            case dsa::TimeUnitAttr::microsecond: multiplier = 1000; break;
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
   } else if (auto decimalType = mlir::dyn_cast_or_null<dsa::ArrowDecimalType>(arrowType)) {
      if (t.getIntOrFloatBitWidth() != 128) {
         return rewriter.create<arith::TruncIOp>(loc, t, physicalVal);
      }
   }
   return mlir::Value();
}
class CBAppendLowering : public OpConversionPattern<dsa::Append> {
   mlir::LogicalResult append(ConversionPatternRewriter& rewriter, Location loc, mlir::Type arrowType, Value builderVal, Value isValid, mlir::Value val) const {
      if (arrowType.isIndex()) {
         ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val});
      } else if (isIntegerType(arrowType, 1)) {
         ArrowColumnBuilder::addBool(rewriter, loc)({builderVal, isValid, val});
      } else if (auto intType = mlir::dyn_cast_or_null<IntegerType>(arrowType)) {
         switch (intType.getWidth()) {
            case 8: ArrowColumnBuilder::addInt8(rewriter, loc)({builderVal, isValid, val}); break;
            case 16: ArrowColumnBuilder::addInt16(rewriter, loc)({builderVal, isValid, val}); break;
            case 32: ArrowColumnBuilder::addInt32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val}); break;
            default: assert(false && "should not happen");
         }
      } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(arrowType)) {
         switch (floatType.getWidth()) {
            case 32: ArrowColumnBuilder::addFloat32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: ArrowColumnBuilder::addFloat64(rewriter, loc)({builderVal, isValid, val}); break;
         }
      } else if (mlir::isa<dsa::ArrowStringType>(arrowType)) {
         ArrowColumnBuilder::addBinary(rewriter, loc)({builderVal, isValid, val});
      } else if (auto fixedWidthType = mlir::dyn_cast_or_null<dsa::ArrowFixedSizedBinaryType>(arrowType)) {
         ArrowColumnBuilder::addFixedSized(rewriter, loc)({builderVal, isValid, val});
      } else if (mlir::isa<dsa::ArrowDecimalType>(arrowType)) {
         ArrowColumnBuilder::addDecimal(rewriter, loc)({builderVal, isValid, val});
      } else if (mlir::isa<dsa::ArrowTimeStampType, dsa::ArrowDate64Type, dsa::ArrowDayTimeIntervalType>(arrowType)) {
         ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val});
      } else if (mlir::isa<dsa::ArrowDate32Type, dsa::ArrowMonthIntervalType>(arrowType)) {
         ArrowColumnBuilder::addInt32(rewriter, loc)({builderVal, isValid, val});
      } else {
         return mlir::failure();
      }
      return mlir::success();
   }

   public:
   using OpConversionPattern<dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!mlir::isa<dsa::ColumnBuilderType>(appendOp.getDs().getType())) {
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
      if (auto arrowListType = mlir::dyn_cast_or_null<dsa::ArrowListType>(arrowType)) {
         ArrowColumnBuilder::addList(rewriter, loc)({builderVal, isValid});
         auto childBuilder = ArrowColumnBuilder::getChildBuilder(rewriter, loc)({builderVal})[0];
         mlir::Value val = adaptor.getVal();
         auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         auto end = rewriter.create<util::BufferGetLen>(loc, rewriter.getIndexType(), val);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         bool failure = false;
         rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{}, [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange iters) {
            auto ptr = builder.create<util::BufferGetElementRef>(loc, mlir::cast<util::BufferType>(adaptor.getVal().getType()).getElementType(), val, iv);
            auto loaded = builder.create<util::LoadOp>(loc, ptr);
            auto arrowType = arrowListType.getType();
            auto physicalType = typeConverter->convertType(arrowType);
            auto inputType = loaded.getType();
            mlir::Value arrowValue = arrowTypeFrom(rewriter, loc, arrowType, physicalType, inputType, loaded);
            if (append(rewriter, loc, arrowListType.getType(), childBuilder, isValid, arrowValue).failed()) {
               failure = true;
               return;
            }
            builder.create<mlir::scf::YieldOp>(loc);
         });
         if (failure) {
            return mlir::failure();
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

class CreateColumnBuilderLowering : public OpConversionPattern<dsa::CreateDS> {
   std::string arrowDescrFromType(mlir::Type type) const {
      if (type.isIndex()) {
         return "int[64]";
      } else if (isIntegerType(type, 1)) {
         return "bool";
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         return "int[" + std::to_string(intWidth) + "]";
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         return "uint[" + std::to_string(uIntWidth) + "]";
      } else if (auto decimalType = mlir::dyn_cast_or_null<dsa::ArrowDecimalType>(type)) {
         auto prec = std::min(decimalType.getP(), (int64_t) 38);
         return "decimal[" + std::to_string(prec) + "," + std::to_string(decimalType.getS()) + "]";
      } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
         return "float[" + std::to_string(floatType.getWidth()) + "]";
      } else if (mlir::isa<dsa::ArrowStringType>(type)) { //todo: do we still need the strings?
         return "string";
      } else if (mlir::isa<dsa::ArrowDate32Type>(type)) {
         return "date[32]";
      } else if (mlir::isa<dsa::ArrowDate64Type>(type)) {
         return "date[64]";
      } else if (auto fixedSizedBinaryType = mlir::dyn_cast_or_null<dsa::ArrowFixedSizedBinaryType>(type)) {
         return "fixed_sized[" + std::to_string(fixedSizedBinaryType.getByteWidth()) + "]";
      } else if (auto intervalType = mlir::dyn_cast_or_null<dsa::ArrowMonthIntervalType>(type)) {
         return "interval_months";
      } else if (auto intervalType = mlir::dyn_cast_or_null<dsa::ArrowDayTimeIntervalType>(type)) {
         return "interval_daytime";
      } else if (auto timestampType = mlir::dyn_cast_or_null<dsa::ArrowTimeStampType>(type)) {
         return "timestamp[" + std::to_string(static_cast<uint32_t>(timestampType.getUnit())) + "]";
      } else if (auto listType = mlir::dyn_cast_or_null<dsa::ArrowListType>(type)) {
         return "list[" + arrowDescrFromType(listType.getType()) + "]";
      }
      assert(false);
      return "";
   }

   public:
   using OpConversionPattern<dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!mlir::isa<dsa::ColumnBuilderType>(createOp.getDs().getType())) {
         return failure();
      }
      auto loc = createOp->getLoc();
      mlir::Value typeDescr = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(getContext()), arrowDescrFromType(mlir::cast<dsa::ColumnBuilderType>(createOp.getType()).getType()));
      Value columnBuilder = ArrowColumnBuilder::create(rewriter, loc)({typeDescr})[0];
      rewriter.replaceOp(createOp, columnBuilder);
      return success();
   }
};
class CreateTableLowering : public OpConversionPattern<dsa::CreateTable> {
   public:
   using OpConversionPattern<dsa::CreateTable>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::CreateTable createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = createOp->getLoc();
      mlir::Value table = ArrowTable::createEmpty(rewriter, loc)({})[0];
      for (auto x : llvm::zip(createOp.getColumnNames(), adaptor.getColumns())) {
         auto name = mlir::cast<StringAttr>(std::get<0>(x)).getValue();
         auto column = std::get<1>(x);
         mlir::Value columnName = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(getContext()), name);
         table = ArrowTable::addColumn(rewriter, loc)({table, columnName, column})[0];
      }
      rewriter.replaceOp(createOp, table);
      return success();
   }
};

class ColumnnBuilderConcat : public OpConversionPattern<dsa::Concat> {
   public:
   using OpConversionPattern<dsa::Concat>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::Concat op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!mlir::isa<dsa::ColumnBuilderType>(op.getLeft().getType()) || !mlir::isa<dsa::ColumnBuilderType>(op.getLeft().getType())) {
         return failure();
      }
      ArrowColumnBuilder::merge(rewriter, op->getLoc())({adaptor.getLeft(), adaptor.getRight()});
      rewriter.replaceOp(op, adaptor.getLeft());
      return success();
   }
};
class ColumnnBuilderFinish : public OpConversionPattern<dsa::FinishColumn> {
   public:
   using OpConversionPattern<dsa::FinishColumn>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::FinishColumn op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!mlir::isa<dsa::ColumnBuilderType>(op.getBuilder().getType())) {
         return failure();
      }
      auto column = ArrowColumnBuilder::finish(rewriter, op->getLoc())({
         adaptor.getBuilder(),
      })[0];
      rewriter.replaceOp(op, column);
      return success();
   }
};
class ArrowTypeToLowering : public OpConversionPattern<dsa::ArrowTypeTo> {
   public:
   using OpConversionPattern<dsa::ArrowTypeTo>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::ArrowTypeTo op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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

class ArrowTypeFromLowering : public OpConversionPattern<dsa::ArrowTypeFrom> {
   public:
   using OpConversionPattern<dsa::ArrowTypeFrom>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::ArrowTypeFrom op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
   auto* context = t.getContext();
   auto indexType = IndexType::get(context);
   auto i8ptrType = util::RefType::get(context, IntegerType::get(context, 8));
   mlir::Type valueType = converted;
   if (mlir::isa<dsa::ArrowStringType, dsa::ArrowListType>(t)) {
      valueType = mlir::IntegerType::get(context, 32);
   } else if (t == mlir::IntegerType::get(context, 1)) {
      valueType = mlir::IntegerType::get(context, 8);
   }

   types.push_back(indexType);
   types.push_back(indexType);
   types.push_back(i8ptrType);
   types.push_back(util::RefType::get(context, valueType));
   types.push_back(i8ptrType);
   types.push_back(i8ptrType);
   return types;
}
class AtLowering : public OpConversionPattern<dsa::At> {
   public:
   using OpConversionPattern<dsa::At>::OpConversionPattern;
   static Value getBit(OpBuilder builder, Location loc, Value bits, Value pos) {
      auto i1Type = IntegerType::get(builder.getContext(), 1);
      auto i8Type = IntegerType::get(builder.getContext(), 8);

      auto indexType = IndexType::get(builder.getContext());
      Value const3 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 3));
      Value const7 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 7));
      Value const1Byte = builder.create<arith::ConstantOp>(loc, i8Type, builder.getIntegerAttr(i8Type, 1));

      Value div8 = builder.create<arith::ShRUIOp>(loc, indexType, pos, const3);
      Value rem8 = builder.create<arith::AndIOp>(loc, indexType, pos, const7);
      Value loadedByte = builder.create<util::LoadOp>(loc, i8Type, bits, div8);
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
      if (mlir::isa<dsa::ArrowStringType>(baseType)) {
         Value pos1 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, index);
         pos1.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value const1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         Value ip1 = rewriter.create<arith::AddIOp>(loc, indexType, index, const1);
         Value pos2 = rewriter.create<util::LoadOp>(loc, rewriter.getI32Type(), valueBuffer, ip1);
         pos2.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
         Value len = rewriter.create<arith::SubIOp>(loc, rewriter.getI32Type(), pos2, pos1);
         Value pos1AsIndex = rewriter.create<arith::IndexCastOp>(loc, indexType, pos1);
         Value ptr = rewriter.create<util::ArrayElementPtrOp>(loc, util::RefType::get(context, rewriter.getI8Type()), varLenBuffer, pos1AsIndex);
         val = rewriter.create<util::CreateVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()), ptr, len);
      } else if (isIntegerType(baseType, 1)) {
         Value realPos = rewriter.create<arith::AddIOp>(loc, indexType, columnOffset, index);
         val = getBit(rewriter, loc, originalValueBuffer, realPos);
      } else if (baseType.isIntOrIndexOrFloat()) {
         //for integers and floats: just load the value
         val = rewriter.create<util::LoadOp>(loc, baseType, valueBuffer, index);
         val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
      } else if (mlir::isa<dsa::ArrowDate32Type, dsa::ArrowDate64Type, dsa::ArrowMonthIntervalType, dsa::ArrowDayTimeIntervalType, dsa::ArrowTimeStampType, dsa::ArrowDecimalType>(baseType)) {
         //dates, timestamps, etc are also just integers
         val = rewriter.create<util::LoadOp>(loc, typeConverter->convertType(baseType), valueBuffer, index);
         val.getDefiningOp()->setAttr("nosideffect", rewriter.getUnitAttr());
      } else if (auto fixedSizeType = mlir::dyn_cast<dsa::ArrowFixedSizedBinaryType>(baseType)) {
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
      } else if (auto listType = mlir::dyn_cast_or_null<dsa::ArrowListType>(baseType)) {
         auto bufferType = util::BufferType::get(context, typeConverter->convertType(listType.getType()));
         auto columnInfoType = rewriter.getTupleType(getColumnInfoTypes(listType.getType(), typeConverter->convertType(listType.getType())));
         auto castedRef = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(columnInfoType), childPtr);
         auto loaded = rewriter.create<util::LoadOp>(loc, castedRef);
         auto unpacked = rewriter.create<util::UnPackOp>(loc, loaded);
         mlir::Value childColumnOffset = unpacked.getResult(0);
         mlir::Value childNullMultiplier = unpacked.getResult(1);
         mlir::Value childValidityBuffer = unpacked.getResult(2);
         mlir::Value childOriginalValueBuffer = unpacked.getResult(3);
         mlir::Value childValueBuffer = rewriter.create<util::ArrayElementPtrOp>(loc, childOriginalValueBuffer.getType(), childOriginalValueBuffer, childColumnOffset); // pointer to the column

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

         auto allocated = rewriter.create<util::AllocOp>(loc, bufferType.getElementType(), len);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(allocated->getBlock()->getTerminator());
            rewriter.create<util::DeAllocOp>(loc, allocated);
         }

         rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{}, [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange iters) {
            mlir::Value loadedValue = loadValue(rewriter, loc, listType.getType(), childOriginalValueBuffer, childValueBuffer, childValidityBuffer, childVarLenBuffer, childChildPtr, childNullMultiplier, childColumnOffset, iv);
            loadedValue = arrowTypeTo(rewriter, loc, listType.getType(), typeConverter->convertType(listType.getType()), bufferType.getT(), loadedValue);
            assert(!!loadedValue);
            mlir::Value pos = rewriter.create<mlir::arith::SubIOp>(loc, iv, start);
            rewriter.create<util::StoreOp>(loc, loadedValue, allocated, pos);
            builder.create<mlir::scf::YieldOp>(loc);
         });
         val = rewriter.create<util::BufferCreateOp>(loc, bufferType, allocated, len);
      } else {
         assert(val && "unhandled type!!");
      }
      return val;
   }

   public:
   LogicalResult matchAndRewrite(dsa::At atOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
         auto unpacked = rewriter.create<util::UnPackOp>(loc, adaptor.getCollection());
         index = unpacked.getResult(0);
         auto info = unpacked.getResult(1);
         size_t column = atOp.getPos();
         size_t baseOffset = 1 + column * 6; // each column contains the following 5 values
         // columnOffset: Offset where the values for the column begins in the originalValueBuffer
         columnOffset = rewriter.create<util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset);
         // nullMultiplier: necessary to compute the position of validity bit in validityBuffer
         nullMultiplier = rewriter.create<util::GetTupleOp>(loc, rewriter.getIndexType(), info, baseOffset + 1);
         // validityBuffer: pointer to bytes to encode invalid values (e.g. null)
         validityBuffer = rewriter.create<util::GetTupleOp>(loc, mlir::cast<TupleType>(info.getType()).getType(baseOffset + 2), info, baseOffset + 2);
         // originalValueBuffer: pointer to the location of the row in memory
         originalValueBuffer = rewriter.create<util::GetTupleOp>(loc, mlir::cast<TupleType>(info.getType()).getType(baseOffset + 3), info, baseOffset + 3);
         valueBuffer = rewriter.create<util::ArrayElementPtrOp>(loc, originalValueBuffer.getType(), originalValueBuffer, columnOffset); // pointer to the column
         // varLenBuffer: pointer to variable sized data store
         varLenBuffer = rewriter.create<util::GetTupleOp>(loc, mlir::cast<TupleType>(info.getType()).getType(baseOffset + 4), info, baseOffset + 4);
         childPtr = rewriter.create<util::GetTupleOp>(loc, mlir::cast<TupleType>(info.getType()).getType(baseOffset + 5), info, baseOffset + 5);
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

class GetRecordBatchLenLowering : public OpConversionPattern<dsa::GetRecordBatchLen> {
   public:
   using OpConversionPattern<dsa::GetRecordBatchLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::GetRecordBatchLen getRecordBatchLen, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<util::GetTupleOp>(getRecordBatchLen, rewriter.getIndexType(), adaptor.getBatch(), 0);
      return success();
   }
};

class GetRecordLowering : public OpConversionPattern<dsa::GetRecord> {
   public:
   using OpConversionPattern<dsa::GetRecord>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::GetRecord getRecord, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<util::PackOp>(getRecord, typeConverter->convertType(getRecord.getRecord().getType()), mlir::ValueRange({adaptor.getPos(), adaptor.getBatch()}));
      return success();
   }
};

} // end namespace
namespace lingodb::compiler::dialect::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) { // NOLINT(misc-use-internal-linkage)
   auto* context = patterns.getContext();
   auto indexType = IndexType::get(context);

   typeConverter.addConversion([context, indexType, &typeConverter](dsa::RecordBatchType recordBatchType) {
      std::vector<Type> types;
      types.push_back(indexType);
      if (auto tupleT = mlir::dyn_cast_or_null<TupleType>(recordBatchType.getRowType())) {
         for (auto t : tupleT.getTypes()) {
            auto converted = typeConverter.convertType(t);
            auto columnInfoTypes = getColumnInfoTypes(t, converted);
            types.insert(types.end(), columnInfoTypes.begin(), columnInfoTypes.end());
         }
      }
      return (Type) TupleType::get(context, types);
   });
   typeConverter.addConversion([context, &typeConverter, indexType](dsa::RecordType recordType) {
      return (Type) TupleType::get(context, {indexType, typeConverter.convertType(dsa::RecordBatchType::get(context, recordType.getRowType()))});
   });
   patterns.insert<ColumnnBuilderConcat, CBAppendLowering, ColumnnBuilderFinish, CreateColumnBuilderLowering, CreateTableLowering, ArrowTypeToLowering, ArrowTypeFromLowering, AtLowering, GetRecordBatchLenLowering, GetRecordLowering>(typeConverter, patterns.getContext());
}
} // end namespace lingodb::compiler::dialect::dsa
