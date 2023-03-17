#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "runtime-defs/Hashtable.h"
#include "runtime-defs/LazyJoinHashtable.h"
#include "runtime-defs/TableBuilder.h"
using namespace mlir;
namespace {

class TBAppendLowering : public OpConversionPattern<mlir::dsa::Append> {
   public:
   using OpConversionPattern<mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!appendOp.getDs().getType().isa<mlir::dsa::ResultTableType>()) {
         return failure();
      }
      Value builderVal = adaptor.getDs();
      Value val = adaptor.getVal();
      Value isValid = adaptor.getValid();
      auto loc = appendOp->getLoc();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      mlir::Type type = getBaseType(val.getType());
      if (type.isIndex()) {
         rt::ResultTable::addInt64(rewriter, loc)({builderVal, isValid, val});
      } else if (isIntegerType(type, 1)) {
         rt::ResultTable::addBool(rewriter, loc)({builderVal, isValid, val});
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         if (auto numBytesAttr = appendOp->getAttrOfType<mlir::IntegerAttr>("numBytes")) {
            if(!val.getType().isInteger(64)){
               val = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), val);
            }
            rt::ResultTable::addFixedSized(rewriter, loc)({builderVal, isValid, val});
         }else {
            switch (intWidth) {
               case 8: rt::ResultTable::addInt8(rewriter, loc)({builderVal, isValid, val}); break;
               case 16: rt::ResultTable::addInt16(rewriter, loc)({builderVal, isValid, val}); break;
               case 32: rt::ResultTable::addInt32(rewriter, loc)({builderVal, isValid, val}); break;
               case 64: rt::ResultTable::addInt64(rewriter, loc)({builderVal, isValid, val}); break;
               case 128: rt::ResultTable::addDecimal(rewriter, loc)({builderVal, isValid, val}); break;
               default: assert(false&&"should not happen");
            }
         }
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 32: rt::ResultTable::addFloat32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::ResultTable::addFloat64(rewriter, loc)({builderVal, isValid, val}); break;
         }
      } else if (auto stringType = type.dyn_cast_or_null<mlir::util::VarLen32Type>()) {
         rt::ResultTable::addBinary(rewriter, loc)({builderVal, isValid, val});
      }
      rewriter.eraseOp(appendOp);
      return success();
   }
};
class NextRowLowering : public OpConversionPattern<mlir::dsa::NextRow> {
   public:
   using OpConversionPattern<mlir::dsa::NextRow>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::NextRow op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rt::ResultTable::nextRow(rewriter, op->getLoc())({adaptor.getBuilder()});
      rewriter.eraseOp(op);
      return success();
   }
};
class CreateTableBuilderLowering : public OpConversionPattern<mlir::dsa::CreateDS> {
   public:
   using OpConversionPattern<mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getDs().getType().isa<mlir::dsa::ResultTableType>()) {
         return failure();
      }
      auto loc = createOp->getLoc();
      mlir::Value schema = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(getContext()), createOp.getInitAttr().value().cast<StringAttr>().str());
      Value tableBuilder = rt::ResultTable::create(rewriter, loc)({schema})[0];
      rewriter.replaceOp(createOp, tableBuilder);
      return success();
   }
};

} // end namespace
namespace mlir::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateTableBuilderLowering, TBAppendLowering, NextRowLowering>(typeConverter, patterns.getContext());
}
} // end namespace mlir::dsa
