#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "runtime-defs/Hashtable.h"
#include "runtime-defs/LazyJoinHashtable.h"
#include "runtime-defs/TableBuilder.h"
#include "runtime-defs/Vector.h"
using namespace mlir;
namespace {

class FinalizeTBLowering : public OpConversionPattern<mlir::dsa::Finalize> {
   public:
   using OpConversionPattern<mlir::dsa::Finalize>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Finalize finalizeOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!finalizeOp.ht().getType().isa<mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      mlir::Value res = rt::TableBuilder::build(rewriter, finalizeOp->getLoc())(adaptor.ht())[0];
      rewriter.replaceOp(finalizeOp, res);
      return success();
   }
};

class TBAppendLowering : public OpConversionPattern<mlir::dsa::Append> {
   public:
   using OpConversionPattern<mlir::dsa::Append>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::Append appendOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!appendOp.ds().getType().isa<mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      Value builderVal = adaptor.ds();
      Value val = adaptor.val();
      Value isValid = adaptor.valid();
      auto loc = appendOp->getLoc();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      mlir::Type type = getBaseType(val.getType());
      if (type.isIndex()) {
         rt::TableBuilder::addInt64(rewriter, loc)({builderVal, isValid, val});
      } else if (isIntegerType(type, 1)) {
         rt::TableBuilder::addBool(rewriter, loc)({builderVal, isValid, val});
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         switch (intWidth) {
            case 8: rt::TableBuilder::addInt8(rewriter, loc)({builderVal, isValid, val}); break;
            case 16: rt::TableBuilder::addInt16(rewriter, loc)({builderVal, isValid, val}); break;
            case 32: rt::TableBuilder::addInt32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::TableBuilder::addInt64(rewriter, loc)({builderVal, isValid, val}); break;
            case 128: rt::TableBuilder::addDecimal(rewriter, loc)({builderVal, isValid, val}); break;
            default: {
               val = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), val);
               rt::TableBuilder::addFixedSized(rewriter, loc)({builderVal, isValid, val});
               break;
            }
         }
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 32: rt::TableBuilder::addFloat32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::TableBuilder::addFloat64(rewriter, loc)({builderVal, isValid, val}); break;
         }
      } else if (auto stringType = type.dyn_cast_or_null<mlir::util::VarLen32Type>()) {
         rt::TableBuilder::addBinary(rewriter, loc)({builderVal, isValid, val});
      }
      rewriter.eraseOp(appendOp);
      return success();
   }
};
class NextRowLowering : public OpConversionPattern<mlir::dsa::NextRow> {
   public:
   using OpConversionPattern<mlir::dsa::NextRow>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::NextRow op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rt::TableBuilder::nextRow(rewriter, op->getLoc())({adaptor.builder()});
      rewriter.eraseOp(op);
      return success();
   }
};
class CreateTableBuilderLowering : public OpConversionPattern<mlir::dsa::CreateDS> {
   public:
   using OpConversionPattern<mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.ds().getType().isa<mlir::dsa::TableBuilderType>()) {
         return failure();
      }
      auto loc = createOp->getLoc();
      mlir::Value schema = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(getContext()), createOp.init_attr().getValue().cast<StringAttr>().str());
      Value tableBuilder = rt::TableBuilder::create(rewriter, loc)({schema})[0];
      rewriter.replaceOp(createOp, tableBuilder);
      return success();
   }
};

} // end namespace
namespace mlir::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<CreateTableBuilderLowering, TBAppendLowering, FinalizeTBLowering, NextRowLowering>(typeConverter, patterns.getContext());
}
} // end namespace mlir::dsa
