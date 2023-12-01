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
      Value builderVal = adaptor.getDs();
      Value val = adaptor.getVal();
      Value isValid = adaptor.getValid();
      auto loc = appendOp->getLoc();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      mlir::Type type = getBaseType(val.getType());
      if (type.isIndex()) {
         rt::ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val});
      } else if (isIntegerType(type, 1)) {
         rt::ArrowColumnBuilder::addBool(rewriter, loc)({builderVal, isValid, val});
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         if (auto numBytesAttr = appendOp->getAttrOfType<mlir::IntegerAttr>("numBytes")) {
            if (!val.getType().isInteger(64)) {
               val = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), val);
            }
            rt::ArrowColumnBuilder::addFixedSized(rewriter, loc)({builderVal, isValid, val});
         } else {
            switch (intWidth) {
               case 8: rt::ArrowColumnBuilder::addInt8(rewriter, loc)({builderVal, isValid, val}); break;
               case 16: rt::ArrowColumnBuilder::addInt16(rewriter, loc)({builderVal, isValid, val}); break;
               case 32: rt::ArrowColumnBuilder::addInt32(rewriter, loc)({builderVal, isValid, val}); break;
               case 64: rt::ArrowColumnBuilder::addInt64(rewriter, loc)({builderVal, isValid, val}); break;
               case 128: rt::ArrowColumnBuilder::addDecimal(rewriter, loc)({builderVal, isValid, val}); break;
               default: assert(false && "should not happen");
            }
         }
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         switch (floatType.getWidth()) {
            case 32: rt::ArrowColumnBuilder::addFloat32(rewriter, loc)({builderVal, isValid, val}); break;
            case 64: rt::ArrowColumnBuilder::addFloat64(rewriter, loc)({builderVal, isValid, val}); break;
         }
      } else if (auto stringType = type.dyn_cast_or_null<mlir::util::VarLen32Type>()) {
         rt::ArrowColumnBuilder::addBinary(rewriter, loc)({builderVal, isValid, val});
      }
      rewriter.eraseOp(appendOp);
      return success();
   }
};

class CreateColumnBuilderLowering : public OpConversionPattern<mlir::dsa::CreateDS> {
   public:
   using OpConversionPattern<mlir::dsa::CreateDS>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::CreateDS createOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (!createOp.getDs().getType().isa<mlir::dsa::ColumnBuilderType>()) {
         return failure();
      }
      auto loc = createOp->getLoc();
      mlir::Value typeDescr = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(getContext()), createOp.getInitAttr().value().cast<StringAttr>().str());
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
} // end namespace
namespace mlir::dsa {
void populateDSAToStdPatterns(mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns) {
   patterns.insert<ColumnnBuilderConcat, CBAppendLowering, ColumnnBuilderFinish, CreateColumnBuilderLowering, CreateTableLowering>(typeConverter, patterns.getContext());
}
} // end namespace mlir::dsa
