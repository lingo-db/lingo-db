#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {

template <class Op>
class SimpleTypeConversionPattern : public ConversionPattern {
   public:
   explicit SimpleTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, Op::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type> convertedTypes;
      assert(typeConverter->convertTypes(op->getResultTypes(), convertedTypes).succeeded());
      rewriter.replaceOpWithNewOp<Op>(op, convertedTypes, ValueRange(operands), op->getAttrs());
      return success();
   }
};

class SizeOfLowering :  public OpConversionPattern<mlir::util::SizeOfOp> {
   public:
   using OpConversionPattern<mlir::util::SizeOfOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::SizeOfOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::util::SizeOfOp>(op, rewriter.getIndexType(), TypeAttr::get(typeConverter->convertType(op.type())));
      return success();
   }
};

} // end anonymous namespace

namespace {
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void mlir::util::populateUtilTypeConversionPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   patterns.add<SimpleTypeConversionPattern<GetTupleOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<UndefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<PackOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<UnPackOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<ToGenericMemrefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<ToMemrefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<IsRefValidOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<InvalidRefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<StoreOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<AllocOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<AllocaOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<DeAllocOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<GenericMemrefCastOp>>(typeConverter, patterns.getContext());
   patterns.add<SizeOfLowering>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<LoadOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<TupleElementPtrOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<ArrayElementPtrOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<FilterTaggedPtr>>(typeConverter, patterns.getContext());

   typeConverter.addConversion([&](mlir::util::RefType genericMemrefType) {
      return mlir::util::RefType::get(genericMemrefType.getContext(), typeConverter.convertType(genericMemrefType.getElementType()));
   });
   typeConverter.addConversion([&](mlir::util::VarLen32Type varType) {
      return varType;
   });
}
