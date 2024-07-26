#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {
using namespace lingodb::compiler::dialect;
template <class Op>
class SimpleTypeConversionPattern : public ConversionPattern {
   public:
   explicit SimpleTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, Op::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type> convertedTypes;
      if (typeConverter->convertTypes(op->getResultTypes(), convertedTypes).failed()) {
         return failure();
      }
      rewriter.replaceOpWithNewOp<Op>(op, convertedTypes, ValueRange(operands), op->getAttrs());
      return success();
   }
};

class SizeOfLowering : public OpConversionPattern<util::SizeOfOp> {
   public:
   using OpConversionPattern<util::SizeOfOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::SizeOfOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<util::SizeOfOp>(op, rewriter.getIndexType(), TypeAttr::get(typeConverter->convertType(op.getType())));
      return success();
   }
};

} // end anonymous namespace

namespace {
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UtilToLLVMLoweringPass)
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void util::populateUtilTypeConversionPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
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
   patterns.add<SimpleTypeConversionPattern<TagPtr>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<UnTagPtr>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<BufferCastOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<BufferCreateOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<BufferGetLen>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<BufferGetRef>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<BufferGetMemRefOp>>(typeConverter, patterns.getContext());
   patterns.add<SimpleTypeConversionPattern<BufferGetElementRef>>(typeConverter, patterns.getContext());

   typeConverter.addConversion([&](util::RefType genericMemrefType) {
      return util::RefType::get(genericMemrefType.getContext(), typeConverter.convertType(genericMemrefType.getElementType()));
   });
   typeConverter.addConversion([&](util::BufferType genericMemrefType) {
      return util::BufferType::get(genericMemrefType.getContext(), typeConverter.convertType(genericMemrefType.getT()));
   });
   typeConverter.addConversion([&](util::VarLen32Type varType) {
      return varType;
   });
}
