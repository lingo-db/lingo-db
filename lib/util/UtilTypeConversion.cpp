#include "llvm/ADT/Sequence.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {

class CombineOpLowering : public ConversionPattern {
   public:
   explicit CombineOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::CombineOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto constop = mlir::dyn_cast_or_null<mlir::util::CombineOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::CombineOp>(op, typeConverter->convertType(constop.tuple().getType()), ValueRange(operands));
      return success();
   }
};
class UndefTupleOpLowering : public ConversionPattern {
   public:
   explicit UndefTupleOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::UndefTupleOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto undefTupleOp = mlir::dyn_cast_or_null<mlir::util::UndefTupleOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::UndefTupleOp>(op, typeConverter->convertType(undefTupleOp.tuple().getType()));
      return success();
   }
};
class SetTupleOpLowering : public ConversionPattern {
   public:
   explicit SetTupleOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::SetTupleOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::SetTupleOpAdaptor setTupleOpAdaptor(operands);
      auto setTupleOp = mlir::dyn_cast_or_null<mlir::util::SetTupleOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::SetTupleOp>(op, typeConverter->convertType(setTupleOp.tuple_out().getType()), setTupleOpAdaptor.tuple(), setTupleOpAdaptor.val(), setTupleOp.offset());
      return success();
   }
};
class GetTupleOpLowering : public ConversionPattern {
   public:
   explicit GetTupleOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::GetTupleOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::GetTupleOpAdaptor getTupleOpAdaptor(operands);
      auto getTupleOp = mlir::dyn_cast_or_null<mlir::util::GetTupleOp>(op);
      rewriter.replaceOpWithNewOp<mlir::util::GetTupleOp>(op, typeConverter->convertType(getTupleOp.val().getType()), getTupleOpAdaptor.tuple(), getTupleOp.offset());

      return success();
   }
};
class SplitOpLowering : public ConversionPattern {
   public:
   explicit SplitOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::SplitOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::SplitOpAdaptor splitOpAdaptor(operands);

      auto splitOp = mlir::dyn_cast_or_null<mlir::util::SplitOp>(op);
      llvm::SmallVector<Type> valTypes;
      for(auto v:splitOp.vals()){
         Type converted=typeConverter->convertType(v.getType());
         converted=converted?converted:v.getType();
         valTypes.push_back(converted);
      }

      rewriter.replaceOpWithNewOp<mlir::util::SplitOp>(op, valTypes,operands);

      return success();
   }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void mlir::util::populateUtilTypeConversionPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns) {
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<CombineOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SplitOpLowering>(typeConverter, patterns.getContext());
}
