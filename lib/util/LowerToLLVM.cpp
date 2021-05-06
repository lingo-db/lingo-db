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

static mlir::LLVM::LLVMStructType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      types.push_back(typeConverter.convertType(t));
   }
   return mlir::LLVM::LLVMStructType::getLiteral(tupleType.getContext(), types);
}

class PackOpLowering : public ConversionPattern {
   public:
   explicit PackOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::PackOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto constop = mlir::dyn_cast_or_null<mlir::util::PackOp>(op);
      auto tupleType = constop.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), structType);
      unsigned pos = 0;
      for (auto val : constop.vals()) {
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), tpl, val,
                                                    rewriter.getI64ArrayAttr(pos++));
      }
      rewriter.replaceOp(op, tpl);
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
      auto tupleType = undefTupleOp.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), structType);
      rewriter.replaceOp(op, tpl);
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
      auto setTupleOp = mlir::dyn_cast_or_null<mlir::util::SetTupleOp>(op);
      auto tupleType = setTupleOp.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), structType, setTupleOp.tuple(), setTupleOp.val(),
                                                       rewriter.getI64ArrayAttr(setTupleOp.offset()));

      rewriter.replaceOp(op, tpl);
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
      auto getTupleOp = mlir::dyn_cast_or_null<mlir::util::GetTupleOp>(op);
      auto resType = typeConverter->convertType(getTupleOp.val().getType());
      Value tpl = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), resType, getTupleOp.tuple(),
                                                        rewriter.getI64ArrayAttr(getTupleOp.offset()));

      rewriter.replaceOp(op, tpl);
      return success();
   }
};
class UnPackOpLowering : public ConversionPattern {
   public:
   explicit UnPackOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::UnPackOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto unPackOp = mlir::dyn_cast_or_null<mlir::util::UnPackOp>(op);
      auto tupleType = unPackOp.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      unsigned pos = 0;
      std::vector<Value> values;
      for (auto type : structType.getBody()) {
         values.push_back(rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), type, unPackOp.tuple(), rewriter.getI64ArrayAttr(pos++)));
      }
      rewriter.replaceOp(op, values);
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

void mlir::util::populateUtilToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<PackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UnPackOpLowering>(typeConverter, patterns.getContext());
}
