#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/Dialect/util/UtilTypes.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
using namespace lingodb::compiler::dialect;
static mlir::LLVM::LLVMStructType convertTuple(TupleType tupleType, const TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      types.push_back(typeConverter.convertType(t));
   }
   return mlir::LLVM::LLVMStructType::getLiteral(tupleType.getContext(), types);
}

class PackOpLowering : public OpConversionPattern<util::PackOp> {
   public:
   using OpConversionPattern<util::PackOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::PackOp packOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto tupleType = mlir::dyn_cast_or_null<TupleType>(packOp.getTuple().getType());
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::UndefOp>(packOp->getLoc(), structType);
      unsigned pos = 0;
      for (auto val : adaptor.getVals()) {
         tpl = rewriter.create<LLVM::InsertValueOp>(packOp->getLoc(), tpl, val, rewriter.getDenseI64ArrayAttr(pos++));
      }
      rewriter.replaceOp(packOp, tpl);
      return success();
   }
};
class UndefOpLowering : public OpConversionPattern<util::UndefOp> {
   public:
   using OpConversionPattern<util::UndefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::UndefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto ty = typeConverter->convertType(op->getResult(0).getType());
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, ty);
      return success();
   }
};
class GetTupleOpLowering : public OpConversionPattern<util::GetTupleOp> {
   public:
   public:
   using OpConversionPattern<util::GetTupleOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::GetTupleOp getTupleOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto resType = typeConverter->convertType(getTupleOp.getVal().getType());
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(getTupleOp, resType, adaptor.getTuple(), rewriter.getDenseI64ArrayAttr(getTupleOp.getOffset()));
      return success();
   }
};
class SizeOfOpLowering : public ConversionPattern {
   public:
   DataLayout defaultLayout;
   LLVMTypeConverter& llvmTypeConverter;
   explicit SizeOfOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, util::SizeOfOp::getOperationName(), 1, context), defaultLayout(), llvmTypeConverter(typeConverter) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto sizeOfOp = mlir::dyn_cast_or_null<util::SizeOfOp>(op);
      Type t = typeConverter->convertType(sizeOfOp.getType());
      const DataLayout* layout = &defaultLayout;
      if (const DataLayoutAnalysis* analysis = llvmTypeConverter.getDataLayoutAnalysis()) {
         layout = &analysis->getAbove(op);
      }
      size_t typeSize = layout->getTypeSize(t);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(typeSize));
      return success();
   }
};

class ToGenericMemrefOpLowering : public OpConversionPattern<util::ToGenericMemrefOp> {
   public:
   using OpConversionPattern<util::ToGenericMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::ToGenericMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value elementPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), ptrType, adaptor.getMemref(), rewriter.getDenseI64ArrayAttr(1));
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class ToMemrefOpLowering : public OpConversionPattern<util::ToMemrefOp> {
   public:
   using OpConversionPattern<util::ToMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::ToMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto memrefType = mlir::cast<MemRefType>(op.getMemref().getType());

      auto targetType = typeConverter->convertType(memrefType);

      auto targetPointerType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);

      Value elementPtr = adaptor.getRef();
      auto offset = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value deadBeefConst = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xdeadbeef));
      auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), targetPointerType, deadBeefConst);

      Value alignedPtr = elementPtr;
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, allocatedPtr, rewriter.getDenseI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, alignedPtr, rewriter.getDenseI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, offset, rewriter.getDenseI64ArrayAttr(2));
      rewriter.replaceOp(op, tpl);
      return success();
   }
};
class IsRefValidOpLowering : public OpConversionPattern<util::IsRefValidOp> {
   public:
   using OpConversionPattern<util::IsRefValidOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::IsRefValidOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::ne, adaptor.getRef(), rewriter.create<mlir::LLVM::ZeroOp>(op->getLoc(), adaptor.getRef().getType()));
      return success();
   }
};
class InvalidRefOpLowering : public OpConversionPattern<util::InvalidRefOp> {
   public:
   using OpConversionPattern<util::InvalidRefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::InvalidRefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, typeConverter->convertType(op.getType()));
      return success();
   }
};
class AllocaOpLowering : public OpConversionPattern<util::AllocaOp> {
   public:
   using OpConversionPattern<util::AllocaOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::AllocaOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();
      auto genericMemrefType = mlir::cast<util::RefType>(allocOp.getRef().getType());
      Value entries;
      if (allocOp.getSize()) {
         entries = adaptor.getSize();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      mlir::Value allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(loc, elemPtrType, elemType, entries, 0);
      rewriter.replaceOp(allocOp, allocatedElementPtr);

      return success();
   }
};
class AllocOpLowering : public OpConversionPattern<util::AllocOp> {
   public:
   using OpConversionPattern<util::AllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::AllocOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();

      auto genericMemrefType = mlir::cast<util::RefType>(allocOp.getRef().getType());
      Value entries;
      if (allocOp.getSize()) {
         entries = adaptor.getSize();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }

      mlir::Value bytesPerEntry = rewriter.create<util::SizeOfOp>(loc, rewriter.getIndexType(), genericMemrefType.getElementType());
      bytesPerEntry = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI64Type(), bytesPerEntry).getResult(0);
      Value sizeInBytes = rewriter.create<mlir::LLVM::MulOp>(loc, rewriter.getI64Type(), entries, bytesPerEntry);
      LLVM::LLVMFuncOp mallocFunc = LLVM::lookupOrCreateMallocFn(allocOp->getParentOfType<ModuleOp>(), rewriter.getI64Type()).value(); //todo: check for error
      auto result = rewriter.create<mlir::LLVM::CallOp>(loc, mallocFunc, mlir::ValueRange{sizeInBytes}).getResult();
      rewriter.replaceOp(allocOp, result);

      return success();
   }
};
class DeAllocOpLowering : public OpConversionPattern<util::DeAllocOp> {
   public:
   using OpConversionPattern<util::DeAllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::DeAllocOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>()).value(); //todo: check for error
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, freeFunc, adaptor.getRef());
      return success();
   }
};
class StoreElementOpLowering : public OpConversionPattern<util::StoreElementOp> {
   public:
   using OpConversionPattern<util::StoreElementOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::StoreElementOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto elemType = typeConverter->convertType(op.getRef().getType().getElementType());
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value structIdx = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), op.getIdxAttr());
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, elemType, adaptor.getRef(), ValueRange({zero, structIdx}));
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getVal(), elementPtr);
      return success();
   }
};
class LoadElementOpLowering : public OpConversionPattern<util::LoadElementOp> {
   public:
   using OpConversionPattern<util::LoadElementOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::LoadElementOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto structType = typeConverter->convertType(op.getRef().getType().getElementType());
      auto elementType = mlir::cast<mlir::LLVM::LLVMStructType>(structType).getBody()[op.getIdx()];
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value structIdx = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), op.getIdxAttr());
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, structType, adaptor.getRef(), ValueRange({zero, structIdx}));
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elementType, elementPtr);
      return success();
   }
};
class StoreOpLowering : public OpConversionPattern<util::StoreOp> {
   public:
   using OpConversionPattern<util::StoreOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.getRef();
      auto elemType = typeConverter->convertType(op.getRef().getType().getElementType());
      if (adaptor.getIdx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), elemType, elementPtr, adaptor.getIdx());
      }
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getVal(), elementPtr);
      return success();
   }
};
class LoadOpLowering : public OpConversionPattern<util::LoadOp> {
   public:
   using OpConversionPattern<util::LoadOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.getRef();
      auto elemType = typeConverter->convertType(op.getRef().getType().getElementType());
      if (adaptor.getIdx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), elemType, elementPtr, adaptor.getIdx());
      }
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemType, elementPtr);
      return success();
   }
};
class UnalignedLoadOpLowering : public OpConversionPattern<util::UnalignedLoadOp> {
   public:
   using OpConversionPattern<util::UnalignedLoadOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::UnalignedLoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.getRef();
      auto elemType = typeConverter->convertType(op.getRef().getType().getElementType());
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemType, elementPtr,/*alignment=*/1);
      return success();
   }
};
class CastOpLowering : public OpConversionPattern<util::GenericMemrefCastOp> {
   public:
   using OpConversionPattern<util::GenericMemrefCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::GenericMemrefCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, adaptor.getVal());
      return success();
   }
};
class BufferCastOpLowering : public OpConversionPattern<util::BufferCastOp> {
   public:
   using OpConversionPattern<util::BufferCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BufferCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, adaptor.getVal());
      return success();
   }
};
class BufferCreateOpLowering : public OpConversionPattern<util::BufferCreateOp> {
   public:
   using OpConversionPattern<util::BufferCreateOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BufferCreateOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Type t = typeConverter->convertType(mlir::cast<util::BufferType>(op.getResult().getType()).getT());
      DataLayout defaultLayout;
      const DataLayout* layout = &defaultLayout;
      auto& llvmTypeConverter = *reinterpret_cast<const LLVMTypeConverter*>(getTypeConverter());
      if (const DataLayoutAnalysis* analysis = llvmTypeConverter.getDataLayoutAnalysis()) {
         layout = &analysis->getAbove(op);
      }
      size_t typeSize = layout->getTypeSize(t);
      mlir::Type i128Ty = rewriter.getIntegerType(128);
      auto typeSizeValue = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, typeSize));

      mlir::Value p1 = rewriter.create<mlir::LLVM::ZExtOp>(op->getLoc(), i128Ty, adaptor.getLen());
      p1 = rewriter.create<mlir::LLVM::MulOp>(op->getLoc(), i128Ty, p1, typeSizeValue);
      mlir::Value p2 = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), i128Ty, adaptor.getPtr());
      auto const64 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
      auto shlp2 = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), p2, const64);
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, p1, shlp2);
      return success();
   }
};
class TupleElementPtrOpLowering : public OpConversionPattern<util::TupleElementPtrOp> {
   public:
   using OpConversionPattern<util::TupleElementPtrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::TupleElementPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto elemType = typeConverter->convertType(op.getRef().getType().getElementType());
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value structIdx = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(op.getIdx()));
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, elemType, adaptor.getRef(), ValueRange({zero, structIdx}));
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class ArrayElementPtrOpLowering : public OpConversionPattern<util::ArrayElementPtrOp> {
   public:
   using OpConversionPattern<util::ArrayElementPtrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::ArrayElementPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto elemType = typeConverter->convertType(op.getRef().getType().getElementType());
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, elemType, adaptor.getRef(), adaptor.getIdx());
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};

class CreateVarLenLowering : public OpConversionPattern<util::CreateVarLen> {
   public:
   using OpConversionPattern<util::CreateVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::CreateVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "createVarLen32", {mlir::LLVM::LLVMPointerType::get(getContext()), rewriter.getI32Type()}, rewriter.getIntegerType(128)).value(); //todo: check for error
      auto result = rewriter.create<mlir::LLVM::CallOp>(op->getLoc(), fn, mlir::ValueRange{adaptor.getRef(), adaptor.getLen()}).getResult();
      rewriter.replaceOp(op, result);
      return success();
   }
};
class VarLenCmpLowering : public OpConversionPattern<util::VarLenCmp> {
   public:
   using OpConversionPattern<util::VarLenCmp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenCmp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      Value shiftAmount = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      Value first64Left = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), adaptor.getLeft());
      Value last64Left = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::LShrOp>(loc, adaptor.getLeft(), shiftAmount));
      Value last64Right = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::LShrOp>(loc, adaptor.getRight(), shiftAmount));
      Value first64Right = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), adaptor.getRight());
      // this both compares the lengths and the first 4 chars of the string
      Value first64Eq = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, first64Left, first64Right);
      Value last64Eq = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, last64Left, last64Right);
      Value totalEq = rewriter.create<LLVM::AndOp>(loc, last64Eq, first64Eq);
      Value mask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xFFFFFFFF));
      Value c12 = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(12));
      Value len = rewriter.create<LLVM::AndOp>(loc, first64Left, mask);
      Value lenGt12 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, len, c12);
      Value needsDetailedCmp = rewriter.create<LLVM::AndOp>(loc, lenGt12, first64Eq);
      rewriter.replaceOp(op, mlir::ValueRange{totalEq, needsDetailedCmp});
      return success();
   }
};
class VarLenCmpSimpleLowering : public OpConversionPattern<util::VarLenCmpSimple> {
   public:
   using OpConversionPattern<util::VarLenCmpSimple>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenCmpSimple op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();

      Value totalEq = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, adaptor.getLeft(), adaptor.getRight());
      rewriter.replaceOp(op, totalEq);
      return success();
   }
};
class VarLenInvalidLowering : public OpConversionPattern<util::VarLenInvalid> {
   public:
   using OpConversionPattern<util::VarLenInvalid>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenInvalid op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 0xffffffff00000000));
      return success();
   }
};
class VarLenIsInvalidLowering : public OpConversionPattern<util::VarLenIsInvalid> {
   public:
   using OpConversionPattern<util::VarLenIsInvalid>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenIsInvalid op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      Value constInvalid = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 0xffffffff00000000));
      rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq, adaptor.getVarlen(), constInvalid);
      return success();
   }
};
class VarLenTryCheapHashLowering : public OpConversionPattern<util::VarLenTryCheapHash> {
   public:
   using OpConversionPattern<util::VarLenTryCheapHash>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenTryCheapHash op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      Value shiftAmount = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      Value first64 = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), adaptor.getVarlen());
      Value last64 = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::LShrOp>(loc, adaptor.getVarlen(), shiftAmount));

      Value mask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xFFFFFFFF));
      Value c13 = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(13));
      Value len = rewriter.create<LLVM::AndOp>(loc, first64, mask);
      Value lenLt13 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, len, c13);
      Value fHash = rewriter.create<util::Hash64>(loc, rewriter.getIndexType(), first64);
      Value lHash = rewriter.create<util::Hash64>(loc, rewriter.getIndexType(), last64);
      Value hash = rewriter.create<util::HashCombine>(loc, rewriter.getIndexType(), fHash, lHash);
      rewriter.replaceOp(op, mlir::ValueRange{lenLt13, hash});
      return success();
   }
};
class CreateConstVarLenLowering : public OpConversionPattern<util::CreateConstVarLen> {
   public:
   using OpConversionPattern<util::CreateConstVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::CreateConstVarLen op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      size_t len = op.getStr().size();

      mlir::Type i128Ty = rewriter.getIntegerType(128);
      mlir::Value p1, p2;

      uint64_t first4 = 0;
      memcpy(&first4, op.getStr().data(), std::min(4ul, len));
      size_t c1 = (first4 << 32) | len;
      p1 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, c1));
      if (len <= 12) {
         uint64_t last8 = 0;
         if (len > 4) {
            memcpy(&last8, op.getStr().data() + 4, std::min(8ul, len - 4));
         }
         p2 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, last8));
         auto const64 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
         auto shlp2 = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), p2, const64);
         rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, p1, shlp2);
      } else {
         static size_t globalStrConstId = 0;
         mlir::LLVM::GlobalOp globalOp;
         {
            std::string name = "global_str_const_" + std::to_string(globalStrConstId++);
            auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            globalOp = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(), len), true, mlir::LLVM::Linkage::Private, name, op.getStrAttr());
         }
         auto ptr = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), globalOp);
         p2 = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), i128Ty, ptr);
         // shift by 64 (high half) + 2 (storage class bits, GLOBAL=0)
         auto const66 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 66));
         auto shlp2 = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), p2, const66);
         rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, p1, shlp2);
      }
      return success();
   }
};

class VarLenGetLenLowering : public OpConversionPattern<util::VarLenGetLen> {
   public:
   using OpConversionPattern<util::VarLenGetLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenGetLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value len = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), adaptor.getVarlen());
      Value mask = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x7FFFFFFF));
      Value castedLen = rewriter.create<LLVM::AndOp>(op->getLoc(), len, mask);

      rewriter.replaceOp(op, castedLen);
      return success();
   }
};
class VarLenGetRefLowering : public OpConversionPattern<util::VarLenGetRef> {
   public:
   using OpConversionPattern<util::VarLenGetRef>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenGetRef op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      Value shift64 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      Value last64 = rewriter.create<LLVM::TruncOp>(loc, rewriter.getIntegerType(64), rewriter.create<LLVM::LShrOp>(loc, adaptor.getVarlen(), shift64));
      Value shift2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(2));
      Value base = rewriter.create<LLVM::IntToPtrOp>(loc,  mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.create<LLVM::LShrOp>(loc, last64, shift2));

      rewriter.replaceOp(op, base);
      return success();
   }
};
class VarLenGetInlinedStringLowering : public OpConversionPattern<util::VarLenGetInlinedString> {
   public:
   using OpConversionPattern<util::VarLenGetInlinedString>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::VarLenGetInlinedString op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto i128Type = rewriter.getIntegerType(128);

      Value const32 = rewriter.create<LLVM::ConstantOp>(loc, i128Type, rewriter.getIntegerAttr(i128Type, 32));
      rewriter.replaceOpWithNewOp<LLVM::LShrOp>(op, adaptor.getVarlen(), const32);
      return mlir::success();
   }
};

class StringStartsWithLowering : public OpConversionPattern<util::StringStartsWith> {
   public:
   using OpConversionPattern<util::StringStartsWith>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::StringStartsWith op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      llvm::StringRef prefix = op.getPrefix();

      static size_t globalPrefixConstId = 0;
      mlir::LLVM::GlobalOp globalOp;
      {
         std::string name = "global_const_prefix" + std::to_string(globalPrefixConstId++);
         auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(moduleOp.getBody());
         globalOp = rewriter.create<mlir::LLVM::GlobalOp>(loc, mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(), prefix.size()), true, mlir::LLVM::Linkage::Private, name, rewriter.getStringAttr(prefix));
      }

      auto memcmpFn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "memcmp",{mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getIntegerType(64)}, rewriter.getI32Type()).value();
      Value prefixPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, globalOp);
      Value prefixLen = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(prefix.size()));
      Value start = rewriter.create<LLVM::GEPOp>(loc,  mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI8Type(), adaptor.getStr(), mlir::ValueRange{adaptor.getStartIndex()});

      auto callOp = rewriter.create<LLVM::CallOp>(loc, memcmpFn, mlir::ValueRange{start, prefixPtr, prefixLen});
      Value cmp = callOp.getResult();
      Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value isMatch = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, cmp, zero);
      rewriter.replaceOp(op, isMatch);
      return mlir::success();
   }
};

class BytesStartsWithLowering : public OpConversionPattern<util::BytesStartsWith> {
   public:
   using OpConversionPattern<util::BytesStartsWith>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BytesStartsWith op,  OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      llvm::StringRef prefix = op.getPrefix();

      uint64_t mask[2] = {0};
      uint8_t* maskBytes = reinterpret_cast<uint8_t*>(mask);
      for (size_t i = 0; i < prefix.size(); ++i)
         maskBytes[i] = 0xFF;
      llvm::APInt maskAP(128, llvm::ArrayRef<uint64_t>(mask, 2));
      auto maskAttr =  rewriter.getIntegerAttr(rewriter.getIntegerType(128), maskAP);
      Value maskVal = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), maskAttr);

      uint64_t needle[2] = {0};
      uint8_t* needleBytes = reinterpret_cast<uint8_t*>(needle);
      memcpy(needleBytes, prefix.data(), prefix.size());
      llvm::APInt needleAP(128, llvm::ArrayRef<uint64_t>(needle, 2));
      auto needleAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(128), needleAP);
      Value needleVal = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), needleAttr);

      Value haystackVal = adaptor.getStr();
      Value constEight = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 8));
      Value startIdx128 = rewriter.create<LLVM::ZExtOp>(loc, rewriter.getIntegerType(128), adaptor.getStartIndex());
      Value bitOff = rewriter.create<LLVM::MulOp>(loc, startIdx128, constEight);
      Value shiftedHaystackVal = rewriter.create<LLVM::LShrOp>(loc, haystackVal, bitOff);
      Value maskedVal = rewriter.create<mlir::LLVM::AndOp>(loc, shiftedHaystackVal, maskVal);
      Value isMatch = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, needleVal, maskedVal);
      rewriter.replaceOp(op, isMatch);
      return mlir::success();
   }
};

class StringEndsWithLowering : public OpConversionPattern<util::StringEndsWith> {
   public:
   using OpConversionPattern<util::StringEndsWith>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::StringEndsWith op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      llvm::StringRef suffix = op.getSuffix();

      static size_t globalSuffixConstId = 0;
      mlir::LLVM::GlobalOp globalOp;
      {
         std::string name = "global_const_suffix" + std::to_string(globalSuffixConstId++);
         auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(moduleOp.getBody());
         globalOp = rewriter.create<mlir::LLVM::GlobalOp>(loc, mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(), suffix.size()), true, mlir::LLVM::Linkage::Private, name, rewriter.getStringAttr(suffix));
      }

      auto memcmpFn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "memcmp",{mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getIntegerType(64)}, rewriter.getI32Type()).value();
      Value suffixPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, globalOp);
      Value suffixLen = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(suffix.size()));

      Value startIndex = rewriter.create<LLVM::SubOp>(loc, adaptor.getEndIndex(), suffixLen);
      Value start = rewriter.create<LLVM::GEPOp>(loc,  mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI8Type(), adaptor.getStr(), mlir::ValueRange{startIndex});

      auto callOp = rewriter.create<LLVM::CallOp>(loc, memcmpFn, mlir::ValueRange{start, suffixPtr, suffixLen});
      Value cmp = callOp.getResult();
      Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value isMatch = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, cmp, zero);
      rewriter.replaceOp(op, isMatch);
      return mlir::success();

   }
};

class BytesEndsWithLowering : public OpConversionPattern<util::BytesEndsWith> {
   public:
   using OpConversionPattern<util::BytesEndsWith>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BytesEndsWith op,  OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      llvm::StringRef suffix = op.getSuffix();

      uint64_t mask[2] = {0};
      uint8_t* maskBytes = reinterpret_cast<uint8_t*>(mask);
      for (size_t i = 0; i < suffix.size(); ++i)
         maskBytes[i] = 0xFF;
      llvm::APInt maskAP(128, llvm::ArrayRef<uint64_t>(mask, 2));
      auto maskAttr =  rewriter.getIntegerAttr(rewriter.getIntegerType(128), maskAP);
      Value maskVal = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), maskAttr);

      uint64_t needle[2] = {0};
      uint8_t* needleBytes = reinterpret_cast<uint8_t*>(needle);
      memcpy(needleBytes, suffix.data(), suffix.size());
      llvm::APInt needleAP(128, llvm::ArrayRef<uint64_t>(needle, 2));
      auto needleAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(128), needleAP);
      Value needleVal = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), needleAttr);
      auto needleLen = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(suffix.size()));

      Value haystackVal = adaptor.getStr();
      auto shiftValueBits = rewriter.create<mlir::LLVM::SubOp>(loc, adaptor.getEndIndex(), needleLen);
      auto bitsCount = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(8));
      auto shiftValue = rewriter.create<mlir::LLVM::MulOp>(loc, shiftValueBits, bitsCount);
      Value shift128 = rewriter.create<LLVM::ZExtOp>(loc, rewriter.getIntegerType(128), shiftValue);
      Value shiftedHaystack = rewriter.create<mlir::LLVM::LShrOp>(loc, haystackVal, shift128);

      Value maskedVal = rewriter.create<mlir::LLVM::AndOp>(loc, shiftedHaystack, maskVal);
      Value isMatch = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, needleVal, maskedVal);
      rewriter.replaceOp(op, isMatch);
      return mlir::success();
   }
};
class RefMemchrLowering : public OpConversionPattern<util::RefMemchr> {
   public:
   using OpConversionPattern<util::RefMemchr>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::RefMemchr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto i64Type = rewriter.getI64Type();
      auto i32Type = rewriter.getI32Type();

      auto memchrFn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "memchr",{mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), i32Type, i64Type}, mlir::LLVM::LLVMPointerType::get(rewriter.getContext())).value();
      Value character = rewriter.create<LLVM::ZExtOp>(loc, i32Type, adaptor.getByte());
      Value res = rewriter.create<LLVM::CallOp>(loc, memchrFn, mlir::ValueRange{adaptor.getRef(), character, adaptor.getLen()}).getResult();

      Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
      Value found = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, res, nullPtr);

      Value resInt = rewriter.create<LLVM::PtrToIntOp>(loc, i64Type, res);
      Value baseInt = rewriter.create<LLVM::PtrToIntOp>(loc, i64Type, adaptor.getRef());
      Value offset = rewriter.create<LLVM::SubOp>(loc, resInt, baseInt);

      Value pos = rewriter.create<LLVM::SelectOp>(loc, found, offset, adaptor.getLen());

      rewriter.replaceOp(op, mlir::ValueRange{found, pos});
      return mlir::success();
   }
};
class InlineMemchrLowering : public OpConversionPattern<util::InlineMemchr> {
   public:
   using OpConversionPattern<util::InlineMemchr>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::InlineMemchr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto i128 = rewriter.getIntegerType(128);
      auto indexType = typeConverter->convertType(rewriter.getIndexType());

      auto uint8ArrayToUint128Value = [&](uint64_t* arr) {
         llvm::APInt arrAP(128, llvm::ArrayRef<uint64_t>(arr, 2));
         auto arrAttr =  rewriter.getIntegerAttr(i128, arrAP);
         return rewriter.create<LLVM::ConstantOp>(loc, i128, arrAttr);
      };

      uint64_t high[2], low[2], mask[2], allOnes[2];
      memset(reinterpret_cast<uint8_t*>(high), 0x80, 16);
      memset(reinterpret_cast<uint8_t*>(low), 0x7F, 16);
      memset(reinterpret_cast<uint8_t*>(allOnes), 0xFF, 16);
      memset(reinterpret_cast<uint8_t*>(mask), 0x01, 16);

      Value highValue = uint8ArrayToUint128Value(high);
      Value lowValue = uint8ArrayToUint128Value(low);
      Value maskMulValue = uint8ArrayToUint128Value(mask);
      Value allOnesValue = uint8ArrayToUint128Value(allOnes);
      Value byteI128Value = rewriter.create<LLVM::ZExtOp>(loc, i128, adaptor.getByte());
      Value patternValue = rewriter.create<LLVM::MulOp>(loc, byteI128Value, maskMulValue);

      // uint64 t lowChars=( ̃block)&high;
      Value block = adaptor.getData();

      // block ^ pattern => bytes which fully match the pattern are set to 0
      // bytes which do not have a 1-bit somewhere
      Value all0IffMatchedByte = rewriter.create<LLVM::XOrOp>(loc, block, patternValue);
      // all0iffNl = block ^ pattern & low => to perform bitwise addition, if the block byte and pattern disagree on the highest bit, we currently set it to 0
      // otherwise, we keep it as is
      Value all0IffMatchedByteOrDisagreeOnHighestBit = rewriter.create<LLVM::AndOp>(loc, all0IffMatchedByte, lowValue);

      // highestBitSetIffByteNot0 = all0iffNl + low => if the previously computed value is not zero, set the highest bit
      // otherwise, keep the highest bit is 0 (which means that either the block byte and pattern agree or they disagree on the highest bit)
      Value highestBitSetIffByteNot0 = rewriter.create<LLVM::AddOp>(loc, all0IffMatchedByteOrDisagreeOnHighestBit, lowValue);
      // ~highestBitSetIffByteNot0 => the highest bit is 1 if either the block byte and pattern agree or they disagree on the highest bit
      // otherwise, the highest bit is 0
      // the other bits are garbage
      Value negatedHighestBitSetIffByteNot0 = rewriter.create<LLVM::XOrOp>(loc, highestBitSetIffByteNot0, allOnesValue);
      // highestSetIf0 = ~ highestBitSetIffByteNot0 & high => remove the garbage bits from previous result to keep only the highest bits
      Value highestSetIf0 = rewriter.create<LLVM::AndOp>(loc, negatedHighestBitSetIffByteNot0, highValue);

      // when do the block byte and pattern disagree on the highest bit?

      Value const127 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(127));
      Value isLessThan127 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ule, adaptor.getByte(), const127);
      // if the pattern does not have the highest bit set, they disagree when the block byte has the highest bit set
      // ~block => if the highest bit of entries is set, it is no longer set
      // if the highest bit was not set, it is set now
      Value negatedBlock = rewriter.create<LLVM::XOrOp>(loc, block, allOnesValue);
      // (~block)&high => bytes within the block without the highest bit set are now 0x80
      // bytes within the block with the highest bit set are 0x00 (mask)
      // result = highestSetIf0 & lowChar => for block bytes with the highest bit set, set the mask to 0
      // if the pattern has the highest bit set, they disagree when the block byte does not have the highest bit set
      // bytes without their highest bit set are set to 0*
      Value blockUsedForMasking = rewriter.create<LLVM::SelectOp>(loc, isLessThan127, negatedBlock, block);
      Value finalMask = rewriter.create<LLVM::AndOp>(loc, blockUsedForMasking, highValue);

      Value initialResult = rewriter.create<LLVM::AndOp>(loc, highestSetIf0, finalMask);

      Value const16 = rewriter.create<LLVM::ConstantOp>(loc, adaptor.getLen().getType(), rewriter.getIntegerAttr(adaptor.getLen().getType(), 16));
      Value const8 = rewriter.create<LLVM::ConstantOp>(loc, adaptor.getLen().getType(), rewriter.getIntegerAttr(adaptor.getLen().getType(), 8));
      Value uselessByteIdx = rewriter.create<LLVM::SubOp>(loc, const16, adaptor.getLen());
      Value numShiftBits = rewriter.create<LLVM::MulOp>(loc, uselessByteIdx, const8);
      Value numShiftBits128 = rewriter.create<LLVM::ZExtOp>(loc, i128, numShiftBits);
      Value lenMask = rewriter.create<LLVM::LShrOp>(loc, allOnesValue, numShiftBits128);
      Value result = rewriter.create<LLVM::AndOp>(loc, initialResult, lenMask);

      Value zero128 = rewriter.create<LLVM::ConstantOp>(loc, i128, rewriter.getIntegerAttr(i128, 0));
      Value found = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, result, zero128);

      Value trailingZeroes = rewriter.create<LLVM::CountTrailingZerosOp>(loc, i128, result, false);
      Value const3 = rewriter.create<LLVM::ConstantOp>(loc, i128, rewriter.getIntegerAttr(i128, 3));
      Value byteIndex = rewriter.create<LLVM::LShrOp>(loc, trailingZeroes, const3);
      Value pos = rewriter.create<LLVM::TruncOp>(loc, indexType, byteIndex);
      rewriter.replaceOp(op, ValueRange{found, pos});
      return mlir::success();
   }
};
class GetUTF8CodeLenOpLowering : public OpConversionPattern<util::GetUTF8CodeLenOp> {
   public:
   using OpConversionPattern<util::GetUTF8CodeLenOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::GetUTF8CodeLenOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto i8Type = rewriter.getI8Type();
      auto resType = typeConverter->convertType(op.getLen().getType());
      mlir::Value byte = adaptor.getByte();

      auto isByteAtLeast = [&](int64_t bound) -> mlir::Value {
         mlir::Value boundConst = rewriter.create<mlir::LLVM::ConstantOp>(loc, i8Type, rewriter.getIntegerAttr(i8Type, bound));
         mlir::Value cmp = rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::uge, byte, boundConst);
         return rewriter.create<mlir::LLVM::ZExtOp>(loc, resType, cmp);
      };
      mlir::Value len = rewriter.create<mlir::LLVM::ConstantOp>(loc, resType, rewriter.getIntegerAttr(resType, 1));
      len = rewriter.create<mlir::LLVM::AddOp>(loc, len, isByteAtLeast(0xC0));
      len = rewriter.create<mlir::LLVM::AddOp>(loc, len, isByteAtLeast(0xE0));
      len = rewriter.create<mlir::LLVM::AddOp>(loc, len, isByteAtLeast(0xF0));
      rewriter.replaceOp(op, len);
      return mlir::success();
   }
};
class BufferGetLenLowering : public OpConversionPattern<util::BufferGetLen> {
   public:
   using OpConversionPattern<util::BufferGetLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BufferGetLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Type t = typeConverter->convertType(mlir::cast<util::BufferType>(op.getBuffer().getType()).getT());
      DataLayout defaultLayout;
      const DataLayout* layout = &defaultLayout;
      auto& llvmTypeConverter = *reinterpret_cast<const LLVMTypeConverter*>(getTypeConverter());
      if (const DataLayoutAnalysis* analysis = llvmTypeConverter.getDataLayoutAnalysis()) {
         layout = &analysis->getAbove(op);
      }
      size_t typeSize = layout->getTypeSize(t);
      auto bytesPerEntry = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(std::max(1ul, typeSize)));
      Value len = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), adaptor.getBuffer());
      len = rewriter.create<mlir::LLVM::UDivOp>(op->getLoc(), len, bytesPerEntry);
      rewriter.replaceOp(op, len);
      return success();
   }
};
class BufferGetRefLowering : public OpConversionPattern<util::BufferGetRef> {
   public:
   using OpConversionPattern<util::BufferGetRef>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BufferGetRef op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto const64 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      auto shiftedLeft = rewriter.create<mlir::LLVM::LShrOp>(op->getLoc(), adaptor.getBuffer(), const64);
      Value refInt = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), shiftedLeft);
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, mlir::LLVM::LLVMPointerType::get(getContext()), refInt);
      return success();
   }
};
class BufferGetElementRefLowering : public OpConversionPattern<util::BufferGetElementRef> {
   public:
   using OpConversionPattern<util::BufferGetElementRef>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BufferGetElementRef op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto const64 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      auto shiftedLeft = rewriter.create<mlir::LLVM::LShrOp>(op->getLoc(), adaptor.getBuffer(), const64);
      Value refInt = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), shiftedLeft);
      Value ptr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()), refInt);
      auto elemType = typeConverter->convertType(op.getBuffer().getType().getT());
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, elemType, ptr, adaptor.getIdx());
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class Hash64Lowering : public OpConversionPattern<util::Hash64> {
   public:
   using OpConversionPattern<util::Hash64>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::Hash64 op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value p1 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(11400714819323198549ull));
      Value m1 = rewriter.create<LLVM::MulOp>(op->getLoc(), p1, adaptor.getVal());
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), m1);
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), m1, reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashCombineLowering : public OpConversionPattern<util::HashCombine> {
   public:
   using OpConversionPattern<util::HashCombine>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::HashCombine op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), adaptor.getH2());
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), adaptor.getH1(), reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashVarLenLowering : public OpConversionPattern<util::HashVarLen> {
   public:
   using OpConversionPattern<util::HashVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::HashVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "hashVarLenData", {rewriter.getIntegerType(128)}, rewriter.getI64Type()).value(); //todo: check for error
      auto result = rewriter.create<mlir::LLVM::CallOp>(op->getLoc(), fn, mlir::ValueRange{adaptor.getVal()}).getResult();
      rewriter.replaceOp(op, result);
      return success();
   }
};
class BufferGetMemRefOpLowering : public OpConversionPattern<util::BufferGetMemRefOp> {
   public:
   using OpConversionPattern<util::BufferGetMemRefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::BufferGetMemRefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      // Buffer length in bytes
      auto bytesPerEntry = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1ul));
      Value len = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), adaptor.getBuffer());
      len = rewriter.create<LLVM::UDivOp>(op->getLoc(), len, bytesPerEntry);
      // Buffer pointer
      auto const64 = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      auto shiftedLeft = rewriter.create<LLVM::LShrOp>(op->getLoc(), adaptor.getBuffer(), const64);
      Value refInt = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), shiftedLeft);
      Value elementPtr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), LLVM::LLVMPointerType::get(getContext()), refInt);

      // Setup undefined memref
      auto memrefType = cast<MemRefType>(op.getMemref().getType());
      auto targetType = typeConverter->convertType(memrefType);
      auto targetPointerType = LLVM::LLVMPointerType::get(getContext());
      Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);

      // Get values to build a memref: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      // ptr
      Value deadBeefConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xdeadbeef));
      auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), targetPointerType, deadBeefConst);
      // ptr
      Value alignedPtr = elementPtr;
      // i64
      auto offset = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));

      // Insert values into "undefined" memref<?xi8> to make it a valid one
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, allocatedPtr, rewriter.getDenseI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, alignedPtr, rewriter.getDenseI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, offset, rewriter.getDenseI64ArrayAttr(2));
      // array<1 x i64> - dimension size
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, len, rewriter.getDenseI64ArrayAttr({3, 0}));
      // array<1 x i64> - stride
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, bytesPerEntry, rewriter.getDenseI64ArrayAttr({4, 0}));

      rewriter.replaceOp(op, tpl);
      return success();
   }
};

class PtrTagMatchesLowering : public OpConversionPattern<util::PtrTagMatches> {
   public:
   using OpConversionPattern<util::PtrTagMatches>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::PtrTagMatches op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      //optain lookup table:
      // if there is not yet a llvm.mlir global with name bloomMasks than create it in module (external linkage, no value)
      auto moduleOp = op->getParentOfType<ModuleOp>();
      auto globalOp = moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>("bloomMasks");
      if (!globalOp) {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(moduleOp.getBody());
         //Type global_type, /*optional*/bool constant, ::llvm::StringRef sym_name, ::mlir::LLVM::Linkage linkage
         globalOp = rewriter.create<mlir::LLVM::GlobalOp>(loc, mlir::LLVM::LLVMArrayType::get(rewriter.getI16Type(), 2048), true, mlir::LLVM::Linkage::External, "bloomMasks", mlir::Attribute());
      }
      //load the bloom mask from global
      mlir::Value bloomMaskPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, globalOp);

      //take the top 11 bytes from hash value by shifting (64-11) bits to the right
      Value shiftAmount = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(53));
      Value slot = rewriter.create<LLVM::LShrOp>(loc, adaptor.getHash(), shiftAmount);
      //tag = bloomMasks[slot]
      Value tagPtr = rewriter.create<LLVM::GEPOp>(loc, bloomMaskPtr.getType(), rewriter.getI16Type(), bloomMaskPtr, ValueRange{slot});
      Value tag = rewriter.create<LLVM::LoadOp>(loc, rewriter.getI16Type(), tagPtr);
      //entry: (uint16_t)ptr
      Value entry = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI16Type(), adaptor.getRef());
      //return ! (tag & ~ entry)
      Value negatedEntry = rewriter.create<LLVM::XOrOp>(loc, entry, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI16Type(), rewriter.getI16IntegerAttr(0xffff)));
      Value anded = rewriter.create<LLVM::AndOp>(loc, tag, negatedEntry);
      Value isMatch = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, anded, rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI16Type(), rewriter.getI16IntegerAttr(0)));
      rewriter.replaceOp(op, isMatch);

      return success();
   }
};
class UnTagPtrLowering : public OpConversionPattern<util::UnTagPtr> {
   public:
   using OpConversionPattern<util::UnTagPtr>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::UnTagPtr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      Value ptrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), adaptor.getRef());
      //shift 16 bits to right
      Value shiftAmount = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(16));
      Value ptrWithoutTag = rewriter.create<LLVM::LShrOp>(loc, ptrAsInt, shiftAmount);
      ptrWithoutTag = rewriter.create<LLVM::IntToPtrOp>(loc, adaptor.getRef().getType(), ptrWithoutTag);
      rewriter.replaceOp(op, ptrWithoutTag);
      return success();
   }
};
class SetBitConstLowering : public OpConversionPattern<util::SetBitConstOp> {
   public:
   using OpConversionPattern<util::SetBitConstOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::SetBitConstOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      Value bitVal = adaptor.getVal(); // i1
      uint32_t bitPos = op.getBitpos();
      Value bitset = adaptor.getBitset(); // any integer type
      Value bitPosVal = rewriter.create<mlir::LLVM::ConstantOp>(loc, bitset.getType(), rewriter.getIntegerAttr(bitset.getType(), bitPos));
      Value shiftedBit = rewriter.create<mlir::LLVM::ShlOp>(loc, bitset.getType(), rewriter.create<mlir::LLVM::ZExtOp>(loc, bitset.getType(), bitVal), bitPosVal);
      //delete existing bit
      Value invertedMask = rewriter.create<mlir::LLVM::ConstantOp>(loc, bitset.getType(), rewriter.getIntegerAttr(bitset.getType(), ~(1ull << bitPos)));
      bitset = rewriter.create<mlir::LLVM::AndOp>(loc, bitset.getType(), bitset, invertedMask);
      Value result = rewriter.create<mlir::LLVM::OrOp>(loc, bitset.getType(), bitset, shiftedBit);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class IsBitSetConstLowering : public OpConversionPattern<util::IsBitSetConstOp> {
   public:
   using OpConversionPattern<util::IsBitSetConstOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::IsBitSetConstOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      uint32_t bitPos = op.getBitpos();
      Value bitset = adaptor.getBitset(); // any integer type
      Value bitPosVal = rewriter.create<mlir::LLVM::ConstantOp>(loc, bitset.getType(), rewriter.getIntegerAttr(bitset.getType(), bitPos));
      Value shiftedBit = rewriter.create<mlir::LLVM::LShrOp>(loc, bitset.getType(), bitset, bitPosVal);
      Value maskedBit = rewriter.create<mlir::LLVM::AndOp>(loc, bitset.getType(), shiftedBit, rewriter.create<mlir::LLVM::ConstantOp>(loc, bitset.getType(), rewriter.getIntegerAttr(bitset.getType(), 1)));
      Value isSet = rewriter.create<mlir::LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, maskedBit, rewriter.create<mlir::LLVM::ConstantOp>(loc, bitset.getType(), rewriter.getIntegerAttr(bitset.getType(), 0)));
      rewriter.replaceOp(op, isSet);
      return success();
   }
};
class CreateConstArrayLowering: public OpConversionPattern<util::CreateConstArrayOp> {
   public:
   using OpConversionPattern<util::CreateConstArrayOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::CreateConstArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto elemTy = typeConverter->convertType(op.getType().getElementType());
      auto dataAttr = mlir::cast<mlir::ElementsAttr>(op.getData());
      int64_t numElements = dataAttr.getNumElements();

      static size_t globalStrConstId = 0;
      mlir::LLVM::GlobalOp globalOp;
      {
         std::string name = "global_const_array" + std::to_string(globalStrConstId++);
         auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(moduleOp.getBody());
         globalOp = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), mlir::LLVM::LLVMArrayType::get(elemTy, numElements), true, mlir::LLVM::Linkage::Private, name, dataAttr);
         uint64_t alignment = op.getAlignment();
         globalOp.setAlignment(alignment);
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, globalOp);
      return mlir::success();
   }
};
class GetConstArrayAtLowering: public OpConversionPattern<util::GetConstArrayAtOp> {
   public:
   using OpConversionPattern<util::GetConstArrayAtOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::GetConstArrayAtOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto elemTy = typeConverter->convertType(op.getRes().getType());
      auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

      Value elemPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, elemTy, adaptor.getArray(), mlir::ValueRange{adaptor.getIdx()});
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, elemTy, elemPtr);
      return mlir::success();
   }
};
class LoadVectorLowering : public OpConversionPattern<util::LoadVectorOp> {
   public:
   using OpConversionPattern<util::LoadVectorOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::LoadVectorOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto resTy = typeConverter->convertType(op.getRes().getType());
      if (!resTy)
         return mlir::failure();
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resTy, adaptor.getRef(), op.getAlignment());
      return success();
   }
};

class CmpistriLowering : public OpConversionPattern<util::CmpistriOp> {
   public:
   using OpConversionPattern<util::CmpistriOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(util::CmpistriOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value flags = rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getI8Type(), op.getFlagsAttr());
      auto module = op->getParentOfType<ModuleOp>();
      auto pcmpistriFn = LLVM::lookupOrCreateFn(
         module, "llvm.x86.sse42.pcmpistri128",
         {adaptor.getA().getType(), adaptor.getB().getType(), rewriter.getI8Type()}, rewriter.getI32Type());
      if (failed(pcmpistriFn))
         return mlir::failure();

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
         op, pcmpistriFn.value(), ValueRange{adaptor.getA(), adaptor.getB(), flags});
      return mlir::success();
   }
};

} // end anonymous namespace

void util::populateUtilToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](util::RefType genericMemrefType) -> Type {
      return mlir::LLVM::LLVMPointerType::get(patterns.getContext());
   });
   typeConverter.addConversion([&](util::VarLen32Type varLen32Type) {
      MLIRContext* context = &typeConverter.getContext();
      return IntegerType::get(context, 128);
   });
   typeConverter.addConversion([&](util::BufferType bufferType) {
      MLIRContext* context = &typeConverter.getContext();
      return IntegerType::get(context, 128);
   });
   patterns.add<CastOpLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferCastOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SizeOfOpLowering>(typeConverter, patterns.getContext());
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<PackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocaOpLowering>(typeConverter, patterns.getContext());
   patterns.add<DeAllocOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ArrayElementPtrOpLowering>(typeConverter, patterns.getContext());
   patterns.add<TupleElementPtrOpLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferGetRefLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferGetLenLowering>(typeConverter, patterns.getContext());
   patterns.add<ToGenericMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ToMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<IsRefValidOpLowering>(typeConverter, patterns.getContext());
   patterns.add<InvalidRefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<StoreOpLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UnalignedLoadOpLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateConstVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenGetLenLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenGetRefLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenGetInlinedStringLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenCmpLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenCmpSimpleLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenInvalidLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenIsInvalidLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenTryCheapHashLowering>(typeConverter, patterns.getContext());
   patterns.add<StringStartsWithLowering>(typeConverter, patterns.getContext());
   patterns.add<BytesStartsWithLowering>(typeConverter, patterns.getContext());
   patterns.add<RefMemchrLowering>(typeConverter, patterns.getContext());
   patterns.add<InlineMemchrLowering>(typeConverter, patterns.getContext());
   patterns.add<StringEndsWithLowering>(typeConverter, patterns.getContext());
   patterns.add<BytesEndsWithLowering>(typeConverter, patterns.getContext());
   patterns.add<GetUTF8CodeLenOpLowering>(typeConverter, patterns.getContext());
   patterns.add<HashCombineLowering>(typeConverter, patterns.getContext());
   patterns.add<Hash64Lowering>(typeConverter, patterns.getContext());
   patterns.add<HashVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<PtrTagMatchesLowering>(typeConverter, patterns.getContext());
   patterns.add<UnTagPtrLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferCreateOpLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferGetMemRefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferGetElementRefLowering>(typeConverter, patterns.getContext());
   patterns.add<StoreElementOpLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadElementOpLowering>(typeConverter, patterns.getContext());
   patterns.add<IsBitSetConstLowering>(typeConverter, patterns.getContext());
   patterns.add<SetBitConstLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateConstArrayLowering>(typeConverter, patterns.getContext());
   patterns.add<GetConstArrayAtLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadVectorLowering>(typeConverter, patterns.getContext());
   patterns.add<CmpistriLowering>(typeConverter, patterns.getContext());
}
namespace {

class FuncConstTypeConversionPattern : public ConversionPattern {
   public:
   explicit FuncConstTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::func::ConstantOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto constantOp = mlir::cast<mlir::func::ConstantOp>(op);
      auto funcType = mlir::cast<mlir::FunctionType>(constantOp.getType());
      llvm::SmallVector<mlir::Type> convertedFuncInputTypes;
      llvm::SmallVector<mlir::Type> convertedFuncResultsTypes;
      if (typeConverter->convertTypes(funcType.getInputs(), convertedFuncInputTypes).failed()) {
         return failure();
      }
      if (typeConverter->convertTypes(funcType.getResults(), convertedFuncResultsTypes).failed()) {
         return failure();
      }
      rewriter.replaceOpWithNewOp<mlir::func::ConstantOp>(op, rewriter.getFunctionType(convertedFuncInputTypes, convertedFuncResultsTypes), constantOp.getValue());
      return success();
   }
};
class CallIndirectTypeConversionPattern : public ConversionPattern {
   public:
   explicit CallIndirectTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::func::CallIndirectOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto callIndirectOp = mlir::cast<mlir::func::CallIndirectOp>(op);
      mlir::func::CallIndirectOpAdaptor adaptor(operands);
      auto funcType = mlir::cast<mlir::FunctionType>(callIndirectOp.getCallee().getType());
      llvm::SmallVector<mlir::Type> convertedFuncInputTypes;
      llvm::SmallVector<mlir::Type> convertedFuncResultsTypes;
      if (typeConverter->convertTypes(funcType.getInputs(), convertedFuncInputTypes).failed()) {
         return failure();
      }
      if (typeConverter->convertTypes(funcType.getResults(), convertedFuncResultsTypes).failed()) {
         return failure();
      }

      auto newFunctionType = rewriter.getFunctionType(convertedFuncInputTypes, convertedFuncResultsTypes);
      mlir::Value callee = rewriter.create<mlir::UnrealizedConversionCastOp>(op->getLoc(), newFunctionType, adaptor.getCallee()).getResult(0);
      rewriter.replaceOpWithNewOp<mlir::func::CallIndirectOp>(op, callee, adaptor.getCalleeOperands());
      return success();
   }
};
class ArithSelectTypeConversionPattern : public ConversionPattern {
   public:
   explicit ArithSelectTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::arith::SelectOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, operands);
      return success();
   }
};
bool isUtilType(mlir::Type t, TypeConverter& converter) {
   if (auto funcType = mlir::dyn_cast_or_null<mlir::FunctionType>(t)) {
      return llvm::any_of(funcType.getInputs(), [&](auto t) { return isUtilType(t, converter); }) || llvm::any_of(funcType.getResults(), [&](auto t) { return isUtilType(t, converter); });
   } else {
      auto converted = converter.convertType(t);
      return converted && converted != t;
   }
}
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UtilToLLVMLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "convert-util-to-llvm"; }

   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect>();
   }
   void runOnOperation() final {
      Operation* op = getOperation();

      const auto& dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
      LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(op));

      LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
      RewritePatternSet patterns(&getContext());
      util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
      mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
      mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
      mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
      patterns.add<FuncConstTypeConversionPattern>(typeConverter, patterns.getContext());
      patterns.insert<CallIndirectTypeConversionPattern>(typeConverter, &getContext());
      patterns.add<ArithSelectTypeConversionPattern>(typeConverter, patterns.getContext());
      LLVMConversionTarget target(getContext());
      target.addIllegalDialect<util::UtilDialect>();
      target.addLegalDialect<LLVM::LLVMDialect>();
      auto hasUtilType = [&](TypeConverter& converter, TypeRange types) {
         return llvm::any_of(types, [&](auto t) { return isUtilType(t, converter); });
      };
      auto opIsWithoutUtilTypes = [&](Operation* op) { return !hasUtilType(typeConverter, op->getOperandTypes()) && !hasUtilType(typeConverter, op->getResultTypes()); };

      target.addDynamicallyLegalOp<mlir::func::CallOp, mlir::func::CallIndirectOp, mlir::func::ReturnOp, mlir::arith::SelectOp>(opIsWithoutUtilTypes);
      target.addDynamicallyLegalDialect<mlir::cf::ControlFlowDialect>(opIsWithoutUtilTypes);

      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
         auto isSignatureLegal = !hasUtilType(typeConverter, op.getFunctionType().getInputs()) &&
            !hasUtilType(typeConverter, op.getFunctionType().getResults());
         for (auto& block : op.getBody().getBlocks()) {
            if (hasUtilType(typeConverter, block.getArgumentTypes())) {
               return false;
            }
         }
         return isSignatureLegal;
      });
      target.addDynamicallyLegalOp<mlir::func::ConstantOp>([&](mlir::func::ConstantOp op) {
         if (auto functionType = mlir::dyn_cast_or_null<mlir::FunctionType>(op.getType())) {
            auto isLegal = !hasUtilType(typeConverter, functionType.getInputs()) &&
               !hasUtilType(typeConverter, functionType.getResults());
            return isLegal;
         } else {
            return true;
         }
      });

      //target.addLegalOp<func::FuncOp>();
      if (failed(applyPartialConversion(op, target, std::move(patterns))))
         signalPassFailure();
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> util::createUtilToLLVMPass() {
   return std::make_unique<UtilToLLVMLoweringPass>();
}
