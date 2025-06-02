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
      }
      auto const64 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
      auto shlp2 = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), p2, const64);
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, p1, shlp2);
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
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), adaptor.getH1());
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), adaptor.getH2(), reversed);
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
   patterns.add<CreateVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateConstVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenGetLenLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenCmpLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenTryCheapHashLowering>(typeConverter, patterns.getContext());
   patterns.add<HashCombineLowering>(typeConverter, patterns.getContext());
   patterns.add<Hash64Lowering>(typeConverter, patterns.getContext());
   patterns.add<HashVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<PtrTagMatchesLowering>(typeConverter, patterns.getContext());
   patterns.add<UnTagPtrLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferCreateOpLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferGetMemRefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<BufferGetElementRefLowering>(typeConverter, patterns.getContext());
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
