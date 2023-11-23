#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

using namespace mlir;

namespace {

static mlir::LLVM::LLVMStructType convertTuple(TupleType tupleType, const TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      types.push_back(typeConverter.convertType(t));
   }
   return mlir::LLVM::LLVMStructType::getLiteral(tupleType.getContext(), types);
}

class PackOpLowering : public OpConversionPattern<mlir::util::PackOp> {
   public:
   using OpConversionPattern<mlir::util::PackOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::PackOp packOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto tupleType = packOp.getTuple().getType().dyn_cast_or_null<TupleType>();
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
class UndefOpLowering : public OpConversionPattern<mlir::util::UndefOp> {
   public:
   using OpConversionPattern<mlir::util::UndefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::UndefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto ty = typeConverter->convertType(op->getResult(0).getType());
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, ty);
      return success();
   }
};
class GetTupleOpLowering : public OpConversionPattern<mlir::util::GetTupleOp> {
   public:
   public:
   using OpConversionPattern<mlir::util::GetTupleOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::GetTupleOp getTupleOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
      : ConversionPattern(typeConverter, mlir::util::SizeOfOp::getOperationName(), 1, context), defaultLayout(), llvmTypeConverter(typeConverter) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto sizeOfOp = mlir::dyn_cast_or_null<mlir::util::SizeOfOp>(op);
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

class ToGenericMemrefOpLowering : public OpConversionPattern<mlir::util::ToGenericMemrefOp> {
   public:
   using OpConversionPattern<mlir::util::ToGenericMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::ToGenericMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto genericMemrefType = op.getRef().getType().cast<mlir::util::RefType>();
      auto i8PointerType = mlir::LLVM::LLVMPointerType::get(getContext());
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType);
      Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), i8PointerType, adaptor.getMemref(), rewriter.getDenseI64ArrayAttr(1));
      Value elementPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), elemPtrType, alignedPtr);
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class ToMemrefOpLowering : public OpConversionPattern<mlir::util::ToMemrefOp> {
   public:
   using OpConversionPattern<mlir::util::ToMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::ToMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto memrefType = op.getMemref().getType().cast<MemRefType>();

      auto targetType = typeConverter->convertType(memrefType);

      auto targetPointerType = mlir::LLVM::LLVMPointerType::get(getContext());
      Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);

      Value elementPtr = adaptor.getRef();
      auto offset = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value deadBeefConst = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xdeadbeef));
      auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), targetPointerType, deadBeefConst);

      Value alignedPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), targetPointerType, elementPtr);
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, allocatedPtr, rewriter.getDenseI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, alignedPtr, rewriter.getDenseI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, offset, rewriter.getDenseI64ArrayAttr(2));
      rewriter.replaceOp(op, tpl);
      return success();
   }
};
class IsRefValidOpLowering : public OpConversionPattern<mlir::util::IsRefValidOp> {
   public:
   using OpConversionPattern<mlir::util::IsRefValidOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::IsRefValidOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::ne, adaptor.getRef(), rewriter.create<mlir::LLVM::ZeroOp>(op->getLoc(), adaptor.getRef().getType()));
      return success();
   }
};
class InvalidRefOpLowering : public OpConversionPattern<mlir::util::InvalidRefOp> {
   public:
   using OpConversionPattern<mlir::util::InvalidRefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::InvalidRefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, typeConverter->convertType(op.getType()));
      return success();
   }
};
class AllocaOpLowering : public OpConversionPattern<mlir::util::AllocaOp> {
   public:
   using OpConversionPattern<mlir::util::AllocaOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::AllocaOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();
      auto genericMemrefType = allocOp.getRef().getType().cast<mlir::util::RefType>();
      Value entries;
      if (allocOp.getSize()) {
         entries = adaptor.getSize();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }

      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      mlir::Value allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(loc, elemPtrType, entries, 0);
      rewriter.replaceOp(allocOp, allocatedElementPtr);

      return success();
   }
};
class AllocOpLowering : public OpConversionPattern<mlir::util::AllocOp> {
   public:
   using OpConversionPattern<mlir::util::AllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::AllocOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();

      auto genericMemrefType = allocOp.getRef().getType().cast<mlir::util::RefType>();
      Value entries;
      if (allocOp.getSize()) {
         entries = adaptor.getSize();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }

      mlir::Value bytesPerEntry = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), genericMemrefType.getElementType());
      bytesPerEntry = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, rewriter.getI64Type(), bytesPerEntry).getResult(0);
      Value sizeInBytes = rewriter.create<mlir::LLVM::MulOp>(loc, rewriter.getI64Type(), entries, bytesPerEntry);
      LLVM::LLVMFuncOp mallocFunc = LLVM::lookupOrCreateMallocFn(allocOp->getParentOfType<ModuleOp>(), rewriter.getI64Type(), /* todo: opaque pointers*/ false);
      auto result = rewriter.create<mlir::LLVM::CallOp>(loc, mallocFunc, mlir::ValueRange{sizeInBytes}).getResult();
      mlir::Value castedPointer = rewriter.create<LLVM::BitcastOp>(loc, LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType())), result);
      rewriter.replaceOp(allocOp, castedPointer);

      return success();
   }
};
class DeAllocOpLowering : public OpConversionPattern<mlir::util::DeAllocOp> {
   public:
   using OpConversionPattern<mlir::util::DeAllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::DeAllocOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>(), /* todo: opaque pointers*/ false);
      Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)), adaptor.getRef());
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, freeFunc, casted);
      return success();
   }
};

class StoreOpLowering : public OpConversionPattern<mlir::util::StoreOp> {
   public:
   using OpConversionPattern<mlir::util::StoreOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.getRef();
      if (adaptor.getIdx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), elementPtr, adaptor.getIdx());
      }
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getVal(), elementPtr);
      return success();
   }
};
class LoadOpLowering : public OpConversionPattern<mlir::util::LoadOp> {
   public:
   using OpConversionPattern<mlir::util::LoadOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.getRef();
      if (adaptor.getIdx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), elementPtr, adaptor.getIdx());
      }
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elementPtr);
      return success();
   }
};
class CastOpLowering : public OpConversionPattern<mlir::util::GenericMemrefCastOp> {
   public:
   using OpConversionPattern<mlir::util::GenericMemrefCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::GenericMemrefCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetRefType = op.getRes().getType().cast<mlir::util::RefType>();
      auto targetElemType = typeConverter->convertType(targetRefType.getElementType());
      Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), adaptor.getVal());
      rewriter.replaceOp(op, casted);
      return success();
   }
};
class BufferCastOpLowering : public OpConversionPattern<mlir::util::BufferCastOp> {
   public:
   using OpConversionPattern<mlir::util::BufferCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::BufferCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, adaptor.getVal());
      return success();
   }
};
class TupleElementPtrOpLowering : public OpConversionPattern<mlir::util::TupleElementPtrOp> {
   public:
   using OpConversionPattern<mlir::util::TupleElementPtrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::TupleElementPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetMemrefType = op.getType().cast<mlir::util::RefType>();
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(targetMemrefType.getElementType()));
      Value zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value structIdx = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(op.getIdx()));
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, adaptor.getRef(), ValueRange({zero, structIdx}));
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class ArrayElementPtrOpLowering : public OpConversionPattern<mlir::util::ArrayElementPtrOp> {
   public:
   using OpConversionPattern<mlir::util::ArrayElementPtrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::ArrayElementPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetMemrefType = op.getType().cast<mlir::util::RefType>();
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(targetMemrefType.getElementType()));
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, adaptor.getRef(), adaptor.getIdx());
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};

class CreateVarLenLowering : public OpConversionPattern<mlir::util::CreateVarLen> {
   public:
   using OpConversionPattern<mlir::util::CreateVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::CreateVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "createVarLen32", {mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), rewriter.getI32Type()}, rewriter.getIntegerType(128));
      Value castedPointer = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(rewriter.getI8Type()), adaptor.getRef());
      auto result = rewriter.create<mlir::LLVM::CallOp>(op->getLoc(), fn, mlir::ValueRange{castedPointer, adaptor.getLen()}).getResult();
      rewriter.replaceOp(op, result);
      return success();
   }
};
class VarLenCmpLowering : public OpConversionPattern<mlir::util::VarLenCmp> {
   public:
   using OpConversionPattern<mlir::util::VarLenCmp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::VarLenCmp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      Value shiftAmount = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      Value first64Left = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), adaptor.getLeft());
      Value last64Left = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::LShrOp>(loc, adaptor.getLeft(), shiftAmount));
      Value last64Right = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), rewriter.create<LLVM::LShrOp>(loc, adaptor.getRight(), shiftAmount));
      Value first64Right = rewriter.create<LLVM::TruncOp>(loc, rewriter.getI64Type(), adaptor.getRight());
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
class VarLenTryCheapHashLowering : public OpConversionPattern<mlir::util::VarLenTryCheapHash> {
   public:
   using OpConversionPattern<mlir::util::VarLenTryCheapHash>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::VarLenTryCheapHash op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
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
class CreateConstVarLenLowering : public OpConversionPattern<mlir::util::CreateConstVarLen> {
   public:
   using OpConversionPattern<mlir::util::CreateConstVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::CreateConstVarLen op, OpAdaptor adaptor,
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

class VarLenGetLenLowering : public OpConversionPattern<mlir::util::VarLenGetLen> {
   public:
   using OpConversionPattern<mlir::util::VarLenGetLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::VarLenGetLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value len = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), adaptor.getVarlen());
      Value mask = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x7FFFFFFF));
      Value castedLen = rewriter.create<LLVM::AndOp>(op->getLoc(), len, mask);

      rewriter.replaceOp(op, castedLen);
      return success();
   }
};
class BufferGetLenLowering : public OpConversionPattern<mlir::util::BufferGetLen> {
   public:
   using OpConversionPattern<mlir::util::BufferGetLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::BufferGetLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Type t = typeConverter->convertType(op.getBuffer().getType().cast<mlir::util::BufferType>().getT());
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
class BufferGetRefLowering : public OpConversionPattern<mlir::util::BufferGetRef> {
   public:
   using OpConversionPattern<mlir::util::BufferGetRef>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::BufferGetRef op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto const64 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(128), rewriter.getIntegerAttr(rewriter.getIntegerType(128), 64));
      auto shiftedLeft = rewriter.create<mlir::LLVM::LShrOp>(op->getLoc(), adaptor.getBuffer(), const64);
      Value refInt = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), shiftedLeft);
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op.getBuffer().getType().cast<mlir::util::BufferType>().getT())), refInt);
      return success();
   }
};
class Hash64Lowering : public OpConversionPattern<mlir::util::Hash64> {
   public:
   using OpConversionPattern<mlir::util::Hash64>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::Hash64 op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value p1 = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(11400714819323198549ull));
      Value m1 = rewriter.create<LLVM::MulOp>(op->getLoc(), p1, adaptor.getVal());
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), m1);
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), m1, reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashCombineLowering : public OpConversionPattern<mlir::util::HashCombine> {
   public:
   using OpConversionPattern<mlir::util::HashCombine>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::HashCombine op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), adaptor.getH1());
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), adaptor.getH2(), reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashVarLenLowering : public OpConversionPattern<mlir::util::HashVarLen> {
   public:
   using OpConversionPattern<mlir::util::HashVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::HashVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "hashVarLenData", {rewriter.getIntegerType(128)}, rewriter.getI64Type());
      auto result = rewriter.create<mlir::LLVM::CallOp>(op->getLoc(), fn, mlir::ValueRange{adaptor.getVal()}).getResult();
      rewriter.replaceOp(op, result);
      return success();
   }
};

class FilterTaggedPtrLowering : public OpConversionPattern<mlir::util::FilterTaggedPtr> {
   public:
   using OpConversionPattern<mlir::util::FilterTaggedPtr>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::FilterTaggedPtr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto tagMask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xffff000000000000ull));
      auto ptrMask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x0000ffffffffffffull));
      Value maskedHash = rewriter.create<LLVM::AndOp>(loc, adaptor.getHash(), tagMask);
      Value ptrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), adaptor.getRef());
      Value maskedPtr = rewriter.create<LLVM::AndOp>(loc, ptrAsInt, ptrMask);
      maskedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, adaptor.getRef().getType(), maskedPtr);
      Value ored = rewriter.create<LLVM::OrOp>(loc, ptrAsInt, maskedHash);
      Value contained = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ored, ptrAsInt);
      Value nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, adaptor.getRef().getType());

      Value filtered = rewriter.create<LLVM::SelectOp>(loc, contained, maskedPtr, nullPtr);
      rewriter.replaceOp(op, filtered);
      return success();
   }
};
class UnTagPtrLowering : public OpConversionPattern<mlir::util::UnTagPtr> {
   public:
   using OpConversionPattern<mlir::util::UnTagPtr>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::UnTagPtr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto ptrMask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x0000ffffffffffffull));
      Value ptrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), adaptor.getRef());
      Value maskedPtr = rewriter.create<LLVM::AndOp>(loc, ptrAsInt, ptrMask);
      maskedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, adaptor.getRef().getType(), maskedPtr);
      rewriter.replaceOp(op, maskedPtr);
      return success();
   }
};
class TagPtrLowering : public OpConversionPattern<mlir::util::TagPtr> {
   public:
   using OpConversionPattern<mlir::util::TagPtr>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::TagPtr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto tagMask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xffff000000000000ull));
      Value maskedHash = rewriter.create<LLVM::AndOp>(loc, adaptor.getHash(), tagMask);
      Value ptrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), adaptor.getRef());
      Value ored = rewriter.create<LLVM::OrOp>(loc, ptrAsInt, maskedHash);
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, adaptor.getRef().getType(), ored);

      return success();
   }
};

} // end anonymous namespace

void mlir::util::populateUtilToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](mlir::util::RefType genericMemrefType) -> Type {
      return mlir::LLVM::LLVMPointerType::get(typeConverter.convertType(genericMemrefType.getElementType()));
   });
   typeConverter.addConversion([&](mlir::util::VarLen32Type varLen32Type) {
      MLIRContext* context = &typeConverter.getContext();
      return IntegerType::get(context, 128);
   });
   typeConverter.addConversion([&](mlir::util::BufferType bufferType) {
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
   patterns.add<FilterTaggedPtrLowering>(typeConverter, patterns.getContext());
   patterns.add<TagPtrLowering>(typeConverter, patterns.getContext());
   patterns.add<UnTagPtrLowering>(typeConverter, patterns.getContext());
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
      auto funcType = constantOp.getType().cast<mlir::FunctionType>();
      llvm::SmallVector<mlir::Type> convertedFuncInputTypes;
      llvm::SmallVector<mlir::Type> convertedFuncResultsTypes;
      if(typeConverter->convertTypes(funcType.getInputs(), convertedFuncInputTypes).failed()){
         return failure();
      }
      if(typeConverter->convertTypes(funcType.getResults(), convertedFuncResultsTypes).failed()){
         return failure();
      }
      rewriter.replaceOpWithNewOp<mlir::func::ConstantOp>(op, rewriter.getFunctionType(convertedFuncInputTypes, convertedFuncResultsTypes), constantOp.getValue());
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
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UtilToLLVMLoweringPass)
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect>();
   }
   void runOnOperation() final {
      Operation* op = getOperation();
      const auto& dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
      LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(op));

      LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
      RewritePatternSet patterns(&getContext());
      mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
      mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
      mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
      mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
      patterns.add<FuncConstTypeConversionPattern>(typeConverter, patterns.getContext());
      patterns.add<ArithSelectTypeConversionPattern>(typeConverter, patterns.getContext());
      LLVMConversionTarget target(getContext());
      target.addIllegalDialect<mlir::util::UtilDialect>();
      target.addLegalDialect<LLVM::LLVMDialect>();
      auto hasUtilType = [](TypeConverter& converter, TypeRange types) {
         return llvm::any_of(types, [&converter](mlir::Type t) { auto converted = converter.convertType(t);return converted&&converted!=t; });
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
         if (auto functionType = op.getType().dyn_cast_or_null<mlir::FunctionType>()) {
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

std::unique_ptr<mlir::Pass> mlir::util::createUtilToLLVMPass() {
   return std::make_unique<UtilToLLVMLoweringPass>();
}