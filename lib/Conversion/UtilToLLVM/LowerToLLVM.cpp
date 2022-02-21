#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

using namespace mlir;

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
      auto packOp = mlir::dyn_cast_or_null<mlir::util::PackOp>(op);
      mlir::util::PackOpAdaptor adaptor(operands);

      auto tupleType = packOp.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), structType);
      unsigned pos = 0;
      for (auto val : adaptor.vals()) {
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), tpl, val, rewriter.getI64ArrayAttr(pos++));
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
      auto structType = convertTuple(op->getResult(0).getType().cast<TupleType>(), *typeConverter);
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, structType);
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
      mlir::util::SetTupleOpAdaptor adaptor(operands, op->getAttrDictionary());
      auto structType = convertTuple(op->getResult(0).getType().cast<TupleType>(), *typeConverter);
      rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, structType, adaptor.tuple(), adaptor.val(), rewriter.getI64ArrayAttr(adaptor.offset()));
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
      mlir::util::GetTupleOpAdaptor adaptor(operands);
      auto resType = typeConverter->convertType(getTupleOp.val().getType());
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, resType, adaptor.tuple(), rewriter.getI64ArrayAttr(getTupleOp.offset()));
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
      Type t = typeConverter->convertType(sizeOfOp.type());
      const DataLayout* layout = &defaultLayout;
      if (const DataLayoutAnalysis* analysis = llvmTypeConverter.getDataLayoutAnalysis()) {
         layout = &analysis->getAbove(op);
      }
      size_t typeSize = layout->getTypeSize(t);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(typeSize));
      return success();
   }
};
class DimOpLowering : public ConversionPattern {
   public:
   explicit DimOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::DimOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto idxType = typeConverter->convertType(IndexType::get(rewriter.getContext()));
      mlir::util::DimOpAdaptor adaptor(operands);
      auto dimOp = mlir::dyn_cast_or_null<mlir::util::DimOp>(op);
      auto genericMemrefType = dimOp.ref().getType().cast<mlir::util::RefType>();
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      Value size;
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         size = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), idxType, operands[0], rewriter.getI64ArrayAttr({1}));
      } else {
         size = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
      }
      if (!(elemType.isa<mlir::IntegerType>() && elemType.cast<mlir::IntegerType>().getWidth() == 8)) {
         Value elementSize = rewriter.create<util::SizeOfOp>(op->getLoc(), idxType, elemType); //Todo: needs fixing
         size = rewriter.create<LLVM::UDivOp>(op->getLoc(), idxType, size, elementSize);
      }
      rewriter.replaceOp(op, size);
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
      mlir::util::UnPackOpAdaptor adaptor(operands);
      auto tupleType = unPackOp.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      unsigned pos = 0;
      std::vector<Value> values;
      for (auto type : structType.getBody()) {
         if (!unPackOp.getResult(pos).use_empty()) {
            values.push_back(rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), type, adaptor.tuple(), rewriter.getI64ArrayAttr(pos++)));
         } else {
            values.push_back(Value());
            pos++;
         }
      }
      rewriter.replaceOp(op, values);
      return success();
   }
};
mlir::Value getPtrFromGenericMemref(mlir::Location loc, mlir::util::RefType genericMemrefType, mlir::Value memref, OpBuilder builder, TypeConverter* typeConverter) {
   if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      return builder.create<LLVM::ExtractValueOp>(loc, elemPtrType, memref, builder.getI64ArrayAttr(0));
   } else {
      return memref;
   }
}

class ToGenericMemrefOpLowering : public ConversionPattern {
   public:
   explicit ToGenericMemrefOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::ToGenericMemrefOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto* context = getContext();
      auto toGenericMemrefOp = cast<mlir::util::ToGenericMemrefOp>(op);
      auto genericMemrefType = toGenericMemrefOp.ref().getType().cast<mlir::util::RefType>();
      auto targetType = typeConverter->convertType(genericMemrefType);
      auto i8PointerType = mlir::LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType);
      auto idxType = typeConverter->convertType(IndexType::get(context));
      Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), i8PointerType, operands[0],
                                                               rewriter.getI64ArrayAttr(1));
      Value elementPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), elemPtrType, alignedPtr);
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         Value size = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), idxType, operands[0], rewriter.getI64ArrayAttr({3, 0}));
         Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, elementPtr, rewriter.getI64ArrayAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({1}));
         rewriter.replaceOp(op, tpl);

      } else {
         rewriter.replaceOp(op, elementPtr);
      }
      return success();
   }
};
class ToMemrefOpLowering : public ConversionPattern {
   public:
   explicit ToMemrefOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::ToMemrefOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto* context = getContext();
      mlir::util::ToMemrefOpAdaptor adaptor(operands);
      auto toMemrefOp = cast<mlir::util::ToMemrefOp>(op);
      auto memrefType = toMemrefOp.memref().getType().cast<MemRefType>();
      auto genericMemrefType = toMemrefOp.ref().getType().cast<mlir::util::RefType>();

      auto targetType = typeConverter->convertType(memrefType);

      auto targetPointerType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(memrefType.getElementType()));
      auto idxType = typeConverter->convertType(IndexType::get(context));
      Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);

      Value elementPtr = getPtrFromGenericMemref(op->getLoc(), genericMemrefType, adaptor.ref(), rewriter, typeConverter);
      auto offset = rewriter.create<arith::ConstantOp>(op->getLoc(), idxType, rewriter.getIndexAttr(0));
      Value deadBeefConst = rewriter.create<arith::ConstantOp>(op->getLoc(), idxType, rewriter.getIndexAttr(0xdeadbeef));
      auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), targetPointerType, deadBeefConst);

      Value alignedPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), targetPointerType, elementPtr);
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, allocatedPtr, rewriter.getI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, alignedPtr, rewriter.getI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, offset, rewriter.getI64ArrayAttr(2));
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1 && memrefType.getRank() == 1 && memrefType.isDynamicDim(0)) {
         Value size = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), idxType, adaptor.ref(), rewriter.getI64ArrayAttr({1}));
         Value stride = rewriter.create<arith::ConstantOp>(op->getLoc(), idxType, rewriter.getIndexAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({3, 0}));
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, stride, rewriter.getI64ArrayAttr({4, 0}));
      }
      rewriter.replaceOp(op, tpl);
      return success();
   }
};
class IsRefValidOpLowering : public ConversionPattern {
   public:
   explicit IsRefValidOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::IsRefValidOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto casted = mlir::cast<mlir::util::IsRefValidOp>(op);
      auto ptr = getPtrFromGenericMemref(op->getLoc(), casted.ref().getType().cast<mlir::util::RefType>(), operands[0], rewriter, typeConverter);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::ne, ptr, rewriter.create<mlir::LLVM::NullOp>(op->getLoc(), ptr.getType()));
      return success();
   }
};
class InvalidRefOpLowering : public ConversionPattern {
   public:
   explicit InvalidRefOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::InvalidRefOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto casted = mlir::cast<mlir::util::InvalidRefOp>(op);
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(op, typeConverter->convertType(casted.getType()));
      return success();
   }
};
class AllocaOpLowering : public ConversionPattern {
   public:
   explicit AllocaOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::AllocaOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto allocOp = cast<mlir::util::AllocaOp>(op);
      auto loc = allocOp->getLoc();

      auto genericMemrefType = allocOp.ref().getType().cast<mlir::util::RefType>();
      Value entries;
      if (allocOp.size()) {
         entries = allocOp.size();
      } else {
         int64_t staticSize = 1;
         if (genericMemrefType.getSize()) {
            staticSize = genericMemrefType.getSize().getValue();
            assert(staticSize > 0);
         }
         entries = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(staticSize));
      }
      auto bytesPerEntry = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), genericMemrefType.getElementType());
      Value sizeInBytes = rewriter.create<mlir::arith::MulIOp>(loc, rewriter.getIndexType(), entries, bytesPerEntry);
      Value sizeInBytesI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeInBytes);

      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      mlir::Value allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(
         loc, elemPtrType, sizeInBytesI64, 0);
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         assert(false && "not supported");
         return failure();
      } else {
         rewriter.replaceOp(op, allocatedElementPtr);
      }
      return success();
   }
};
class AllocOpLowering : public ConversionPattern {
   public:
   explicit AllocOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::AllocOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto allocOp = cast<mlir::util::AllocOp>(op);
      auto loc = allocOp->getLoc();

      auto genericMemrefType = allocOp.ref().getType().cast<mlir::util::RefType>();
      Value entries;
      if (allocOp.size()) {
         entries = allocOp.size();
      } else {
         int64_t staticSize = 1;
         if (genericMemrefType.getSize()) {
            staticSize = genericMemrefType.getSize().getValue();
            assert(staticSize > 0);
         }
         entries = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(staticSize));
      }

      auto bytesPerEntry = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), genericMemrefType.getElementType());
      Value sizeInBytes = rewriter.create<mlir::arith::MulIOp>(loc, rewriter.getIndexType(), entries, bytesPerEntry);
      Value sizeInBytesI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeInBytes);

      LLVM::LLVMFuncOp mallocFunc = LLVM::lookupOrCreateMallocFn(allocOp->getParentOfType<ModuleOp>(), rewriter.getI64Type());
      auto results = createLLVMCall(rewriter, loc, mallocFunc, {sizeInBytesI64},
                                    LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)));
      mlir::Value castedPointer = rewriter.create<LLVM::BitcastOp>(
         op->getLoc(), LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType())), results[0]);
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         auto targetType = typeConverter->convertType(genericMemrefType);
         Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, castedPointer, rewriter.getI64ArrayAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, sizeInBytesI64, rewriter.getI64ArrayAttr({1}));
         rewriter.replaceOp(op, tpl);
      } else {
         rewriter.replaceOp(op, castedPointer);
      }
      return success();
   }
};
class DeAllocOpLowering : public ConversionPattern {
   public:
   explicit DeAllocOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, util::DeAllocOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::DeAllocOpAdaptor dea(operands);

      auto deAllocOp = cast<util::DeAllocOp>(op);
      mlir::util::DeAllocOpAdaptor adaptor(operands);
      auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
      auto genericMemrefType = deAllocOp.ref().getType().cast<mlir::util::RefType>();

      Value casted = rewriter.create<LLVM::BitcastOp>(
         op->getLoc(), LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)),
         getPtrFromGenericMemref(op->getLoc(), genericMemrefType, adaptor.ref(), rewriter, typeConverter));
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
         op, TypeRange(), SymbolRefAttr::get(freeFunc), casted); //todo check if working
      return success();
   }
};

class StoreOpLowering : public ConversionPattern {
   public:
   explicit StoreOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::StoreOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::StoreOpAdaptor storeOpAdaptor(operands);
      auto storeOp = cast<mlir::util::StoreOp>(op);
      auto genericMemrefType = storeOp.ref().getType().cast<mlir::util::RefType>();
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      Value elementPtr = getPtrFromGenericMemref(op->getLoc(), genericMemrefType, storeOpAdaptor.ref(), rewriter, typeConverter);
      if (storeOpAdaptor.idx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elemPtrType, elementPtr, storeOpAdaptor.idx());
      }
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, storeOpAdaptor.val(), elementPtr);
      return success();
   }
};
class LoadOpLowering : public ConversionPattern {
   public:
   explicit LoadOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::LoadOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::LoadOpAdaptor loadOpAdaptor(operands);
      auto storeOp = cast<mlir::util::LoadOp>(op);
      auto genericMemrefType = storeOp.ref().getType().cast<mlir::util::RefType>();
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      Value elementPtr = getPtrFromGenericMemref(op->getLoc(), genericMemrefType, loadOpAdaptor.ref(), rewriter, typeConverter);
      if (loadOpAdaptor.idx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elemPtrType, elementPtr, loadOpAdaptor.idx());
      }
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elementPtr);
      return success();
   }
};
class CastOpLowering : public ConversionPattern {
   public:
   explicit CastOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::GenericMemrefCastOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::GenericMemrefCastOpAdaptor adaptor(operands);
      auto castedOp = cast<mlir::util::GenericMemrefCastOp>(op);
      auto sourceRefType = castedOp.val().getType().cast<mlir::util::RefType>();
      auto targetRefType = castedOp.res().getType().cast<mlir::util::RefType>();
      auto dynSizeSource = sourceRefType.getSize() && sourceRefType.getSize().getValue() == -1;
      auto dynSizeTarget = targetRefType.getSize() && targetRefType.getSize().getValue() == -1;
      if (!dynSizeSource && dynSizeTarget) {
         assert(false && "can not cast");
         return failure();
      }
      auto targetElemType = typeConverter->convertType(targetRefType.getElementType());
      if ((dynSizeSource && !dynSizeTarget)) {
         auto sourceElemType = typeConverter->convertType(sourceRefType.getElementType());
         auto sourcePointerType = mlir::LLVM::LLVMPointerType::get(sourceElemType);

         Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), sourcePointerType, adaptor.val(), rewriter.getI64ArrayAttr(0));
         Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), alignedPtr);
         rewriter.replaceOp(op, casted);
      } else if (dynSizeSource) {
         auto sourceElemType = typeConverter->convertType(sourceRefType.getElementType());
         auto sourcePointerType = mlir::LLVM::LLVMPointerType::get(sourceElemType);

         auto targetType = typeConverter->convertType(targetRefType);
         auto idxType = typeConverter->convertType(rewriter.getIndexType());
         Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), sourcePointerType, adaptor.val(), rewriter.getI64ArrayAttr(0));
         Value size = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), idxType, adaptor.val(), rewriter.getI64ArrayAttr({1}));
         Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), alignedPtr);
         Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, casted, rewriter.getI64ArrayAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({1}));
         rewriter.replaceOp(op, tpl);

      } else {
         Value casted = rewriter.create<LLVM::BitcastOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), adaptor.val());
         rewriter.replaceOp(op, casted);
      }
      return success();
   }
};
class TupleElementPtrOpLowering : public ConversionPattern {
   public:
   explicit TupleElementPtrOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::TupleElementPtrOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::TupleElementPtrOpAdaptor elementPtrOpAdaptor(operands);
      auto elementPtrOp = cast<mlir::util::TupleElementPtrOp>(op);
      auto genericMemrefType = elementPtrOp.ref().getType().cast<mlir::util::RefType>();
      auto targetMemrefType = elementPtrOp.getType().cast<mlir::util::RefType>();

      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(targetMemrefType.getElementType()));

      Value elementPtr1 = getPtrFromGenericMemref(op->getLoc(), genericMemrefType, elementPtrOpAdaptor.ref(), rewriter, typeConverter);
      Value zero = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value structIdx = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(elementPtrOp.idx()));

      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, elementPtr1, ValueRange({zero, structIdx}));
      rewriter.replaceOp(op, elementPtr);

      return success();
   }
};
class ArrayElementPtrOpLowering : public ConversionPattern {
   public:
   explicit ArrayElementPtrOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::ArrayElementPtrOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::ArrayElementPtrOpAdaptor elementPtrOpAdaptor(operands);
      auto elementPtrOp = cast<mlir::util::ArrayElementPtrOp>(op);
      auto genericMemrefType = elementPtrOp.ref().getType().cast<mlir::util::RefType>();
      auto targetMemrefType = elementPtrOp.getType().cast<mlir::util::RefType>();

      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(targetMemrefType.getElementType()));

      Value elementPtr1 = getPtrFromGenericMemref(op->getLoc(), genericMemrefType, elementPtrOpAdaptor.ref(), rewriter, typeConverter);

      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, elementPtr1, elementPtrOpAdaptor.idx());
      rewriter.replaceOp(op, elementPtr);

      return success();
   }
};

class CreateVarLenLowering : public ConversionPattern {
   public:
   explicit CreateVarLenLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::CreateVarLen::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto* context = getContext();
      mlir::util::CreateVarLenAdaptor adaptor(operands);
      Value lazymask = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0x80000000));
      Value lazylen = rewriter.create<mlir::LLVM::OrOp>(op->getLoc(), lazymask, adaptor.len());
      Value asI64 = rewriter.create<mlir::LLVM::ZExtOp>(op->getLoc(), rewriter.getI64Type(), lazylen);
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type());
      auto structType = LLVM::LLVMStructType::getLiteral(context, std::vector<Type>{rewriter.getI64Type(), rewriter.getI64Type()});
      auto types = std::vector<Type>{elemPtrType, rewriter.getI32Type()};
      Value rawPtr = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), rewriter.getI64Type(), adaptor.ref());
      Value result = rewriter.create<LLVM::UndefOp>(op->getLoc(), structType);
      result = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), result, asI64, rewriter.getI64ArrayAttr(0));
      result = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), result, rawPtr, rewriter.getI64ArrayAttr(1));
      rewriter.replaceOp(op, result);
      return success();
   }
};
class CreateConstVarLenLowering : public ConversionPattern {
   public:
   explicit CreateConstVarLenLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::CreateConstVarLen::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto* context = getContext();
      auto createStrConstOp = mlir::cast<mlir::util::CreateConstVarLen>(op);
      size_t len = createStrConstOp.str().size();
      auto structType = LLVM::LLVMStructType::getLiteral(context, std::vector<Type>{rewriter.getI64Type(), rewriter.getI64Type()});
      Value result = rewriter.create<LLVM::UndefOp>(op->getLoc(), structType);

      uint64_t first4 = 0;
      memcpy(&first4, createStrConstOp.str().data(), std::min(4ul, len));
      size_t c1 = (first4 << 32) | len;
      auto p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(c1));
      result = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), result, p1, rewriter.getI64ArrayAttr(0));
      if (len <= 12) {
         uint64_t last8 = 0;
         if (len > 4) {
            memcpy(&last8, createStrConstOp.str().data() + 4, std::min(8ul, len - 4));
         }
         size_t c2 = last8;
         auto p2 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(c2));

         result = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), result, p2, rewriter.getI64ArrayAttr(1));
         rewriter.replaceOp(op, result);
      } else {
         static size_t globalStrConstId = 0;
         mlir::LLVM::GlobalOp globalOp;
         {
            std::string name = "global_str_const_" + std::to_string(globalStrConstId++);
            auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            globalOp = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(), len), true, mlir::LLVM::Linkage::Private, name, createStrConstOp.strAttr());
         }
         auto ptr = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), globalOp);
         Value rawPtr = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), rewriter.getI64Type(), ptr);
         result = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), result, rawPtr, rewriter.getI64ArrayAttr(1));
         rewriter.replaceOp(op, result);
      }
      return success();
   }
};
class VarLenGetRefLowering : public ConversionPattern {
   public:
   explicit VarLenGetRefLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::VarLenGetRef::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto* context = getContext();
      mlir::util::VarLenGetRefAdaptor adaptor(operands);
      auto structType = LLVM::LLVMStructType::getLiteral(context, std::vector<Type>{rewriter.getI64Type(), rewriter.getI64Type()});

      auto resPtrType = mlir::LLVM::LLVMPointerType::get(structType);

      mlir::Value allocatedElementPtr;
      {
         auto func = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
         OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(&func.getBody().front());
         auto const1 = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
         allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(op->getLoc(), mlir::LLVM::LLVMPointerType::get(structType), const1, 16);
      }
      rewriter.create<LLVM::StoreOp>(op->getLoc(), adaptor.varlen(), allocatedElementPtr);

      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type());
      auto idxType = typeConverter->convertType(IndexType::get(context));
      auto resType = LLVM::LLVMStructType::getLiteral(context, std::vector<Type>{elemPtrType, idxType});
      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "rt_varlen_to_ref", resPtrType, resType);
      auto result = createLLVMCall(rewriter, op->getLoc(), fn, {allocatedElementPtr}, resType)[0];
      rewriter.replaceOp(op, result);
      return success();
   }
};

class VarLenGetLenLowering : public ConversionPattern {
   public:
   explicit VarLenGetLenLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::VarLenGetLen::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::VarLenGetLenAdaptor adaptor(operands);
      Value len = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), rewriter.getI64Type(), adaptor.varlen(), rewriter.getI64ArrayAttr(0));
      Value mask = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x7FFFFFFF));
      Value castedLen = rewriter.create<LLVM::AndOp>(op->getLoc(), len, mask);

      rewriter.replaceOp(op, castedLen);
      return success();
   }
};
class Hash64Lowering : public ConversionPattern {
   public:
   explicit Hash64Lowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::Hash64::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::Hash64Adaptor adaptor(operands);
      Value p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(11400714819323198549ull));
      Value m1 = rewriter.create<LLVM::MulOp>(op->getLoc(), p1, adaptor.val());
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), m1);
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), m1, reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashCombineLowering : public ConversionPattern {
   public:
   explicit HashCombineLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::HashCombine::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::HashCombineAdaptor adaptor(operands);
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), adaptor.h1());
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), adaptor.h2(), reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashVarLenLowering : public ConversionPattern {
   public:
   explicit HashVarLenLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::HashVarLen::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::HashVarLenAdaptor adaptor(operands);
      Value a = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), rewriter.getI64Type(), adaptor.val(), rewriter.getI64ArrayAttr(0));
      Value b = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), rewriter.getI64Type(), adaptor.val(), rewriter.getI64ArrayAttr(1));

      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "rt_hash_varlen", {rewriter.getI64Type(), rewriter.getI64Type()}, rewriter.getI64Type());
      auto result = createLLVMCall(rewriter, op->getLoc(), fn, {a, b}, rewriter.getI64Type())[0];
      rewriter.replaceOp(op, result);
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
   typeConverter.addConversion([&](mlir::util::RefType genericMemrefType) {
      MLIRContext* context = &typeConverter.getContext();
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter.convertType(genericMemrefType.getElementType()));
      auto idxType = typeConverter.convertType(IndexType::get(context));
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         std::vector<Type> types = {elemPtrType, idxType};
         return (Type) mlir::LLVM::LLVMStructType::getLiteral(context, types);
      } else {
         return (Type) elemPtrType;
      }
   });
   typeConverter.addConversion([&](mlir::util::VarLen32Type varLen32Type) {
      MLIRContext* context = &typeConverter.getContext();
      return LLVM::LLVMStructType::getLiteral(context, std::vector<Type>{IntegerType::get(context, 64), IntegerType::get(context, 64)});
   });
   patterns.add<CastOpLowering>(typeConverter, patterns.getContext());
   patterns.add<DimOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SizeOfOpLowering>(typeConverter, patterns.getContext());
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<PackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UnPackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocaOpLowering>(typeConverter, patterns.getContext());
   patterns.add<DeAllocOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ArrayElementPtrOpLowering>(typeConverter, patterns.getContext());
   patterns.add<TupleElementPtrOpLowering>(typeConverter, patterns.getContext());

   patterns.add<ToGenericMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ToMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<IsRefValidOpLowering>(typeConverter, patterns.getContext());
   patterns.add<InvalidRefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<StoreOpLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadOpLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateConstVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenGetRefLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenGetLenLowering>(typeConverter, patterns.getContext());
   patterns.add<HashCombineLowering>(typeConverter, patterns.getContext());
   patterns.add<Hash64Lowering>(typeConverter, patterns.getContext());
   patterns.add<HashVarLenLowering>(typeConverter, patterns.getContext());
}
