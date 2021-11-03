#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/util/UtilTypes.h"
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
class SizeOfOpLowering : public ConversionPattern {
   public:
   explicit SizeOfOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::SizeOfOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto sizeOfOp = mlir::dyn_cast_or_null<mlir::util::SizeOfOp>(op);
      Value const1 = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
      Type ptrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(sizeOfOp.type()));
      Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(rewriter.getUnknownLoc(), ptrType);
      Value size1 = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(), ptrType, nullPtr, const1);
      rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, rewriter.getI64Type(), size1);
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
         size = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, operands[0], rewriter.getI64ArrayAttr({1}));
      } else {
         size = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
      }
      if (!(elemType.isa<mlir::IntegerType>() && elemType.cast<mlir::IntegerType>().getWidth() == 8)) {
         Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), idxType, elemType); //Todo: needs fixing
         size = rewriter.create<LLVM::UDivOp>(rewriter.getUnknownLoc(), idxType, size, elementSize);
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
mlir::Value getPtrFromGenericMemref(mlir::util::RefType genericMemrefType, mlir::Value memref, OpBuilder builder, TypeConverter* typeConverter) {
   if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      return builder.create<LLVM::ExtractValueOp>(builder.getUnknownLoc(), elemPtrType, memref,
                                                  builder.getI64ArrayAttr(0));
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
      Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), i8PointerType, operands[0],
                                                               rewriter.getI64ArrayAttr(1));
      Value elementPtr = rewriter.create<LLVM::BitcastOp>(rewriter.getUnknownLoc(), elemPtrType, alignedPtr);
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         Value size = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, operands[0], rewriter.getI64ArrayAttr({3, 0}));
         Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), targetType);
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, elementPtr, rewriter.getI64ArrayAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({1}));
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

      auto i8PointerType = mlir::LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto idxType = typeConverter->convertType(IndexType::get(context));
      Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), targetType);

      Value elementPtr = getPtrFromGenericMemref(genericMemrefType, adaptor.ref(), rewriter, typeConverter);
      auto offset = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), idxType, rewriter.getIndexAttr(0));
      Value deadBeefConst = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), idxType, rewriter.getIndexAttr(0xdeadbeef));
      auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)), deadBeefConst);

      Value alignedPtr = rewriter.create<LLVM::BitcastOp>(rewriter.getUnknownLoc(), i8PointerType, elementPtr);
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, allocatedPtr, rewriter.getI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, alignedPtr, rewriter.getI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, offset, rewriter.getI64ArrayAttr(2));
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1 && memrefType.getRank() == 1 && memrefType.isDynamicDim(0)) {
         Value size = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, adaptor.ref(), rewriter.getI64ArrayAttr({1}));
         Value stride = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), idxType, rewriter.getIndexAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({3, 0}));
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, stride, rewriter.getI64ArrayAttr({4, 0}));
      }
      rewriter.replaceOp(op, tpl);
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
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      mlir::Value allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(
         loc, elemPtrType, sizeInBytes, 0);
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

      LLVM::LLVMFuncOp mallocFunc = LLVM::lookupOrCreateMallocFn(allocOp->getParentOfType<ModuleOp>(), rewriter.getI64Type());
      auto results = createLLVMCall(rewriter, loc, mallocFunc, {sizeInBytes},
                                    LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)));
      mlir::Value castedPointer = rewriter.create<LLVM::BitcastOp>(
         op->getLoc(), LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType())), results[0]);
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         auto targetType = typeConverter->convertType(genericMemrefType);
         Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), targetType);
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, castedPointer, rewriter.getI64ArrayAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, sizeInBytes, rewriter.getI64ArrayAttr({1}));
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
         getPtrFromGenericMemref(genericMemrefType, adaptor.ref(), rewriter, typeConverter));
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
      Value elementPtr = getPtrFromGenericMemref(genericMemrefType, storeOpAdaptor.ref(), rewriter, typeConverter);
      if (storeOpAdaptor.idx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(), elemPtrType, elementPtr, storeOpAdaptor.idx());
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
      Value elementPtr = getPtrFromGenericMemref(genericMemrefType, loadOpAdaptor.ref(), rewriter, typeConverter);
      if (loadOpAdaptor.idx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(), elemPtrType, elementPtr, loadOpAdaptor.idx());
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

         Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), sourcePointerType, adaptor.val(), rewriter.getI64ArrayAttr(0));
         Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), alignedPtr);
         rewriter.replaceOp(op, casted);
      } else if (dynSizeSource) {
         auto sourceElemType = typeConverter->convertType(sourceRefType.getElementType());
         auto sourcePointerType = mlir::LLVM::LLVMPointerType::get(sourceElemType);

         auto targetType = typeConverter->convertType(targetRefType);
         auto idxType = typeConverter->convertType(rewriter.getIndexType());
         Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), sourcePointerType, adaptor.val(), rewriter.getI64ArrayAttr(0));
         Value size = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, adaptor.val(), rewriter.getI64ArrayAttr({1}));
         Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), alignedPtr);
         Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), targetType);
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, casted, rewriter.getI64ArrayAttr(0));
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({1}));
         rewriter.replaceOp(op, tpl);

      } else {
         Value casted = rewriter.create<LLVM::BitcastOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), adaptor.val());
         rewriter.replaceOp(op, casted);
      }
      return success();
   }
};
class ElementPtrOpLowering : public ConversionPattern {
   public:
   explicit ElementPtrOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::ElementPtrOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::ElementPtrOpAdaptor elementPtrOpAdaptor(operands);
      auto elementPtrOp = cast<mlir::util::ElementPtrOp>(op);
      auto genericMemrefType = elementPtrOp.ref().getType().cast<mlir::util::RefType>();
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      Value elementPtr1 = getPtrFromGenericMemref(genericMemrefType, elementPtrOpAdaptor.ref(), rewriter, typeConverter);
      Value elementPtr = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(), elemPtrType, elementPtr1, elementPtrOp.idx());
      rewriter.replaceOp(op, elementPtr);
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
   /*typeConverter.addConversion([&](mlir::util::BufferType bufferType) {
      return TupleType::get(&typeConverter.getContext(), TypeRange({MemRefType::get({-1}, bufferType.getElementType()), IndexType::get(&typeConverter.getContext())}));
   });*/
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
   typeConverter.addSourceMaterialization([&](OpBuilder&, TupleType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, TupleType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, RefType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, RefType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addSourceMaterialization([&](OpBuilder&, IndexType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, IndexType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
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
   patterns.add<ElementPtrOpLowering>(typeConverter, patterns.getContext());

   patterns.add<ToGenericMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ToMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<StoreOpLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadOpLowering>(typeConverter, patterns.getContext());
}
