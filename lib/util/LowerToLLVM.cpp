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
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeTranslation.h"
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
      Value const1 = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
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
      auto genericMemrefType = dimOp.generic_memref().getType().cast<mlir::util::GenericMemrefType>();

      Value size;
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         size = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, operands[0], rewriter.getI64ArrayAttr({3, 0}));
      }else{
         size = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
      }
      rewriter.replaceOp(op,size);
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

static Value createGenericMemrefFromPtr(TypeConverter* typeConverter, Type t, Value ptr, ConversionPatternRewriter& rewriter) {
   Type targetType = typeConverter->convertType(mlir::util::GenericMemrefType::get(t.getContext(), t, llvm::Optional<int64_t>()));
   Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), targetType);
   Type idxType = rewriter.getIndexType();
   auto offset = rewriter.create<mlir::ConstantOp>(rewriter.getUnknownLoc(), idxType, rewriter.getIndexAttr(0));
   Value deadBeefConst = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), idxType, rewriter.getIndexAttr(0xdeadbeef));
   auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)), deadBeefConst);
   tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, allocatedPtr, rewriter.getI64ArrayAttr(0));
   tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, ptr, rewriter.getI64ArrayAttr(1));
   tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, offset, rewriter.getI64ArrayAttr(2));
   return tpl;
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
      auto genericMemrefType = toGenericMemrefOp.generic_memref().getType().cast<mlir::util::GenericMemrefType>();
      auto targetType = typeConverter->convertType(genericMemrefType);
      auto i8PointerType = mlir::LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType);
      auto idxType = typeConverter->convertType(IndexType::get(context));
      auto arrType = LLVM::LLVMArrayType::get(idxType, 1);
      Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), targetType);
      Value allocatedPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), i8PointerType, operands[0],
                                                                 rewriter.getI64ArrayAttr(0));
      Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), i8PointerType, operands[0],
                                                               rewriter.getI64ArrayAttr(1));
      Value offset = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, operands[0],
                                                           rewriter.getI64ArrayAttr(2));
      Value elementPtr = rewriter.create<LLVM::BitcastOp>(rewriter.getUnknownLoc(), elemPtrType, alignedPtr);
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, allocatedPtr, rewriter.getI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, elementPtr, rewriter.getI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, offset, rewriter.getI64ArrayAttr(2));
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         Value stride = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), arrType, operands[0], rewriter.getI64ArrayAttr(4));
         Value size = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, operands[0], rewriter.getI64ArrayAttr({3, 0}));
         Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), idxType, elemType);//Todo: needs fixing
         size = rewriter.create<LLVM::UDivOp>(rewriter.getUnknownLoc(), idxType, size, elementSize);
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({3, 0}));
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, stride, rewriter.getI64ArrayAttr({4}));
      }
      rewriter.replaceOp(op, tpl);
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
      auto toMemrefOp = cast<mlir::util::ToMemrefOp>(op);
      auto memrefType = toMemrefOp.memref().getType().cast<MemRefType>();
      auto genericMemrefType = toMemrefOp.generic_memref().getType().cast<mlir::util::GenericMemrefType>();

      auto targetType = typeConverter->convertType(memrefType);

      auto i8PointerType = mlir::LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType);
      auto idxType = typeConverter->convertType(IndexType::get(context));
      auto arrType = LLVM::LLVMArrayType::get(idxType, 1);
      Value tpl = rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), targetType);
      Value allocatedPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), i8PointerType, operands[0],
                                                                 rewriter.getI64ArrayAttr(0));
      Value elementPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), elemPtrType, operands[0],
                                                               rewriter.getI64ArrayAttr(1));
      Value offset = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, operands[0],
                                                           rewriter.getI64ArrayAttr(2));
      Value alignedPtr = rewriter.create<LLVM::BitcastOp>(rewriter.getUnknownLoc(), i8PointerType, elementPtr);
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, allocatedPtr, rewriter.getI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, alignedPtr, rewriter.getI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, offset, rewriter.getI64ArrayAttr(2));
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1 && memrefType.getRank() == 1 && memrefType.isDynamicDim(0)) {
         Value size = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), idxType, operands[0], rewriter.getI64ArrayAttr({3, 0}));
         Value stride = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), arrType, operands[0], rewriter.getI64ArrayAttr(4));
         Value elementSize = rewriter.create<util::SizeOfOp>(rewriter.getUnknownLoc(), idxType, elemType);
         size = rewriter.create<LLVM::MulOp>(rewriter.getUnknownLoc(), idxType, size, elementSize);
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, size, rewriter.getI64ArrayAttr({3, 0}));
         tpl = rewriter.create<LLVM::InsertValueOp>(rewriter.getUnknownLoc(), targetType, tpl, stride, rewriter.getI64ArrayAttr({4}));
      }
      rewriter.replaceOp(op, tpl);
      return success();
   }
};
template <class UtilOp, class MemrefOp>
class AllocOpLowering : public ConversionPattern {
   public:
   explicit AllocOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, UtilOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto allocOp = cast<UtilOp>(op);
      auto loc = allocOp->getLoc();

      auto genericMemrefType = allocOp.generic_memref().getType().template cast<mlir::util::GenericMemrefType>();
      auto memrefType = MemRefType::get({-1}, rewriter.getIntegerType(8));
      Value entries;
      if (allocOp.size()) {
         entries = allocOp.size();
      } else {
         int64_t staticSize = 1;
         if (genericMemrefType.getSize()) {
            staticSize = genericMemrefType.getSize().getValue();
            assert(staticSize > 0);
         }
         entries = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(staticSize));
      }
      auto bytesPerEntry = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getIndexType(), genericMemrefType.getElementType());
      Value sizeInBytes = rewriter.create<mlir::MulIOp>(loc, rewriter.getIndexType(), entries, bytesPerEntry);
      auto i8MemRef = rewriter.create<MemrefOp>(loc, memrefType, sizeInBytes);
      rewriter.replaceOpWithNewOp<mlir::util::ToGenericMemrefOp>(op, genericMemrefType, i8MemRef);
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
      auto loc = deAllocOp->getLoc();

      auto memrefType = MemRefType::get({}, rewriter.getIntegerType(8));
      auto i8MemRef = rewriter.create<util::ToMemrefOp>(loc, memrefType, deAllocOp.generic_memref());
      rewriter.replaceOpWithNewOp<mlir::memref::DeallocOp>(op, i8MemRef);
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
      auto genericMemrefType = storeOp.generic_memref().getType().cast<mlir::util::GenericMemrefType>();
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      Value elementPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), elemPtrType, storeOpAdaptor.generic_memref(),
                                                               rewriter.getI64ArrayAttr(1));
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
      auto genericMemrefType = storeOp.generic_memref().getType().cast<mlir::util::GenericMemrefType>();
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      Value elementPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), elemPtrType, loadOpAdaptor.generic_memref(),
                                                               rewriter.getI64ArrayAttr(1));
      if (loadOpAdaptor.idx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(), elemPtrType, elementPtr, loadOpAdaptor.idx());
      }
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elementPtr);
      return success();
   }
};
class MemberRefOpLowering : public ConversionPattern {
   public:
   explicit MemberRefOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::MemberRefOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::MemberRefOpAdaptor memberRefOpAdaptor(operands);
      auto memberRefOp = cast<mlir::util::MemberRefOp>(op);
      auto genericMemrefType = memberRefOp.source_ref().getType().cast<mlir::util::GenericMemrefType>();
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType);
      auto resultType = memberRefOp.result_ref().getType().cast<mlir::util::GenericMemrefType>();
      auto memberType = typeConverter->convertType(resultType.getElementType());
      auto memberPtrType = LLVM::LLVMPointerType::get(memberType);

      Value elementPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), elemPtrType, memberRefOpAdaptor.source_ref(),
                                                               rewriter.getI64ArrayAttr(1));
      Value ptrIndex;
      if (memberRefOpAdaptor.memref_idx()) {
         ptrIndex = memberRefOpAdaptor.memref_idx();
      } else {
         ptrIndex = rewriter.create<ConstantOp>(
            rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIntegerAttr(rewriter.getIndexType(), 0xdeadbeef));
      }
      Value tupleIndex = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getIntegerType(32), memberRefOp.tuple_idxAttr().getInt()));
      auto ptr = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(), memberPtrType, elementPtr, ValueRange{ptrIndex, tupleIndex});

      rewriter.replaceOp(op, createGenericMemrefFromPtr(typeConverter, memberType, ptr, rewriter));
      return success();
   }
};
class ToRawPtrOpLowering : public ConversionPattern {
   public:
   explicit ToRawPtrOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::ToRawPointerOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::ToRawPointerOpAdaptor loadOpAdaptor(operands);
      auto castedOp = cast<mlir::util::ToRawPointerOp>(op);
      auto genericMemrefType = castedOp.ref().getType().cast<mlir::util::GenericMemrefType>();
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      Value elementPtr = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), elemPtrType, loadOpAdaptor.ref(),
                                                               rewriter.getI64ArrayAttr(1));
      rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, rewriter.getIntegerType(64), elementPtr);
      return success();
   }
};
class FromRawPtrOpLowering : public ConversionPattern {
   public:
   explicit FromRawPtrOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::util::FromRawPointerOp::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      mlir::util::FromRawPointerOpAdaptor loadOpAdaptor(operands);
      auto castedOp = cast<mlir::util::FromRawPointerOp>(op);
      auto genericMemrefType = castedOp.ref().getType().cast<mlir::util::GenericMemrefType>();
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType);

      auto ptr = rewriter.create<LLVM::IntToPtrOp>(rewriter.getUnknownLoc(), elemPtrType, loadOpAdaptor.ptr());
      rewriter.replaceOp(op, createGenericMemrefFromPtr(typeConverter, elemType, ptr, rewriter));

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
   typeConverter.addConversion([&](mlir::util::GenericMemrefType genericMemrefType) {
      MLIRContext* context = &typeConverter.getContext();
      auto allocatedPtrType = mlir::LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter.convertType(genericMemrefType.getElementType()));
      auto idxType = typeConverter.convertType(IndexType::get(context));
      auto arrType = LLVM::LLVMArrayType::get(idxType, 1);
      std::vector<Type> types = {allocatedPtrType, elemPtrType, idxType};
      if (genericMemrefType.getSize() && genericMemrefType.getSize().getValue() == -1) {
         types.push_back(arrType);
         types.push_back(arrType);
      }
      return mlir::LLVM::LLVMStructType::getLiteral(context, types);
   });
   /*typeConverter.addSourceMaterialization([&](OpBuilder&, util::BufferType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](OpBuilder&, util::BufferType type, ValueRange valueRange, Location loc) {
      return valueRange.front();
   });*/
   patterns.add<DimOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SizeOfOpLowering>(typeConverter, patterns.getContext());
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<PackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UnPackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocOpLowering<util::AllocOp, memref::AllocOp>>(typeConverter, patterns.getContext());
   patterns.add<AllocOpLowering<util::AllocaOp, memref::AllocaOp>>(typeConverter, patterns.getContext());
   patterns.add<DeAllocOpLowering>(typeConverter, patterns.getContext());

   patterns.add<ToGenericMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ToMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<StoreOpLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadOpLowering>(typeConverter, patterns.getContext());
   patterns.add<MemberRefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<FromRawPtrOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ToRawPtrOpLowering>(typeConverter, patterns.getContext());
}
