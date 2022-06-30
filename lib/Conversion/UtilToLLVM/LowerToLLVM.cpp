#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

class PackOpLowering : public OpConversionPattern<mlir::util::PackOp> {
   public:
   using OpConversionPattern<mlir::util::PackOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::PackOp packOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto tupleType = packOp.tuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::UndefOp>(packOp->getLoc(), structType);
      unsigned pos = 0;
      for (auto val : adaptor.vals()) {
         tpl = rewriter.create<LLVM::InsertValueOp>(packOp->getLoc(), tpl, val, rewriter.getI64ArrayAttr(pos++));
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
   using OpConversionPattern<mlir::util::GetTupleOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::GetTupleOp getTupleOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto resType = typeConverter->convertType(getTupleOp.val().getType());
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(getTupleOp, resType, adaptor.tuple(), rewriter.getI64ArrayAttr(getTupleOp.offset()));
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

class ToGenericMemrefOpLowering : public OpConversionPattern<mlir::util::ToGenericMemrefOp> {
   public:
   using OpConversionPattern<mlir::util::ToGenericMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::ToGenericMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto* context = getContext();
      auto genericMemrefType = op.ref().getType().cast<mlir::util::RefType>();
      auto i8PointerType = mlir::LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType);
      Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), i8PointerType, adaptor.memref(), rewriter.getI64ArrayAttr(1));
      Value elementPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), elemPtrType, alignedPtr);
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class ToMemrefOpLowering : public OpConversionPattern<mlir::util::ToMemrefOp> {
   public:
   using OpConversionPattern<mlir::util::ToMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::ToMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto memrefType = op.memref().getType().cast<MemRefType>();

      auto targetType = typeConverter->convertType(memrefType);

      auto targetPointerType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(memrefType.getElementType()));
      Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);

      Value elementPtr = adaptor.ref();
      auto offset = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value deadBeefConst = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xdeadbeef));
      auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), targetPointerType, deadBeefConst);

      Value alignedPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), targetPointerType, elementPtr);
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, allocatedPtr, rewriter.getI64ArrayAttr(0));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, alignedPtr, rewriter.getI64ArrayAttr(1));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, offset, rewriter.getI64ArrayAttr(2));
      rewriter.replaceOp(op, tpl);
      return success();
   }
};
class IsRefValidOpLowering : public OpConversionPattern<mlir::util::IsRefValidOp> {
   public:
   using OpConversionPattern<mlir::util::IsRefValidOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::IsRefValidOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::ne, adaptor.ref(), rewriter.create<mlir::LLVM::NullOp>(op->getLoc(), adaptor.ref().getType()));
      return success();
   }
};
class InvalidRefOpLowering : public OpConversionPattern<mlir::util::InvalidRefOp> {
   public:
   using OpConversionPattern<mlir::util::InvalidRefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::InvalidRefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(op, typeConverter->convertType(op.getType()));
      return success();
   }
};
class AllocaOpLowering : public OpConversionPattern<mlir::util::AllocaOp> {
   public:
   using OpConversionPattern<mlir::util::AllocaOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::AllocaOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();
      auto genericMemrefType = allocOp.ref().getType().cast<mlir::util::RefType>();
      Value entries;
      if (allocOp.size()) {
         entries = allocOp.size();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }
      auto bytesPerEntry = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getI64Type(), genericMemrefType.getElementType());
      Value sizeInBytes = rewriter.create<mlir::arith::MulIOp>(loc, rewriter.getI64Type(), entries, bytesPerEntry);
      Value sizeInBytesI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeInBytes);

      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()));
      mlir::Value allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(loc, elemPtrType, sizeInBytesI64, 0);
      rewriter.replaceOp(allocOp, allocatedElementPtr);

      return success();
   }
};
class AllocOpLowering : public OpConversionPattern<mlir::util::AllocOp> {
   public:
   using OpConversionPattern<mlir::util::AllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::AllocOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();

      auto genericMemrefType = allocOp.ref().getType().cast<mlir::util::RefType>();
      Value entries;
      if (allocOp.size()) {
         entries = allocOp.size();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }

      auto bytesPerEntry = rewriter.create<mlir::util::SizeOfOp>(loc, rewriter.getI64Type(), genericMemrefType.getElementType());
      Value sizeInBytes = rewriter.create<mlir::arith::MulIOp>(loc, rewriter.getI64Type(), entries, bytesPerEntry);
      Value sizeInBytesI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeInBytes);

      LLVM::LLVMFuncOp mallocFunc = LLVM::lookupOrCreateMallocFn(allocOp->getParentOfType<ModuleOp>(), rewriter.getI64Type());
      auto results = createLLVMCall(rewriter, loc, mallocFunc, {sizeInBytesI64},
                                    LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)));
      mlir::Value castedPointer = rewriter.create<LLVM::BitcastOp>(loc, LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType())), results[0]);
      rewriter.replaceOp(allocOp, castedPointer);

      return success();
   }
};
class DeAllocOpLowering : public OpConversionPattern<mlir::util::DeAllocOp> {
   public:
   using OpConversionPattern<mlir::util::DeAllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::DeAllocOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
      Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)), adaptor.ref());
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange(), SymbolRefAttr::get(freeFunc), casted);
      return success();
   }
};

class StoreOpLowering : public OpConversionPattern<mlir::util::StoreOp> {
   public:
   using OpConversionPattern<mlir::util::StoreOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.ref();
      if (adaptor.idx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), elementPtr, adaptor.idx());
      }
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.val(), elementPtr);
      return success();
   }
};
class LoadOpLowering : public OpConversionPattern<mlir::util::LoadOp> {
   public:
   using OpConversionPattern<mlir::util::LoadOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.ref();
      if (adaptor.idx()) {
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), elementPtr, adaptor.idx());
      }
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elementPtr);
      return success();
   }
};
class CastOpLowering : public OpConversionPattern<mlir::util::GenericMemrefCastOp> {
   public:
   using OpConversionPattern<mlir::util::GenericMemrefCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::GenericMemrefCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetRefType = op.res().getType().cast<mlir::util::RefType>();
      auto targetElemType = typeConverter->convertType(targetRefType.getElementType());
      Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(targetElemType), adaptor.val());
      rewriter.replaceOp(op, casted);
      return success();
   }
};
class TupleElementPtrOpLowering : public OpConversionPattern<mlir::util::TupleElementPtrOp> {
   public:
   using OpConversionPattern<mlir::util::TupleElementPtrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::TupleElementPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetMemrefType = op.getType().cast<mlir::util::RefType>();
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(targetMemrefType.getElementType()));
      Value zero = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value structIdx = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(op.idx()));
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, adaptor.ref(), ValueRange({zero, structIdx}));
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
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, adaptor.ref(), adaptor.idx());
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};

class CreateVarLenLowering : public OpConversionPattern<mlir::util::CreateVarLen> {
   public:
   using OpConversionPattern<mlir::util::CreateVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::CreateVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Type i128Ty = rewriter.getIntegerType(128);
      Value lazymask = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0x80000000));
      Value lazylen = rewriter.create<mlir::LLVM::OrOp>(op->getLoc(), lazymask, adaptor.len());
      Value asI128 = rewriter.create<mlir::LLVM::ZExtOp>(op->getLoc(), i128Ty, lazylen);
      Value rawPtr = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), i128Ty, adaptor.ref());
      auto const64 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
      auto shlPtr = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), rawPtr, const64);
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, asI128, shlPtr);
      return success();
   }
};
class CreateConstVarLenLowering : public OpConversionPattern<mlir::util::CreateConstVarLen> {
   public:
   using OpConversionPattern<mlir::util::CreateConstVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::CreateConstVarLen op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      size_t len = op.str().size();

      mlir::Type i128Ty = rewriter.getIntegerType(128);
      mlir::Value p1, p2;

      uint64_t first4 = 0;
      memcpy(&first4, op.str().data(), std::min(4ul, len));
      size_t c1 = (first4 << 32) | len;
      p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, c1));
      if (len <= 12) {
         uint64_t last8 = 0;
         if (len > 4) {
            memcpy(&last8, op.str().data() + 4, std::min(8ul, len - 4));
         }
         p2 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, last8));
      } else {
         static size_t globalStrConstId = 0;
         mlir::LLVM::GlobalOp globalOp;
         {
            std::string name = "global_str_const_" + std::to_string(globalStrConstId++);
            auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            globalOp = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(), len), true, mlir::LLVM::Linkage::Private, name, op.strAttr());
         }
         auto ptr = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), globalOp);
         p2 = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), i128Ty, ptr);
      }
      auto const64 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
      auto shlp2 = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), p2, const64);
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, p1, shlp2);
      return success();
   }
};

class VarLenGetLenLowering : public OpConversionPattern<mlir::util::VarLenGetLen> {
   public:
   using OpConversionPattern<mlir::util::VarLenGetLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::VarLenGetLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value len = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), adaptor.varlen());
      Value mask = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x7FFFFFFF));
      Value castedLen = rewriter.create<LLVM::AndOp>(op->getLoc(), len, mask);

      rewriter.replaceOp(op, castedLen);
      return success();
   }
};
class Hash64Lowering : public OpConversionPattern<mlir::util::Hash64> {
   public:
   using OpConversionPattern<mlir::util::Hash64>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::Hash64 op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(11400714819323198549ull));
      Value m1 = rewriter.create<LLVM::MulOp>(op->getLoc(), p1, adaptor.val());
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), m1);
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), m1, reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashCombineLowering: public OpConversionPattern<mlir::util::HashCombine> {
   public:
   using OpConversionPattern<mlir::util::HashCombine>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::HashCombine op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), adaptor.h1());
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), adaptor.h2(), reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashVarLenLowering : public OpConversionPattern<mlir::util::HashVarLen> {
   public:
   using OpConversionPattern<mlir::util::HashVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::util::HashVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "hashVarLenData", {rewriter.getIntegerType(128)}, rewriter.getI64Type());
      auto result = createLLVMCall(rewriter, op->getLoc(), fn, adaptor.val(), rewriter.getI64Type())[0];
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
      Value maskedHash = rewriter.create<LLVM::AndOp>(loc, adaptor.hash(), tagMask);
      Value ptrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), adaptor.ref());
      Value maskedPtr = rewriter.create<LLVM::AndOp>(loc, ptrAsInt, ptrMask);
      maskedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, adaptor.ref().getType(), maskedPtr);
      Value ored = rewriter.create<LLVM::OrOp>(loc, ptrAsInt, maskedHash);
      Value contained = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ored, ptrAsInt);
      Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, adaptor.ref().getType());

      Value filtered = rewriter.create<LLVM::SelectOp>(loc, contained, maskedPtr, nullPtr);
      rewriter.replaceOp(op, filtered);
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
   typeConverter.addConversion([&](mlir::util::RefType genericMemrefType) -> Type {
      return mlir::LLVM::LLVMPointerType::get(typeConverter.convertType(genericMemrefType.getElementType()));
   });
   typeConverter.addConversion([&](mlir::util::VarLen32Type varLen32Type) {
      MLIRContext* context = &typeConverter.getContext();
      return IntegerType::get(context, 128);
   });
   patterns.add<CastOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SizeOfOpLowering>(typeConverter, patterns.getContext());
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<PackOpLowering>(typeConverter, patterns.getContext());
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
   patterns.add<VarLenGetLenLowering>(typeConverter, patterns.getContext());
   patterns.add<HashCombineLowering>(typeConverter, patterns.getContext());
   patterns.add<Hash64Lowering>(typeConverter, patterns.getContext());
   patterns.add<HashVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<FilterTaggedPtrLowering>(typeConverter, patterns.getContext());
}
