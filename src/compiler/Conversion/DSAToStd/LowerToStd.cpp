#include "lingodb/compiler/Conversion/ArrowToStd/ArrowToStd.h"
#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/IR/BuiltinTypes.h>

#include "lingodb/compiler/runtime/ExecutionContext.h"

#include <lingodb/compiler/runtime/ArrowColumn.h>
using namespace mlir;
namespace arrow = lingodb::compiler::dialect::arrow;
namespace util = lingodb::compiler::dialect::util;
namespace rt = lingodb::compiler::runtime;

namespace {
struct ArrowToStdLoweringPass
   : public PassWrapper<ArrowToStdLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArrowToStdLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "lower-arrow"; }

   ArrowToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, arrow::ArrowDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect>();
   }
   void runOnOperation() final;
};
static TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      Type converted = typeConverter.convertType(t);
      converted = converted ? converted : t;
      types.push_back(converted);
   }
   return TupleType::get(tupleType.getContext(), TypeRange(types));
}
template <class Fn>
static void createAtArrayCreation(ConversionPatternRewriter& rewriter, mlir::Value array, const Fn& fn) {
   if (auto creationOp = array.getDefiningOp()) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(creationOp);
      fn(rewriter);
   } else {
      fn(rewriter);
   }
}

class ArrayLoadFixedSizedLowering : public OpConversionPattern<arrow::LoadFixedSizedOp> {
   public:
   using OpConversionPattern<arrow::LoadFixedSizedOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::LoadFixedSizedOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Value valueBuffer;
      createAtArrayCreation(rewriter, adaptor.getArray(), [&](mlir::ConversionPatternRewriter& rewriter) {
         auto bufferArrayPtr = rewriter.create<util::TupleElementPtrOp>(op.getLoc(), util::RefType::get(util::RefType::get(util::RefType::get(rewriter.getI8Type()))), adaptor.getArray(), 5);
         auto bufferArray = rewriter.create<util::LoadOp>(op.getLoc(), bufferArrayPtr);
         //load buffer with main values (offset 1)
         auto c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
         auto buffer = rewriter.create<util::LoadOp>(op.getLoc(), bufferArray, c1);
         valueBuffer = rewriter.create<util::GenericMemrefCastOp>(op.getLoc(), util::RefType::get(op.getType()), buffer);
      });

      rewriter.replaceOpWithNewOp<util::LoadOp>(op, valueBuffer, adaptor.getOffset());
      return success();
   }
};
static Value getBit(OpBuilder builder, Location loc, Value bits, Value pos) {
   auto i1Type = IntegerType::get(builder.getContext(), 1);
   auto i8Type = IntegerType::get(builder.getContext(), 8);

   auto indexType = IndexType::get(builder.getContext());
   Value const3 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 3));
   Value const7 = builder.create<arith::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 7));
   Value const1Byte = builder.create<arith::ConstantOp>(loc, i8Type, builder.getIntegerAttr(i8Type, 1));

   Value div8 = builder.create<arith::ShRUIOp>(loc, indexType, pos, const3);
   Value rem8 = builder.create<arith::AndIOp>(loc, indexType, pos, const7);
   Value loadedByte = builder.create<util::LoadOp>(loc, i8Type, bits, div8);
   Value rem8AsByte = builder.create<arith::IndexCastOp>(loc, i8Type, rem8);
   Value shifted = builder.create<arith::ShRUIOp>(loc, i8Type, loadedByte, rem8AsByte);
   Value res1 = shifted;

   Value anded = builder.create<arith::AndIOp>(loc, i8Type, res1, const1Byte);
   Value res = builder.create<arith::CmpIOp>(loc, i1Type, mlir::arith::CmpIPredicate::eq, anded, const1Byte);
   return res;
}
class ArrayLoadBoolLowering : public OpConversionPattern<arrow::LoadBoolOp> {
   public:
   using OpConversionPattern<arrow::LoadBoolOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::LoadBoolOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Value buffer;
      createAtArrayCreation(rewriter, adaptor.getArray(), [&](mlir::ConversionPatternRewriter& rewriter) {
         auto bufferArrayPtr = rewriter.create<util::TupleElementPtrOp>(op.getLoc(), util::RefType::get(util::RefType::get(util::RefType::get(rewriter.getI8Type()))), adaptor.getArray(), 5);
         auto bufferArray = rewriter.create<util::LoadOp>(op.getLoc(), bufferArrayPtr);
         //load buffer with main values (offset 1)
         auto c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
         buffer = rewriter.create<util::LoadOp>(op.getLoc(), bufferArray, c1);
      });
      rewriter.replaceOp(op, getBit(rewriter, op.getLoc(), buffer, adaptor.getOffset()));

      return success();
   }
};

class ArrayLoadVariableSizeBinaryLowering : public OpConversionPattern<arrow::LoadVariableSizeBinaryOp> {
   public:
   using OpConversionPattern<arrow::LoadVariableSizeBinaryOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::LoadVariableSizeBinaryOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Value valueBuffer;
      mlir::Value binaryBuffer;
      mlir::Value c1;
      createAtArrayCreation(rewriter, adaptor.getArray(), [&](mlir::ConversionPatternRewriter& rewriter) {
         auto bufferArrayPtr = rewriter.create<util::TupleElementPtrOp>(op.getLoc(), util::RefType::get(util::RefType::get(util::RefType::get(rewriter.getI8Type()))), adaptor.getArray(), 5);
         auto bufferArray = rewriter.create<util::LoadOp>(op.getLoc(), bufferArrayPtr);
         //load buffer with main values (offset 1)
         c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
         auto c2 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 2);
         valueBuffer = rewriter.create<util::LoadOp>(op.getLoc(), bufferArray, c1);
         valueBuffer = rewriter.create<util::GenericMemrefCastOp>(op.getLoc(), util::RefType::get(rewriter.getI32Type()), valueBuffer);
         binaryBuffer = rewriter.create<util::LoadOp>(op.getLoc(), bufferArray, c2);
      });
      auto pos1 = rewriter.create<util::LoadOp>(op.getLoc(), valueBuffer, adaptor.getOffset());
      Value ip1 = rewriter.create<arith::AddIOp>(op.getLoc(), rewriter.getIndexType(), adaptor.getOffset(), c1);
      Value pos2 = rewriter.create<util::LoadOp>(op.getLoc(), rewriter.getI32Type(), valueBuffer, ip1);
      Value len = rewriter.create<arith::SubIOp>(op.getLoc(), rewriter.getI32Type(), pos2, pos1);
      auto pos1AsIndex = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), pos1);
      Value ptr = rewriter.create<util::ArrayElementPtrOp>(op.getLoc(), util::RefType::get(rewriter.getI8Type()), binaryBuffer, pos1AsIndex);
      rewriter.replaceOp(op, {len, ptr});
      return success();
   }
};

class ArrayIsValidLowering : public OpConversionPattern<arrow::IsValidOp> {
   public:
   using OpConversionPattern<arrow::IsValidOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::IsValidOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Value buffer;
      createAtArrayCreation(rewriter, adaptor.getArray(), [&](mlir::ConversionPatternRewriter& rewriter) {
         auto bufferArrayPtr = rewriter.create<util::TupleElementPtrOp>(op.getLoc(), util::RefType::get(util::RefType::get(util::RefType::get(rewriter.getI8Type()))), adaptor.getArray(), 5);
         auto bufferArray = rewriter.create<util::LoadOp>(op.getLoc(), bufferArrayPtr);
         //load  validity buffer (offset 0)
         auto c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
         buffer = rewriter.create<util::LoadOp>(op.getLoc(), bufferArray, c0);
      });
      rewriter.replaceOp(op, getBit(rewriter, op.getLoc(), buffer, adaptor.getOffset()));
      return success();
   }
};

class BuilderFromPtrLowering : public OpConversionPattern<arrow::BuilderFromPtr> {
   public:
   using OpConversionPattern<arrow::BuilderFromPtr>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::BuilderFromPtr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, adaptor.getPtr());
      return success();
   }
};
class BuilderAppendFixedSizedLowering : public OpConversionPattern<arrow::AppendFixedSizedOp> {
   public:
   using OpConversionPattern<arrow::AppendFixedSizedOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::AppendFixedSizedOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto intType = op.getValue().getType();
      auto loc = op.getLoc();
      auto builderVal = adaptor.getBuilder();
      auto isValid = adaptor.getValid();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      auto val = adaptor.getValue();
      // create stack slot (using alloca) for the value, so that we can then provide the pointer to the value to a generic function. (alloca needs to be in the beginning of the function, not nested in some loop)
      auto funcParent = op->getParentOfType<func::FuncOp>();
      mlir::Value stackSlot;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(&funcParent.getBody().front());
         stackSlot = rewriter.create<util::AllocaOp>(loc, util::RefType::get(val.getType()), mlir::Value{});
      }
      // store the value in the allocated stack slot
      rewriter.create<util::StoreOp>(loc, val, stackSlot, mlir::Value{});
      rt::ArrowColumnBuilder::addFixedSized(rewriter, loc)({builderVal, isValid, stackSlot});
      rewriter.eraseOp(op);

      return success();
   }
};
class BuilderAppendBoolLowering : public OpConversionPattern<arrow::AppendBoolOp> {
   public:
   using OpConversionPattern<arrow::AppendBoolOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::AppendBoolOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto intType = op.getValue().getType();
      auto loc = op.getLoc();
      auto builderVal = adaptor.getBuilder();
      auto isValid = adaptor.getValid();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      auto val = adaptor.getValue();

      rt::ArrowColumnBuilder::addBool(rewriter, loc)({builderVal, isValid, val});
      rewriter.eraseOp(op);

      return success();
   }
};
class BuilderAppendVariableSizeBinaryLowering : public OpConversionPattern<arrow::AppendVariableSizeBinaryOp> {
   public:
   using OpConversionPattern<arrow::AppendVariableSizeBinaryOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(arrow::AppendVariableSizeBinaryOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto intType = op.getValue().getType();
      auto loc = op.getLoc();
      auto builderVal = adaptor.getBuilder();
      auto isValid = adaptor.getValid();
      if (!isValid) {
         isValid = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      auto val = adaptor.getValue();

      rt::ArrowColumnBuilder::addBinary(rewriter, loc)({builderVal, isValid, val});
      rewriter.eraseOp(op);

      return success();
   }
};
} // end anonymous namespace
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
void ArrowToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<util::UtilDialect>()->getFunctionHelper().setParentModule(module);
   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalDialect<gpu::GPUDialect>();
   target.addLegalDialect<async::AsyncDialect>();
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   target.addIllegalDialect<arrow::ArrowDialect>();
   TypeConverter typeConverter;
   typeConverter.addConversion([&](mlir::Type type) { return type; });
   static auto hasDSAType = [](TypeConverter& converter, TypeRange types) -> bool {
      return llvm::any_of(types, [&converter](mlir::Type t) { auto converted = converter.convertType(t);return converted&&converted!=t; });
   };
   auto opIsWithoutDSATypes = [&](Operation* op) { return !hasDSAType(typeConverter, op->getOperandTypes()) && !hasDSAType(typeConverter, op->getResultTypes()); };
   target.addDynamicallyLegalDialect<scf::SCFDialect>(opIsWithoutDSATypes);
   target.addDynamicallyLegalDialect<arith::ArithDialect>(opIsWithoutDSATypes);

   target.addLegalDialect<cf::ControlFlowDialect>();

   target.addDynamicallyLegalDialect<util::UtilDialect>(opIsWithoutDSATypes);
   target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
      auto isLegal = !hasDSAType(typeConverter, op.getFunctionType().getInputs()) &&
         !hasDSAType(typeConverter, op.getFunctionType().getResults());
      return isLegal;
   });
   target.addDynamicallyLegalOp<func::ConstantOp>([&](func::ConstantOp op) {
      if (auto functionType = mlir::dyn_cast_or_null<mlir::FunctionType>(op.getType())) {
         auto isLegal = !hasDSAType(typeConverter, functionType.getInputs()) &&
            !hasDSAType(typeConverter, functionType.getResults());
         return isLegal;
      } else {
         return true;
      }
   });
   target.addDynamicallyLegalOp<func::CallOp, func::CallIndirectOp, func::ReturnOp>(opIsWithoutDSATypes);

   target.addDynamicallyLegalOp<util::SizeOfOp>(
      [&typeConverter](util::SizeOfOp op) {
         auto isLegal = !hasDSAType(typeConverter, op.getType());
         return isLegal;
      });
   auto* ctxt = &getContext();
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });

   typeConverter.addConversion([&](arrow::ArrayType arrayType) {
      auto idxType = IndexType::get(ctxt);
      auto bufferType = util::RefType::get(ctxt, mlir::IntegerType::get(ctxt, 8));
      auto bufferArrayType = util::RefType::get(ctxt, bufferType);
      //todo: childArrayType
      return util::RefType::get(&getContext(), mlir::TupleType::get(ctxt, {idxType, idxType, idxType, idxType, idxType, bufferArrayType}));
   });
   typeConverter.addConversion([&](arrow::ArrayBuilderType builderType) {
      return util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8));
   });
   RewritePatternSet patterns(&getContext());

   mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   mlir::scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
   patterns.insert<SimpleTypeConversionPattern<mlir::func::ConstantOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, &getContext());
   patterns.insert<ArrayIsValidLowering>(typeConverter, &getContext());
   patterns.insert<ArrayLoadFixedSizedLowering>(typeConverter, &getContext());
   patterns.insert<ArrayLoadVariableSizeBinaryLowering>(typeConverter, &getContext());
   patterns.insert<ArrayLoadBoolLowering>(typeConverter, &getContext());
   patterns.insert<BuilderFromPtrLowering>(typeConverter, &getContext());
   patterns.insert<BuilderAppendFixedSizedLowering>(typeConverter, &getContext());
   patterns.insert<BuilderAppendBoolLowering>(typeConverter, &getContext());
   patterns.insert<BuilderAppendVariableSizeBinaryLowering>(typeConverter, &getContext());
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::arrow::createLowerToStdPass() {
   return std::make_unique<ArrowToStdLoweringPass>();
}
