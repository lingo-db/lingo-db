#include "lingodb/compiler/Conversion/DSAToStd/DSAToStd.h"
#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSAOps.h"
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
using namespace mlir;

namespace {
using namespace lingodb::compiler::dialect;
class SetResultOpLowering : public OpConversionPattern<dsa::SetResultOp> {
   public:
   using OpConversionPattern<dsa::SetResultOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::SetResultOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Type> types;
      auto parentModule = op->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp = parentModule.lookupSymbol<mlir::func::FuncOp>("rt_get_execution_context");
      if (!funcOp) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointToStart(parentModule.getBody());
         funcOp = rewriter.create<mlir::func::FuncOp>(op->getLoc(), "rt_get_execution_context", rewriter.getFunctionType({}, {util::RefType::get(getContext(), rewriter.getI8Type())}), rewriter.getStringAttr("private"), mlir::ArrayAttr{}, mlir::ArrayAttr{});
      }

      mlir::Value executionContext = rewriter.create<mlir::func::CallOp>(op->getLoc(), funcOp, mlir::ValueRange{}).getResult(0);
      auto resultId = rewriter.create<mlir::arith::ConstantIntOp>(op->getLoc(), op.getResultId(), rewriter.getI32Type());
      lingodb::compiler::runtime::ExecutionContext::setResult(rewriter, op->getLoc())({executionContext, resultId, adaptor.getState()});
      rewriter.eraseOp(op);
      return success();
   }
};
class DownCastLowering : public OpConversionPattern<dsa::DownCast> {
   public:
   using OpConversionPattern<dsa::DownCast>::OpConversionPattern;
   LogicalResult matchAndRewrite(dsa::DownCast op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
   }
};
} // end anonymous namespace

namespace {
struct DSAToStdLoweringPass
   : public PassWrapper<DSAToStdLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DSAToStdLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "lower-dsa"; }

   DSAToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, dsa::DSADialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect>();
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
void DSAToStdLoweringPass::runOnOperation() {
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
   typeConverter.addConversion([&](dsa::ColumnBuilderType tableType) {
      return util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](dsa::ColumnType tableType) {
      return util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](dsa::TableType tableType) {
      return util::RefType::get(&getContext(), IntegerType::get(&getContext(), 8));
   });
   typeConverter.addConversion([&](::dsa::ArrowFixedSizedBinaryType t) {
      if (t.getByteWidth() > 8) return mlir::Type();
      size_t bits = 0;
      if (t.getByteWidth() == 1) {
         bits = 8;
      } else if (t.getByteWidth() == 2) {
         bits = 16;
      } else if (t.getByteWidth() <= 4) {
         bits = 32;
      } else {
         bits = 64;
      }
      return (Type) mlir::IntegerType::get(ctxt, bits);
   });
   typeConverter.addConversion([&](dsa::ArrowStringType type) {
      return util::VarLen32Type::get(&getContext());
   });
   typeConverter.addConversion([&](dsa::ArrowListType type) {
      return util::BufferType::get(&getContext(), typeConverter.convertType(type.getType()));
   });
   typeConverter.addConversion([&](dsa::ArrowDecimalType type) {
      return IntegerType::get(ctxt, 128);
   });
   typeConverter.addConversion([&](dsa::ArrowDate32Type t) {
      return IntegerType::get(ctxt, 32);
   });
   typeConverter.addConversion([&](dsa::ArrowDate64Type t) {
      return IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](dsa::ArrowTimeStampType t) {
      return IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](dsa::ArrowMonthIntervalType t) {
      return mlir::IntegerType::get(ctxt, 32);
   });
   typeConverter.addConversion([&](dsa::ArrowDayTimeIntervalType t) {
      return mlir::IntegerType::get(ctxt, 64);
   });
   RewritePatternSet patterns(&getContext());

   mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   dsa::populateDSAToStdPatterns(typeConverter, patterns);
   util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   mlir::scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
   patterns.insert<SimpleTypeConversionPattern<mlir::func::ConstantOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, &getContext());
   patterns.insert<SetResultOpLowering>(typeConverter, &getContext());
   patterns.insert<DownCastLowering>(typeConverter, &getContext());

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::dsa::createLowerToStdPass() {
   return std::make_unique<DSAToStdLoweringPass>();
}
