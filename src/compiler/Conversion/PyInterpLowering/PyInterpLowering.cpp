#include "lingodb/compiler/Conversion/PyInterpLowering/PyInterpLoweringPass.h"

#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"
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

#include <lingodb/compiler/runtime/PythonRuntime.h>
using namespace mlir;
namespace py_interp = lingodb::compiler::dialect::py_interp;
namespace util = lingodb::compiler::dialect::util;
namespace rt = lingodb::compiler::runtime;

namespace {
struct PyInterpLoweringPass
   : public PassWrapper<PyInterpLoweringPass, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PyInterpLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "lower-arrow"; }

   PyInterpLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, py_interp::PyInterpDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect>();
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

class ImportLowering : public OpConversionPattern<py_interp::ImportOp> {
   public:
   using OpConversionPattern<py_interp::ImportOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::ImportOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto val = rt::PythonRuntime::test(rewriter, op.getLoc())({})[0];
      rewriter.replaceOp(op, val);

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
void PyInterpLoweringPass::runOnOperation() {
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
   target.addIllegalDialect<py_interp::PyInterpDialect>();
   TypeConverter typeConverter;
   typeConverter.addConversion([&](mlir::Type type) { return type; });
   static auto hasArrowType = [](TypeConverter& converter, TypeRange types) -> bool {
      return llvm::any_of(types, [&converter](mlir::Type t) { auto converted = converter.convertType(t);return converted&&converted!=t; });
   };
   auto opIsWithoutArrowTypes = [&](Operation* op) { return !hasArrowType(typeConverter, op->getOperandTypes()) && !hasArrowType(typeConverter, op->getResultTypes()); };
   target.addDynamicallyLegalDialect<scf::SCFDialect>(opIsWithoutArrowTypes);
   target.addDynamicallyLegalDialect<arith::ArithDialect>(opIsWithoutArrowTypes);

   target.addLegalDialect<cf::ControlFlowDialect>();

   target.addDynamicallyLegalDialect<util::UtilDialect>(opIsWithoutArrowTypes);
   target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
      auto isLegal = !hasArrowType(typeConverter, op.getFunctionType().getInputs()) &&
         !hasArrowType(typeConverter, op.getFunctionType().getResults());
      return isLegal;
   });
   target.addDynamicallyLegalOp<func::ConstantOp>([&](func::ConstantOp op) {
      if (auto functionType = mlir::dyn_cast_or_null<mlir::FunctionType>(op.getType())) {
         auto isLegal = !hasArrowType(typeConverter, functionType.getInputs()) &&
            !hasArrowType(typeConverter, functionType.getResults());
         return isLegal;
      } else {
         return true;
      }
   });
   target.addDynamicallyLegalOp<func::CallOp, func::CallIndirectOp, func::ReturnOp>(opIsWithoutArrowTypes);

   target.addDynamicallyLegalOp<util::SizeOfOp>(
      [&typeConverter](util::SizeOfOp op) {
         auto isLegal = !hasArrowType(typeConverter, op.getType());
         return isLegal;
      });
   auto* ctxt = &getContext();
   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });

   typeConverter.addConversion([&](py_interp::PyObjectType) {
      //todo: childArrayType
      return util::RefType::get(ctxt, mlir::IntegerType::get(ctxt, 8));
   });
   RewritePatternSet patterns(&getContext());

   mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   mlir::scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
   patterns.insert<SimpleTypeConversionPattern<mlir::func::ConstantOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, &getContext());
   patterns.insert<ImportLowering>(typeConverter, &getContext());
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::py_interp::createLowerToStdPass() {
   return std::make_unique<PyInterpLoweringPass>();
}