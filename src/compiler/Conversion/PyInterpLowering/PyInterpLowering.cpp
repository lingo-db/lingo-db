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

#include <lingodb/compiler/Dialect/DB/IR/DBOpsTypes.h.inc>
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
      auto nameVal = rewriter.create<util::CreateConstVarLen>(op.getLoc(), util::VarLen32Type::get(rewriter.getContext()),op.getName());
      auto val = rt::PythonRuntime::import(rewriter, op.getLoc())({})[0];
      rewriter.replaceOp(op, val);

      return success();
   }
};
class CreateModuleLowering : public OpConversionPattern<py_interp::CreateModule> {
   public:
   using OpConversionPattern<py_interp::CreateModule>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::CreateModule op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      mlir::Value moduleNameVal = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()),op.getName());
      mlir::Value codeVal = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()),op.getCode());
      auto val = rt::PythonRuntime::createModule(rewriter, op.getLoc())({moduleNameVal, codeVal})[0];
      rewriter.replaceOp(op, val);
      return success();
   }
};
class GetAttrLowering : public OpConversionPattern<py_interp::GetAttr> {
   public:
   using OpConversionPattern<py_interp::GetAttr>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::GetAttr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      mlir::Value attrNameVal = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()),op.getName());
      auto val = rt::PythonRuntime::getAttr(rewriter, op.getLoc())({adaptor.getOn(), attrNameVal})[0];
      rewriter.replaceOp(op, val);
      return success();
   }
};
class GetAttr2Lowering : public OpConversionPattern<py_interp::GetAttr2> {
   public:
   using OpConversionPattern<py_interp::GetAttr2>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::GetAttr2 op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      auto val = rt::PythonRuntime::getAttr2(rewriter, op.getLoc())({adaptor.getOn(), adaptor.getName()})[0];
      rewriter.replaceOp(op, val);
      return success();
   }
};
class SetAttrLowering : public OpConversionPattern<py_interp::SetAttr> {
   public:
   using OpConversionPattern<py_interp::SetAttr>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::SetAttr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
      mlir::Value attrNameVal = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()),op.getName());
      rt::PythonRuntime::setAttr(rewriter, op.getLoc())({adaptor.getOn(), attrNameVal, adaptor.getValue()});
      rewriter.eraseOp(op);
      return success();
   }
};

class CallLowering : public OpConversionPattern<py_interp::Call> {
   public:
        using OpConversionPattern<py_interp::Call>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::Call op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      if (op.getKwnames().empty()) {
         std::optional<util::FunctionSpec> fn;
         switch (adaptor.getArgs().size()) {
            case 0:
               fn = rt::PythonRuntime::call0;
               break;
            case 1:
               fn =rt::PythonRuntime::call1;
               break;
            case 2:
               fn =rt::PythonRuntime::call2;
               break;
            case 3:
               fn =rt::PythonRuntime::call3;
               break;
            case 4:
               fn =rt::PythonRuntime::call4;
               break;
            case 5:
               fn =rt::PythonRuntime::call5;
               break;
            case 6:
               fn =rt::PythonRuntime::call6;
               break;
            case 7:
               fn =rt::PythonRuntime::call7;
               break;
            case 8:
               fn =rt::PythonRuntime::call8;
               break;
            case 9:
               fn =rt::PythonRuntime::call9;
               break;
            case 10:
               fn =rt::PythonRuntime::call10;
               break;
         }
         if (!fn) return failure();
         std::vector<mlir::Value> args;
         args.push_back(adaptor.getOn());
         args.insert(args.end(), adaptor.getArgs().begin(), adaptor.getArgs().end());
         auto res = fn.value()(rewriter, op->getLoc())(args)[0];
         rewriter.replaceOp(op, res);
         return success();
      }
      return failure();


   }
};
class DecRefLowering : public OpConversionPattern<py_interp::DecRef> {
   public:
   using OpConversionPattern<py_interp::DecRef>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::DecRef op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rt::PythonRuntime::decref(rewriter, op.getLoc())({adaptor.getObj()});
      rewriter.eraseOp(op);
      return success();
   }
};
class IncRefLowering : public OpConversionPattern<py_interp::IncRef> {
   public:
   using OpConversionPattern<py_interp::IncRef>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::IncRef op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      rt::PythonRuntime::incref(rewriter, op.getLoc())({adaptor.getObj()});
      rewriter.eraseOp(op);
      return success();
   }
};
class ConstStrLowering : public OpConversionPattern<py_interp::ConstStrPyObject> {
   public:
   using OpConversionPattern<py_interp::ConstStrPyObject>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::ConstStrPyObject op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto strConst = rewriter.create<util::CreateConstVarLen>(op.getLoc(), util::VarLen32Type::get(rewriter.getContext()),mlir::cast<mlir::StringAttr>(op.getValue()));
      rewriter.replaceOp(op, rt::PythonRuntime::fromVarLen32(rewriter, op.getLoc())({strConst})[0]);
      return success();
   }
};
class CastToPyObjectLowering : public OpConversionPattern<py_interp::CastToPyObject> {
   public:
   using OpConversionPattern<py_interp::CastToPyObject>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::CastToPyObject op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto t = op.getFrom().getType();
      if (op.getPythonType()=="builtins.float") {
         if (auto floatType = mlir::dyn_cast<mlir::FloatType>(t)){
            mlir::Value val = adaptor.getFrom();
            if (floatType.isF32()) {
               val = rewriter.create<arith::ExtFOp>(op.getLoc(), rewriter.getF64Type(), val);
            }
            rewriter.replaceOp(op,  rt::PythonRuntime::fromDouble(rewriter, op.getLoc())({val})[0]);
         }else {
            return failure();
         }
      }else if (op.getPythonType()=="builtins.bool") {
         assert(mlir::isa<mlir::IntegerType>(t)&& mlir::cast<mlir::IntegerType>(t).getWidth()==1&&"bool cast source must be integer");
         rewriter.replaceOp(op, rt::PythonRuntime::fromBool(rewriter, op.getLoc())({adaptor.getFrom()})[0]);
      }else if (op.getPythonType()=="builtins.int") {
         if (auto intType = mlir::dyn_cast<mlir::IntegerType>(t)) {
            mlir::Value val = adaptor.getFrom();
            if (intType.getWidth() < 64) {
               val = rewriter.create<arith::ExtSIOp>(op.getLoc(), rewriter.getI64Type(), val);
            } else if (intType.getWidth() > 64) {
               return failure();
            }
            rewriter.replaceOp(op, rt::PythonRuntime::fromInt64(rewriter, op.getLoc())({val})[0]);
         }else {
            return failure();
         }
      } else if (mlir::isa<util::VarLen32Type>(t)&&op.getPythonType()=="builtins.str") {
         rewriter.replaceOp(op, rt::PythonRuntime::fromVarLen32(rewriter, op.getLoc())({adaptor.getFrom()})[0]);
      } else if (mlir::isa<mlir::IntegerType>(t) && op.getPythonType()=="datetime.date"){
         rewriter.replaceOp(op, rt::PythonRuntime::fromDate(rewriter, op.getLoc())({adaptor.getFrom()})[0]);
      } else {
         return failure();
      }
      return success();
   }
};
class CastFromPyObjectLowering : public OpConversionPattern<py_interp::CastFromPyObject> {
   public:
   using OpConversionPattern<py_interp::CastFromPyObject>::OpConversionPattern;
   LogicalResult matchAndRewrite(py_interp::CastFromPyObject op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto t = op.getTo().getType();
      if (auto floatType = mlir::dyn_cast<mlir::FloatType>(t)) {
         mlir::Value val = rt::PythonRuntime::toDouble(rewriter, op.getLoc())({adaptor.getFrom()})[0];
         if (floatType.isF32()) {
            val = rewriter.create<arith::TruncFOp>(op.getLoc(), rewriter.getF32Type(), val);
         }
         rewriter.replaceOp(op, val);
      }else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(t)) {
         if (intType.getWidth()==1) {
            rewriter.replaceOp(op, rt::PythonRuntime::toBool(rewriter, op.getLoc())({adaptor.getFrom()})[0]);
         }else {
            mlir::Value val = rt::PythonRuntime::toInt64(rewriter, op.getLoc())({adaptor.getFrom()})[0];
            if (intType.getWidth() < 64) {
               val = rewriter.create<arith::TruncIOp>(op.getLoc(), intType, val);
            } else if (intType.getWidth() > 64) {
               return failure();
            }
            rewriter.replaceOp(op, val);
         }
      } else if (mlir::isa<util::VarLen32Type>(t)) {
         rewriter.replaceOp(op, rt::PythonRuntime::toVarLen32(rewriter, op.getLoc())({adaptor.getFrom()})[0]);
      } else {
         return failure();
      }
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
#ifdef USE_CPYTHON_WASM_RUNTIME
      return mlir::IntegerType::get(ctxt, 32);
#else
      return util::RefType::get(ctxt, mlir::IntegerType::get(ctxt, 8));
#endif
   });
   typeConverter.addConversion([&](FunctionType funcType) {
      llvm::SmallVector<Type> convertedInputs;
      for (auto inputType : funcType.getInputs()) {
         convertedInputs.push_back(typeConverter.convertType(inputType));
      }
      llvm::SmallVector<Type> convertedResults;
      for (auto resultType : funcType.getResults()) {
         convertedResults.push_back(typeConverter.convertType(resultType));
      }
      return FunctionType::get(ctxt, convertedInputs, convertedResults);
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
   patterns.insert<CreateModuleLowering>(typeConverter, &getContext());
   patterns.insert<GetAttrLowering>(typeConverter, &getContext());
   patterns.insert<GetAttr2Lowering>(typeConverter, &getContext());
   patterns.insert<SetAttrLowering>(typeConverter, &getContext());
   patterns.insert<CallLowering>(typeConverter, &getContext());
   patterns.insert<CastToPyObjectLowering>(typeConverter, &getContext());
   patterns.insert<CastFromPyObjectLowering>(typeConverter, &getContext());
   patterns.insert<DecRefLowering>(typeConverter, &getContext());
   patterns.insert<IncRefLowering>(typeConverter, &getContext());
   patterns.insert<ConstStrLowering>(typeConverter, &getContext());
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::py_interp::createLowerToStdPass() {
   return std::make_unique<PyInterpLoweringPass>();
}