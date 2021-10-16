#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "runner/jit.h"

#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <runner/runner.h>
#include <runtime/helpers.h>

namespace {
struct ToLLVMLoweringPass
   : public mlir::PassWrapper<ToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::arith::ArithmeticDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void ToLLVMLoweringPass::runOnOperation() {
   // The first thing to define is the conversion target. This will define the
   // final target for this lowering. For this lowering, we are only targeting
   // the LLVM dialect.
   mlir::LLVMConversionTarget target(getContext());
   target.addLegalOp<mlir::ModuleOp>();

   // During this lowering, we will also be lowering the MemRef types, that are
   // currently being operated on, to a representation in LLVM. To perform this
   // conversion we use a TypeConverter as part of the lowering. This converter
   // details how one type maps to another. This is necessary now that we will be
   // doing more complicated lowerings, involving loop region arguments.
   mlir::LowerToLLVMOptions options(&getContext());
   //options.emitCWrappers = true;
   mlir::LLVMTypeConverter typeConverter(&getContext(), options);
   typeConverter.addSourceMaterialization([&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
      return valueRange.front();
   });

   // Now that the conversion target has been defined, we need to provide the
   // patterns used for lowering. At this point of the compilation process, we
   // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
   // are already exists a set of patterns to transform `affine` and `std`
   // dialects. These patterns lowering in multiple stages, relying on transitive
   // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
   // patterns must be applied to fully transform an illegal operation into a
   // set of legal ones.
   mlir::RewritePatternSet patterns(&getContext());
   populateAffineToStdConversionPatterns(patterns);
   populateLoopToStdConversionPatterns(patterns);
   mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
   populateStdToLLVMConversionPatterns(typeConverter, patterns);
   mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
   mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

namespace runner {

extern const unsigned char kPrecompiledBitcode[];
extern const size_t kPrecompiledBitcodeSize;
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
   return std::make_unique<ToLLVMLoweringPass>();
}
int loadMLIR(std::string inputFilename, mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return -1;
   }

   // Parse the input mlir.
   llvm::SourceMgr sourceMgr;
   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
   module = mlir::parseSourceFile(sourceMgr, &context);
   if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
   }
   return 0;
}
int loadMLIRFromString(const std::string& input, mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
   module = mlir::parseSourceString(input, &context);
   if (!module) {
      llvm::errs() << "Error can't load module\n";
      return 3;
   }
   return 0;
}
static std::unique_ptr<llvm::Module>
convertMLIRModule(mlir::ModuleOp module, llvm::LLVMContext& context) {
   //////////////////////////////////////////////////////////////////////////////////////
   auto bitcode = llvm::StringRef(reinterpret_cast<const char*>(kPrecompiledBitcode),
                                  kPrecompiledBitcodeSize);

   /// Read from file into memory buffer.
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrError =
      llvm::MemoryBuffer::getMemBuffer(bitcode, "precompiled", false);

   std::unique_ptr<llvm::MemoryBuffer> buffer = move(bufferOrError.get());

   /// Parse the IR module.
   llvm::Expected<std::unique_ptr<llvm::Module>> moduleOrError =
      llvm::getOwningLazyBitcodeModule(move(buffer), context);
   if (!moduleOrError) {
      // NOTE: llvm::handleAllErrors() fails linking with RTTI-disabled LLVM builds
      // (ARROW-5148)
      std::string str;
      llvm::raw_string_ostream stream(str);
      stream << moduleOrError.takeError();
      llvm::dbgs() << stream.str() << "\n";
   }
   std::unique_ptr<llvm::Module> irModule = move(moduleOrError.get());

   //////////////////////////////////////////////////////////////////////////////////////

   std::unique_ptr<llvm::Module> mainModule =
      translateModuleToLLVMIR(module, context);
   llvm::Linker::linkModules(*mainModule, std::move(irModule), llvm::Linker::LinkOnlyNeeded);
   return mainModule;
}

struct RunnerContext {
   mlir::MLIRContext context;
   mlir::OwningModuleRef module;
   size_t numArgs;
   size_t numResults;
};
Runner::Runner() : context(nullptr) {
   llvm::DebugFlag = true;
}
bool Runner::load(std::string file) {
   RunnerContext* ctxt = new RunnerContext;
   this->context = (void*) ctxt;

   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();

   registry.insert<mlir::scf::SCFDialect>();

   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::memref::MemRefDialect>();

   mlir::MLIRContext& context = ctxt->context;
   context.appendDialectRegistry(registry);
   mlir::registerLLVMDialectTranslation(context);

   llvm::SourceMgr sourceMgr;
   llvm::DebugFlag = false;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (loadMLIR(file, context, ctxt->module))
      return false;
   return true;
}
bool Runner::loadString(std::string input) {
   RunnerContext* ctxt = new RunnerContext;
   this->context = (void*) ctxt;

   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   registry.insert<mlir::scf::SCFDialect>();

   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::memref::MemRefDialect>();

   mlir::MLIRContext& context = ctxt->context;
   context.appendDialectRegistry(registry);
   mlir::registerLLVMDialectTranslation(context);

   llvm::DebugFlag = false;
   if (loadMLIRFromString(input, context, ctxt->module))
      return false;
   return true;
}
bool Runner::optimize() {
   auto start = std::chrono::high_resolution_clock::now();
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::PassManager pm(&ctxt->context);
   pm.addPass(mlir::createInlinerPass());
   pm.addPass(mlir::createSymbolDCEPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createSimplifyAggregationsPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createExtractNestedOperatorsPass());
   pm.addPass(mlir::createCSEPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createDecomposeLambdasPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createImplicitToExplicitJoinsPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createPushdownPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createUnnestingPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createOptimizeJoinOrderPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createCombinePredicatesPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createOptimizeImplementationsPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createIntroduceTmpPass());
   pm.addNestedPass<mlir::FuncOp>(mlir::relalg::createLowerToDBPass());
   pm.addPass(mlir::createCanonicalizerPass());

   if (mlir::failed(pm.run(ctxt->module.get()))) {
      return false;
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "optimization took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
   return true;
}
bool Runner::lower() {
   auto start = std::chrono::high_resolution_clock::now();
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::PassManager pm(&ctxt->context);
   pm.addPass(mlir::db::createLowerToStdPass());
   pm.addPass(mlir::createCanonicalizerPass());
   if (mlir::failed(pm.run(ctxt->module.get()))) {
      return false;
   }

   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "lowering to std took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
   return true;
}
bool Runner::lowerToLLVM() {
   auto start = std::chrono::high_resolution_clock::now();
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::ModuleOp moduleOp = ctxt->module.get();
   if (auto mainFunc = moduleOp.lookupSymbol<mlir::FuncOp>("main")) {
      ctxt->numArgs = mainFunc.getNumArguments();
      ctxt->numResults = mainFunc.getNumResults();
      mlir::db::codegen::FunctionRegistry registry(moduleOp->getContext());
      registry.registerFunctions();
      mlir::OpBuilder builder(&mainFunc.body().front().front());
      registry.getFunction(builder, mlir::db::codegen::FunctionRegistry::FunctionId::SetExecutionContext);
   }
   mlir::PassManager pm2(&ctxt->context);
   pm2.addPass(mlir::createLowerToCFGPass());
   pm2.addPass(createLowerToLLVMPass());
   if (mlir::failed(pm2.run(ctxt->module.get()))) {
      return false;
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "lowering to llvm took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
   return true;
}
void Runner::dump() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   ctxt->module->dump();
}

bool Runner::runJit(runtime::ExecutionContext* context, size_t repeats, std::function<void(uint8_t*)> callback) {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   // Initialize LLVM targets.
   llvm::InitializeNativeTarget();
   llvm::InitializeNativeTargetAsmPrinter();
   auto targetTriple = llvm::sys::getDefaultTargetTriple();
   std::string errorMessage;
   const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
   if (!target) {
      assert(false && "could not get target");
      return false;
   }

   // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
   // the module.
   llvm::orc::ThreadSafeContext llvmContext{std::make_unique<llvm::LLVMContext>()};
   auto startConv = std::chrono::high_resolution_clock::now();
   std::unique_ptr<llvm::Module> converted = convertMLIRModule(ctxt->module.get(), *llvmContext.getContext());
   auto endConv = std::chrono::high_resolution_clock::now();
   std::cout << "conversion: " << std::chrono::duration_cast<std::chrono::milliseconds>(endConv - startConv).count() << " ms" << std::endl;

   auto start = std::chrono::high_resolution_clock::now();
   runner::JIT jit(llvmContext);
   if (jit.addModule(std::move(converted))) {
      assert(false);
   }
   uint8_t* res;

   // Invoke the JIT-compiled function.
   auto* lookupResult = jit.getPointerToFunction("main");

   if (!lookupResult) {
      llvm::errs() << "JIT invocation failed\n";
      return false;
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "jit: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

   {
      auto* lookupResult2 = jit.getPointerToFunction("_mlir_ciface_set_execution_context");
      if (!lookupResult2) {
         llvm::errs() << "JIT invocation failed\n";
         return false;
      }
      typedef uint8_t* (*myfunc)(void*);
      auto fn = (myfunc) lookupResult2;
      fn(context);
   }
   auto* funcPtr = lookupResult;
   std::vector<size_t> measuredTimes;
   for (size_t i = 0; i < repeats; i++) {
      auto executionStart = std::chrono::high_resolution_clock::now();
      if (ctxt->numResults == 1) {
         typedef uint8_t* (*myfunc)();
         auto fn = (myfunc) funcPtr;
         res = fn();
      } else {
         typedef void (*myfunc)();
         auto fn = (myfunc) funcPtr;
         fn();
      }
      auto executionEnd = std::chrono::high_resolution_clock::now();
      measuredTimes.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(executionEnd - executionStart).count());
   }
   std::cout << "runtime: " << (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]) << " ms" << std::endl;

   if (ctxt->numResults == 1) {
      callback(res);
   }

   return true;
}
Runner::~Runner() {
   if (this->context) {
      delete (RunnerContext*) this->context;
   }
}
} // namespace runner
