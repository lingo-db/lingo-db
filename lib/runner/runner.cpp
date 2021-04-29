#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
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
#include <llvm/Support/ErrorOr.h>
#include <runner/runner.h>

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

int dumpLLVMIR(mlir::ModuleOp module) {
   // Convert the module to LLVM IR in a new LLVM IR context.
   llvm::LLVMContext llvmContext;
   auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
   if (!llvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
   }

   // Initialize LLVM targets.
   llvm::InitializeNativeTarget();
   llvm::InitializeNativeTargetAsmPrinter();
   mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

   /// Optionally run an optimization pipeline over the llvm module.
   auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/false ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
   if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
      return -1;
   }
   llvm::errs() << *llvmModule << "\n";
   return 0;
}
namespace {
struct ToLLVMLoweringPass
   : public mlir::PassWrapper<ToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
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
   options.useBarePtrCallConv = true;
   mlir::LLVMTypeConverter typeConverter(&getContext(), options);

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
   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
   return std::make_unique<ToLLVMLoweringPass>();
}

struct RunnerContext {
   mlir::MLIRContext context;
   mlir::OwningModuleRef module;
};
Runner::Runner() : context(nullptr) {}
bool Runner::load(std::string file) {
   RunnerContext* ctxt = new RunnerContext;
   this->context = (void*) ctxt;

   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::memref::MemRefDialect>();

   mlir::MLIRContext& context = ctxt->context;
   context.appendDialectRegistry(registry);
   mlir::registerLLVMDialectTranslation(context);

   llvm::SourceMgr sourceMgr;
   llvm::DebugFlag = false;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(file, context, ctxt->module))
      return false;
   return true;
}
bool Runner::lower() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::PassManager pm(&ctxt->context);
   pm.addPass(mlir::db::createLowerToStdPass());
   if (mlir::failed(pm.run(ctxt->module.get()))) {
      return false;
   }
   return true;
}
bool Runner::lowerToLLVM() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   mlir::PassManager pm2(&ctxt->context);
   pm2.addPass(mlir::createLowerToCFGPass());
   pm2.addPass(createLowerToLLVMPass());
   if (mlir::failed(pm2.run(ctxt->module.get()))) {
      return false;
   }
   return true;
}
void Runner::dump() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   ctxt->module->dump();
}
void Runner::dumpLLVM() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   dumpLLVMIR(ctxt->module.get());
}
bool Runner::runJit() {
   RunnerContext* ctxt = (RunnerContext*) this->context;
   // Initialize LLVM targets.
   llvm::InitializeNativeTarget();
   llvm::InitializeNativeTargetAsmPrinter();

   // An optimization pipeline to use within the execution engine.
   auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/false ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

   // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
   // the module.
   auto maybeEngine = mlir::ExecutionEngine::create(
      ctxt->module.get(), /*llvmModuleBuilder=*/nullptr, optPipeline);
   assert(maybeEngine && "failed to construct an execution engine");
   auto& engine = maybeEngine.get();

   //int32_t res=0;
   std::vector<void*> args = {}; // {&res};
   // Invoke the JIT-compiled function.
   auto invocationResult = engine->invokePacked("main", args);
   if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      return false;
   }

   return true;
}
Runner::~Runner() {
   if (this->context) {
      delete (RunnerContext*) this->context;
   }
}