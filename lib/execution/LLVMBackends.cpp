#include "execution/LLVMBackends.h"
#include "execution/BackendPasses.h"
#include "execution/Error.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/util/FunctionHelper.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <spawn.h>

#include "dlfcn.h"
#include "unistd.h"
#include "utility/Tracer.h"
namespace {
static utility::Tracer::Event execution("LLVM", "execution");

static bool lowerToLLVMDialect(mlir::ModuleOp& moduleOp, bool verify) {
   std::string error;
   auto targetTriple = llvm::sys::getDefaultTargetTriple();

   // Look up the target using the target triple.
   auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
   if (!target) {
      llvm::errs() << error;
      return 1;
   }

   // Create the TargetMachine.
   const auto* cpu = "generic";
   const auto* features = "";
   llvm::TargetOptions opt;
   auto rm = std::optional<llvm::Reloc::Model>();
   auto* targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, rm);

   if (!targetMachine) {
      llvm::errs() << "Could not create TargetMachine!";
      return 1;
   }

   // Retrieve the data layout from the TargetMachine.
   const llvm::DataLayout& dataLayout = targetMachine->createDataLayout();
   mlir::Attribute dataLayoutSpec = mlir::translateDataLayout(dataLayout, moduleOp->getContext());
   moduleOp->setAttr("util.dataLayout", dataLayoutSpec);

   mlir::PassManager pm2(moduleOp->getContext());
   pm2.enableVerifier(verify);
   pm2.addPass(mlir::createConvertSCFToCFPass());
   pm2.addPass(mlir::util::createUtilToLLVMPass());
   pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
   pm2.addPass(mlir::createArithToLLVMConversionPass());
   pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
   pm2.addPass(mlir::createArithToLLVMConversionPass());
   pm2.addPass(mlir::createConvertFuncToLLVMPass());
   pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
   pm2.addNestedPass<mlir::LLVM::LLVMFuncOp>(execution::createEnforceCABI());
   pm2.addPass(mlir::createCSEPass());
   if (mlir::failed(pm2.run(moduleOp))) {
      return false;
   }
   return true;
}
static void addLLVMExecutionContextFuncs(mlir::ModuleOp& moduleOp) {
   mlir::OpBuilder builder(moduleOp->getContext());
   builder.setInsertionPointToStart(moduleOp.getBody());
   auto pointerType = mlir::LLVM::LLVMPointerType::get(moduleOp->getContext());
   auto globalOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), builder.getI64Type(), false, mlir::LLVM::Linkage::Private, "execution_context", builder.getI64IntegerAttr(0));
   auto setExecContextFn = builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), "rt_set_execution_context", mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(builder.getContext()), builder.getI64Type()), mlir::LLVM::Linkage::External);
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto* block = setExecContextFn.addEntryBlock(builder);
      auto execContext = block->getArgument(0);
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      builder.create<mlir::LLVM::StoreOp>(builder.getUnknownLoc(), execContext, ptr);
      builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
   }
   if (auto getExecContextFn = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(moduleOp.lookupSymbol("rt_get_execution_context"))) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto* block = getExecContextFn.addEntryBlock(builder);
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      auto execContext = builder.create<mlir::LLVM::LoadOp>(builder.getUnknownLoc(), builder.getI64Type(), ptr);
      auto execContextAsPtr = builder.create<mlir::LLVM::IntToPtrOp>(builder.getUnknownLoc(), pointerType, execContext);
      builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{execContextAsPtr});
   }
}

static llvm::Error performDefaultLLVMPasses(llvm::Module* module) {
   llvm::legacy::FunctionPassManager funcPM(module);
   funcPM.add(llvm::createInstructionCombiningPass());
   funcPM.add(llvm::createReassociatePass());
   funcPM.add(llvm::createGVNPass());
   funcPM.add(llvm::createCFGSimplificationPass());

   funcPM.doInitialization();
   for (auto& func : *module) {
      if (!func.hasOptNone()) {
         funcPM.run(func);
      }
   }
   funcPM.doFinalization();
   return llvm::Error::success();
}

static void linkStatic(mlir::ExecutionEngine* engine, execution::Error& error, execution::mainFnType& mainFunc, execution::setExecutionContextFnType& setExecutionContextFn) {
   auto currPath = std::filesystem::current_path();
   std::ofstream symbolfile("symbolfile");
   mlir::util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
      symbolfile << s << " = " << ptr << ";\n";
   });
   execution::visitBareFunctions([&](std::string s, void* ptr) {
      symbolfile << s << " = " << ptr << ";\n";
   });
   symbolfile.close();

   engine->dumpToObjectFile("llvm-jit-static.o");
   std::string cmd = "g++ -shared -fPIC -o llvm-jit-static.so -Wl,--just-symbols=symbolfile llvm-jit-static.o";
   auto* pPipe = ::popen(cmd.c_str(), "r");
   if (pPipe == nullptr) {
      error.emit() << "Could not compile query module statically (Pipe could not be opened)";
      return;
   }
   std::array<char, 256> buffer;
   std::string result;
   while (not std::feof(pPipe)) {
      auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
      result.append(buffer.data(), bytes);
   }
   auto rc = ::pclose(pPipe);
   if (WEXITSTATUS(rc)) {
      error.emit() << "Could not compile query module statically (Pipe could not be closed)";
      return;
   }
   void* handle = dlopen(std::string(currPath.string() + "/llvm-jit-static.so").c_str(), RTLD_LAZY);
   const char* dlsymError = dlerror();
   if (dlsymError) {
      error.emit() << "Can not open static library: " << std::string(dlsymError);
      return;
   }
   mainFunc = reinterpret_cast<execution::mainFnType>(dlsym(handle, "main"));
   dlsymError = dlerror();
   if (dlsymError) {
      dlclose(handle);
      error.emit() << "Could not load symbol for main function: " << std::string(dlsymError);
      return;
   }
   setExecutionContextFn = reinterpret_cast<execution::setExecutionContextFnType>(dlsym(handle, "rt_set_execution_context"));
   dlsymError = dlerror();
   if (dlsymError) {
      dlclose(handle);
      error.emit() << "Could not load symbol for rt_set_execution_context function: " << std::string(dlsymError);
      return;
   }
   return;
}
#if GPU_ENABLED == 1
static bool lowerToLLVMWithGPU(mlir::ModuleOp& moduleOp, bool verify) {
   LLVMInitializeNVPTXTarget();
   LLVMInitializeNVPTXTargetInfo();
   LLVMInitializeNVPTXTargetMC();
   LLVMInitializeNVPTXAsmPrinter();

   mlir::PassManager pm2(moduleOp->getContext());
   pm2.enableVerifier(verify);

   //pm2.enableIRPrinting();

   pm2.addPass(mlir::createConvertSCFToCFPass());
   pm2.addPass(mlir::createConvertIndexToLLVMPass());
   pm2.addNestedPass<mlir::func::FuncOp>(mlir::util::createUtilToLLVMPass());
   //pm2.addPass(mlir::util::createUtilToLLVMPass());
   pm2.addPass(mlir::createGpuKernelOutliningPass());
   pm2.addPass(mlir::createGpuNVVMAttachTarget(mlir::GpuNVVMAttachTargetOptions{.chip = "sm_35"}));
   pm2.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createStripDebugInfoPass());
   pm2.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps());
   pm2.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createReconcileUnrealizedCastsPass());
   pm2.addNestedPass<mlir::func::FuncOp>(mlir::createGpuAsyncRegionPass());
   pm2.addPass(mlir::createGpuToLLVMConversionPass());
   pm2.addPass(mlir::createGpuModuleToBinaryPass());
   pm2.addPass(mlir::createAsyncToAsyncRuntimePass());
   pm2.addPass(mlir::createAsyncRuntimeRefCountingPass());
   pm2.addPass(mlir::createConvertAsyncToLLVMPass());

   pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
   pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
   pm2.addPass(mlir::createArithToLLVMConversionPass());
   pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
   pm2.addPass(mlir::createConvertFuncToLLVMPass());
   pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
   pm2.addNestedPass<mlir::LLVM::LLVMFuncOp>(execution::createEnforceCABI());
   pm2.addPass(mlir::createCSEPass());
   if (mlir::failed(pm2.run(moduleOp))) {
      return false;
   }
   return true;
}

class GPULLVMBackend : public execution::ExecutionBackend {
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      mlir::registerBuiltinDialectTranslation(*moduleOp->getContext());
      mlir::registerLLVMDialectTranslation(*moduleOp->getContext());
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      auto startLowerToLLVM = std::chrono::high_resolution_clock::now();
      if (!lowerToLLVMWithGPU(moduleOp, verify)) {
         error.emit() << "Could not lower module to llvm dialect";
         return;
      }
      addLLVMExecutionContextFuncs(moduleOp);
      auto endLowerToLLVM = std::chrono::high_resolution_clock::now();
      timing["lowerToLLVM"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerToLLVM - startLowerToLLVM).count() / 1000.0;
      double translateToLLVMIRTime;
      auto convertFn = [&](mlir::Operation* module, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
         auto startTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         auto res = translateModuleToLLVMIR(module, context, "LLVMDialectModule", false);
         auto endTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         translateToLLVMIRTime = std::chrono::duration_cast<std::chrono::microseconds>(endTranslationToLLVMIR - startTranslationToLLVMIR).count() / 1000.0;
         return std::move(res);
      };
      double llvmPassesTime;

      auto optimizeFn = [&](llvm::Module* module) -> llvm::Error {
         module->dump();
         auto startLLVMIRPasses = std::chrono::high_resolution_clock::now();
         auto error = performDefaultLLVMPasses(module);
         auto endLLVMIRPasses = std::chrono::high_resolution_clock::now();
         llvmPassesTime = std::chrono::duration_cast<std::chrono::microseconds>(endLLVMIRPasses - startLLVMIRPasses).count() / 1000.0;
         return error;
      };
      auto startJIT = std::chrono::high_resolution_clock::now();
      // Libraries that we'll pass to the ExecutionEngine for loading.
      std::list<std::string> libPaths = {RUNNER_UTILS_LIB, C_RUNNER_UTILS_LIB, ASYNC_RUNTIME_LIB, CUDA_RUNTIME_LIB};
      llvm::SmallVector<llvm::StringRef, 4> executionEngineLibs;

      using MlirRunnerInitFn = void (*)(llvm::StringMap<void*>&);
      using MlirRunnerDestroyFn = void (*)();

      llvm::StringMap<void*> exportSymbols;
      llvm::SmallVector<MlirRunnerDestroyFn> destroyFns;

      // Handle libraries that do support mlir-runner init/destroy callbacks.
      for (auto& libPath : libPaths) {
         auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(libPath.c_str());
         void* initSym = lib.getAddressOfSymbol("__mlir_runner_init");
         void* destroySim = lib.getAddressOfSymbol("__mlir_runner_destroy");

         // Library does not support mlir runner, load it with ExecutionEngine.
         if (!initSym || !destroySim) {
            executionEngineLibs.push_back(libPath);
            continue;
         }

         auto initFn = reinterpret_cast<MlirRunnerInitFn>(initSym);
         initFn(exportSymbols);

         auto destroyFn = reinterpret_cast<MlirRunnerDestroyFn>(destroySim);
         destroyFns.push_back(destroyFn);
      }
      auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
      if (!tmBuilderOrError) {
         llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
         assert(false && "should not happen");
      }
      auto tmOrError = tmBuilderOrError->createTargetMachine();
      if (!tmOrError) {
         llvm::errs() << "Failed to create a TargetMachine for the host\n";
         assert(false && "should not happen");
      }

      auto maybeEngine = mlir::ExecutionEngine::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = mlir::makeOptimizingTransformer(0, 0, tmOrError->get()), .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .sharedLibPaths = executionEngineLibs, .enableObjectDump = false});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());
      auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
         auto symbolMap = llvm::orc::SymbolMap();
         mlir::util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         execution::visitBareFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         for (auto& exportSymbol : exportSymbols)
            symbolMap[interner(exportSymbol.getKey())] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(exportSymbol.getValue()), llvm::JITSymbolFlags::Exported);
         return symbolMap;
      };
      engine->registerSymbols(runtimeSymbolMap);
      auto mainFnLookupResult = engine->lookup("main");
      if (!mainFnLookupResult) {
         error.emit() << "Could not lookup main function";
         return;
      }
      auto setExecutionContextLookup = engine->lookup("rt_set_execution_context");
      if (!setExecutionContextLookup) {
         error.emit() << "Could not lookup function for setting the execution context";
         return;
      }
      auto mainFunc = reinterpret_cast<execution::mainFnType>(mainFnLookupResult.get());
      auto setExecutionContextFunc = reinterpret_cast<execution::setExecutionContextFnType>(setExecutionContextLookup.get());
      auto endJIT = std::chrono::high_resolution_clock::now();
      setExecutionContextFunc(executionContext);
      auto totalJITTime = std::chrono::duration_cast<std::chrono::microseconds>(endJIT - startJIT).count() / 1000.0;
      totalJITTime -= translateToLLVMIRTime;
      totalJITTime -= llvmPassesTime;

      std::vector<double> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         utility::Tracer::Trace trace(execution);
         mainFunc();
         trace.stop();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         executionContext->reset();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      timing["toLLVMIR"] = translateToLLVMIRTime;
      timing["llvmOptimize"] = llvmPassesTime;
      timing["llvmCodeGen"] = totalJITTime;
      timing["executionTime"] = (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]);
   }
   bool requiresSnapshotting() override {
      return false;
   }
};
#endif

class DefaultCPULLVMBackend : public execution::ExecutionBackend {
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      mlir::registerBuiltinDialectTranslation(*moduleOp->getContext());
      mlir::registerLLVMDialectTranslation(*moduleOp->getContext());
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      auto startLowerToLLVM = std::chrono::high_resolution_clock::now();
      if (!lowerToLLVMDialect(moduleOp, verify)) {
         error.emit() << "Could not lower module to llvm dialect";
         return;
      }
      addLLVMExecutionContextFuncs(moduleOp);
      auto endLowerToLLVM = std::chrono::high_resolution_clock::now();
      timing["lowerToLLVM"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerToLLVM - startLowerToLLVM).count() / 1000.0;
      double translateToLLVMIRTime;
      auto convertFn = [&](mlir::Operation* module, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
         auto startTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         auto res = translateModuleToLLVMIR(module, context, "LLVMDialectModule", false);
         auto endTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         translateToLLVMIRTime = std::chrono::duration_cast<std::chrono::microseconds>(endTranslationToLLVMIR - startTranslationToLLVMIR).count() / 1000.0;
         return std::move(res);
      };
      double llvmPassesTime;

      auto optimizeFn = [&](llvm::Module* module) -> llvm::Error {
         auto startLLVMIRPasses = std::chrono::high_resolution_clock::now();
         auto error = performDefaultLLVMPasses(module);
         auto endLLVMIRPasses = std::chrono::high_resolution_clock::now();
         llvmPassesTime = std::chrono::duration_cast<std::chrono::microseconds>(endLLVMIRPasses - startLLVMIRPasses).count() / 1000.0;
         return error;
      };
      auto startJIT = std::chrono::high_resolution_clock::now();

      auto maybeEngine = mlir::ExecutionEngine::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .enableObjectDump = false});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());
      auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
         auto symbolMap = llvm::orc::SymbolMap();
         mlir::util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         execution::visitBareFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         return symbolMap;
      };
      engine->registerSymbols(runtimeSymbolMap);
      auto mainFnLookupResult = engine->lookup("main");
      if (!mainFnLookupResult) {
         error.emit() << "Could not lookup main function";
         return;
      }
      auto setExecutionContextLookup = engine->lookup("rt_set_execution_context");
      if (!setExecutionContextLookup) {
         error.emit() << "Could not lookup function for setting the execution context";
         return;
      }
      auto mainFunc = reinterpret_cast<execution::mainFnType>(mainFnLookupResult.get());
      auto setExecutionContextFunc = reinterpret_cast<execution::setExecutionContextFnType>(setExecutionContextLookup.get());
      auto endJIT = std::chrono::high_resolution_clock::now();
      setExecutionContextFunc(executionContext);
      auto totalJITTime = std::chrono::duration_cast<std::chrono::microseconds>(endJIT - startJIT).count() / 1000.0;
      totalJITTime -= translateToLLVMIRTime;
      totalJITTime -= llvmPassesTime;

      std::vector<double> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         utility::Tracer::Trace trace(execution);
         mainFunc();
         trace.stop();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         executionContext->reset();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      timing["toLLVMIR"] = translateToLLVMIRTime;
      timing["llvmOptimize"] = llvmPassesTime;
      timing["llvmCodeGen"] = totalJITTime;
      timing["executionTime"] = (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]);
   }
   bool requiresSnapshotting() override {
      return false;
   }
};
static void snapshot(mlir::ModuleOp moduleOp, execution::Error& error, std::string fileName) {
   mlir::PassManager pm(moduleOp->getContext());
   mlir::OpPrintingFlags flags;
   flags.enableDebugInfo(true, false);
   pm.addPass(mlir::createLocationSnapshotPass(flags, fileName));
   if (pm.run(moduleOp).failed()) {
      error.emit() << "Snapshotting failed";
   }
}
static void addDebugInfo(mlir::ModuleOp module, std::string lastSnapShotFile) {
   auto fileAttr = mlir::LLVM::DIFileAttr::get(module->getContext(), lastSnapShotFile, std::filesystem::current_path().string());
   auto compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(mlir::DistinctAttr::create(mlir::UnitAttr::get(module->getContext())), static_cast<uint8_t>(llvm::dwarf::DW_LANG_C), fileAttr, mlir::StringAttr::get(module->getContext(), "LingoDB"), true, mlir::LLVM::DIEmissionKind::Full);
   module->walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
      mlir::DistinctAttr id;
      mlir::LLVM::DICompileUnitAttr compileUnitAt;
      auto subprogramFlags = mlir::LLVM::DISubprogramFlags::Optimized;
      if (!funcOp.isExternal()) {
         id = mlir::DistinctAttr::create(mlir::UnitAttr::get(module.getContext()));
         subprogramFlags = subprogramFlags | mlir::LLVM::DISubprogramFlags::Definition;
         compileUnitAt = compileUnitAttr;
      }
      auto subroutineType = mlir::LLVM::DISubroutineTypeAttr::get(module->getContext(), {});
      auto subProgramAttr = mlir::LLVM::DISubprogramAttr::get(id, compileUnitAt, fileAttr, funcOp.getName(), funcOp.getName(), fileAttr, 0, 0, subprogramFlags, subroutineType);
      funcOp->setLoc(mlir::FusedLocWith<mlir::LLVM::DIScopeAttr>::get(funcOp->getLoc(), subProgramAttr, module->getContext()));
   });
}
class CPULLVMDebugBackend : public execution::ExecutionBackend {
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      mlir::registerBuiltinDialectTranslation(*moduleOp->getContext());
      mlir::registerLLVMDialectTranslation(*moduleOp->getContext());
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      auto startLowerToLLVM = std::chrono::high_resolution_clock::now();
      if (!lowerToLLVMDialect(moduleOp, verify)) {
         error.emit() << "Could not lower module to llvm dialect";
         return;
      }
      addLLVMExecutionContextFuncs(moduleOp);
      auto endLowerToLLVM = std::chrono::high_resolution_clock::now();
      auto llvmSnapshotFile = "snapshot-" + std::to_string(snapShotCounter) + ".mlir";
      snapshot(moduleOp, error, llvmSnapshotFile);
      addDebugInfo(moduleOp, llvmSnapshotFile);
      timing["lowerToLLVM"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerToLLVM - startLowerToLLVM).count() / 1000.0;
      auto convertFn = [&](mlir::Operation* module, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
         return translateModuleToLLVMIR(module, context, "LLVMDialectModule", true);
      };
      //do not optimize in debug mode
      auto optimizeFn = [&](llvm::Module* module) -> llvm::Error { return llvm::Error::success(); };
      //first step: use ExecutionEngine
      auto maybeEngine = mlir::ExecutionEngine::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .enableObjectDump = true});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());
      auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
         auto symbolMap = llvm::orc::SymbolMap();
         mlir::util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         execution::visitBareFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         return symbolMap;
      };
      engine->registerSymbols(runtimeSymbolMap);
      auto mainFnLookupResult = engine->lookup("main");
      if (!mainFnLookupResult) {
         error.emit() << "Could not lookup main function";
         return;
      }
      auto setExecutionContextLookup = engine->lookup("rt_set_execution_context");
      if (!setExecutionContextLookup) {
         error.emit() << "Could not lookup function for setting the execution context";
         return;
      }
      execution::mainFnType mainFunc;
      execution::setExecutionContextFnType setExecutionContextFunc;
      linkStatic(engine.get(), error, mainFunc, setExecutionContextFunc);
      if (error) {
         return;
      }
      setExecutionContextFunc(executionContext);

      std::vector<double> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         executionContext->reset();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      timing["executionTime"] = (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]);
   }
   bool requiresSnapshotting() override {
      return true;
   }
};

class CPULLVMProfilingBackend : public execution::ExecutionBackend {
   inline void assignToThisCore(int coreId) {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(coreId, &mask);
      sched_setaffinity(0, sizeof(mask), &mask);
   }

   pid_t runPerfRecord() {
      assignToThisCore(9);
      pid_t childPid = 0;
      auto parentPid = std::to_string(getpid());
      const char* argV[] = {"perf", "record", "-R", "-e", "ibs_op//p", "-c", "5000", "--intr-regs=r15", "-C", "9", nullptr};
      auto status = posix_spawn(&childPid, "/usr/bin/perf", nullptr, nullptr, const_cast<char**>(argV), environ);
      sleep(5);
      assignToThisCore(9);
      if (status != 0)
         error.emit() << "Launching of perf failed" << status;
      return childPid;
   }
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      mlir::registerBuiltinDialectTranslation(*moduleOp->getContext());
      mlir::registerLLVMDialectTranslation(*moduleOp->getContext());
      LLVMInitializeX86AsmParser();
      reserveLastRegister = true;
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      auto targetTriple = llvm::sys::getDefaultTargetTriple();
      std::string errorMessage;
      const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
      if (!target) {
         error.emit() << "Could not lookup target";
         return;
      }

      // Initialize LLVM targets.
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      auto startLowerToLLVM = std::chrono::high_resolution_clock::now();
      if (!lowerToLLVMDialect(moduleOp, verify)) {
         error.emit() << "Could not lower module to llvm dialect";
         return;
      }
      addLLVMExecutionContextFuncs(moduleOp);
      auto endLowerToLLVM = std::chrono::high_resolution_clock::now();
      timing["lowerToLLVM"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerToLLVM - startLowerToLLVM).count() / 1000.0;
      auto llvmSnapshotFile = "snapshot-" + std::to_string(snapShotCounter) + ".mlir";
      snapshot(moduleOp, error, llvmSnapshotFile);
      addDebugInfo(moduleOp, llvmSnapshotFile);
      mlir::PassManager pm(moduleOp->getContext());
      pm.addPass(execution::createAnnotateProfilingDataPass());
      if (mlir::failed(pm.run(moduleOp))) {
         error.emit() << "Could not annotate profiling information";
         return;
      }
      double translateToLLVMIRTime;
      auto convertFn = [&](mlir::Operation* module, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
         auto startTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         auto res = translateModuleToLLVMIR(module, context, "LLVMDialectModule", true);
         auto endTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         translateToLLVMIRTime = std::chrono::duration_cast<std::chrono::microseconds>(endTranslationToLLVMIR - startTranslationToLLVMIR).count() / 1000.0;
         return std::move(res);
      };
      double llvmPassesTime;

      auto optimizeFn = [&](llvm::Module* module) -> llvm::Error {
         auto startLLVMIRPasses = std::chrono::high_resolution_clock::now();
         auto error = performDefaultLLVMPasses(module);
         auto endLLVMIRPasses = std::chrono::high_resolution_clock::now();
         llvmPassesTime = std::chrono::duration_cast<std::chrono::microseconds>(endLLVMIRPasses - startLLVMIRPasses).count() / 1000.0;
         return error;
      };
      auto startJIT = std::chrono::high_resolution_clock::now();

      auto maybeEngine = mlir::ExecutionEngine::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .enableObjectDump = true});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());
      auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
         auto symbolMap = llvm::orc::SymbolMap();
         mlir::util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         execution::visitBareFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         return symbolMap;
      };
      engine->registerSymbols(runtimeSymbolMap);
      auto mainFnLookupResult = engine->lookup("main");
      if (!mainFnLookupResult) {
         error.emit() << "Could not lookup main function";
         return;
      }
      auto setExecutionContextLookup = engine->lookup("rt_set_execution_context");
      if (!setExecutionContextLookup) {
         error.emit() << "Could not lookup function for setting the execution context";
         return;
      }
      execution::mainFnType mainFunc;
      execution::setExecutionContextFnType setExecutionContextFunc;
      linkStatic(engine.get(), error, mainFunc, setExecutionContextFunc);
      if (error) {
         return;
      }
      auto endJIT = std::chrono::high_resolution_clock::now();
      setExecutionContextFunc(executionContext);
      auto totalJITTime = std::chrono::duration_cast<std::chrono::microseconds>(endJIT - startJIT).count() / 1000.0;
      totalJITTime -= translateToLLVMIRTime;
      totalJITTime -= llvmPassesTime;

      //start profiling
      pid_t pid = runPerfRecord();
      if (error) return;
      uint64_t r15DefaultValue = 0xbadeaffe;
      __asm__ __volatile__("mov %0, %%r15\n\t"
                           : /* no output */
                           : "a"(r15DefaultValue)
                           : "%r15");
      std::vector<double> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         executionContext->reset();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      //finish profiling
      reserveLastRegister = false;
      kill(pid, SIGINT);
      sleep(2);

      timing["toLLVMIR"] = translateToLLVMIRTime;
      timing["llvmOptimize"] = llvmPassesTime;
      timing["llvmCodeGen"] = totalJITTime;
      timing["executionTime"] = (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]);
   }
   bool requiresSnapshotting() override {
      return true;
   }
};

} // namespace

std::unique_ptr<execution::ExecutionBackend> execution::createDefaultLLVMBackend() {
   return std::make_unique<DefaultCPULLVMBackend>();
}
std::unique_ptr<execution::ExecutionBackend> execution::createLLVMDebugBackend() {
   return std::make_unique<CPULLVMDebugBackend>();
}
std::unique_ptr<execution::ExecutionBackend> execution::createLLVMProfilingBackend() {
   return std::make_unique<CPULLVMProfilingBackend>();
}

std::unique_ptr<execution::ExecutionBackend> execution::createGPULLVMBackend() {
#if GPU_ENABLED == 1
   return std::make_unique<GPULLVMBackend>();
#else
   return {};
#endif
}