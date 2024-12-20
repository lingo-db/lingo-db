#include "lingodb/execution/LLVMBackends.h"

#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/execution/BackendPasses.h"
#include "lingodb/execution/Error.h"
#include "lingodb/execution/Instrumentation.h"
#include "lingodb/runtime/GPU/Properties.h"
#include "lingodb/utility/Setting.h"
#include "lingodb/utility/Tracer.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include <csignal>
#include <filesystem>
#include <fstream>

#include <dlfcn.h>
#include <spawn.h>
#include <unistd.h>

namespace {
utility::Tracer::Event execution("Execution", "run");

utility::Tracer::Event llvmCodeGen("Compilation", "LLVMCodeGen");
utility::Tracer::Event llvmOpt("Compilation", "LLVMOptPasses");

utility::GlobalSetting<bool> runLoweringPasses("system.compilation.llvm_lowering", true);
utility::GlobalSetting<std::string> perfFile("system.execution.perf_file", "perf.data");
utility::GlobalSetting<std::string> perfBinary("system.execution.perf_binary", "/usr/bin/perf");

namespace {
using namespace lingodb;
using namespace lingodb::compiler::dialect;
static llvm::Error makeStringError(const llvm::Twine& message) {
   return llvm::make_error<llvm::StringError>(message.str(),
                                              llvm::inconvertibleErrorCode());
}

class LLVMBackend {
   public:
   /// Name of init functions of shared libraries. If a library provides a
   /// function with this name and the one of the destroy function, this function
   /// is called upon loading the library.
   static constexpr const char* const kLibraryInitFnName =
      "__mlir_execution_engine_init";

   /// Name of destroy functions of shared libraries. If a library provides a
   /// function with this name and the one of the init function, this function is
   /// called upon destructing the `LLVMBackend`.
   static constexpr const char* const kLibraryDestroyFnName =
      "__mlir_execution_engine_destroy";

   /// Function type for init functions of shared libraries. The library may
   /// provide a list of symbols that it wants to make available to code run by
   /// the `LLVMBackend`. If the two functions are not defined, only symbols
   /// with public visibility are available to the executed code.
   using LibraryInitFn = void (*)(llvm::StringMap<void*>&);

   /// Function type for destroy functions of shared libraries.
   using LibraryDestroyFn = void (*)();

   LLVMBackend(bool enableObjectDump, bool enableGDBNotificationListener,
               bool enablePerfNotificationListener) : cache(enableObjectDump ? new mlir::SimpleObjectCache() : nullptr),
                                                      functionNames(),
                                                      gdbListener(enableGDBNotificationListener ? llvm::JITEventListener::createGDBRegistrationListener() : nullptr),
                                                      perfListener(nullptr) {
      if (enablePerfNotificationListener) {
         if (auto* listener = llvm::JITEventListener::createPerfJITEventListener())
            perfListener = listener;
         else if (auto* listener = llvm::JITEventListener::createIntelJITEventListener())
            perfListener = listener;
      }
   }

   ~LLVMBackend() {
      // Execute the global destructors from the module being processed.
      // TODO: Allow JIT deinitialize for AArch64. Currently there's a bug causing a
      // crash for AArch64 see related issue #71963.
      if (jit && !jit->getTargetTriple().isAArch64())
         llvm::consumeError(jit->deinitialize(jit->getMainJITDylib()));
      // Run all dynamic library destroy callbacks to prepare for the shutdown.
      for (LibraryDestroyFn destroy : destroyFns)
         destroy();
   }

   /// Creates an execution engine for the given MLIR IR. If TargetMachine is
   /// not provided, default TM is created (i.e. ignoring any command line flags
   /// that could affect the set-up).
   static llvm::Expected<std::unique_ptr<LLVMBackend>>
   create(mlir::Operation* op, const mlir::ExecutionEngineOptions& options = {},
          std::unique_ptr<llvm::TargetMachine> tm = nullptr) {
      auto engine = std::make_unique<LLVMBackend>(
         options.enableObjectDump, options.enableGDBNotificationListener,
         options.enablePerfNotificationListener);

      std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
      auto llvmModule = options.llvmModuleBuilder(op, *ctx);
      if (!llvmModule)
         return makeStringError("could not convert to LLVM IR");

      // If no valid TargetMachine was passed, create a default TM ignoring any
      // input arguments from the user.
      if (!tm) {
         auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
         if (!tmBuilderOrError)
            return tmBuilderOrError.takeError();

         auto tmOrError = tmBuilderOrError->createTargetMachine();
         if (!tmOrError)
            return tmOrError.takeError();
         tm = std::move(tmOrError.get());
      }

      // TODO: Currently, the LLVM module created above has no triple associated
      // with it. Instead, the triple is extracted from the TargetMachine, which is
      // either based on the host defaults or command line arguments when specified
      // (set-up by callers of this method). It could also be passed to the
      // translation or dialect conversion instead of this.
      setupTargetTripleAndDataLayout(llvmModule.get(), tm.get());

      auto dataLayout = llvmModule->getDataLayout();

      // Use absolute library path so that gdb can find the symbol table.
      llvm::SmallVector<llvm::SmallString<256>, 4> sharedLibPaths;
      transform(
         options.sharedLibPaths, std::back_inserter(sharedLibPaths),
         [](llvm::StringRef libPath) {
            llvm::SmallString<256> absPath(libPath.begin(), libPath.end());
            cantFail(llvm::errorCodeToError(llvm::sys::fs::make_absolute(absPath)));
            return absPath;
         });

      // If shared library implements custom execution layer library init and
      // destroy functions, we'll use them to register the library. Otherwise, load
      // the library as JITDyLib below.
      llvm::StringMap<void*> exportSymbols;
      llvm::SmallVector<LibraryDestroyFn> destroyFns;
      llvm::SmallVector<llvm::StringRef> jitDyLibPaths;

      for (auto& libPath : sharedLibPaths) {
         auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
            libPath.str().str().c_str());
         void* initSym = lib.getAddressOfSymbol(kLibraryInitFnName);
         void* destroySim = lib.getAddressOfSymbol(kLibraryDestroyFnName);

         // Library does not provide call backs, rely on symbol visiblity.
         if (!initSym || !destroySim) {
            jitDyLibPaths.push_back(libPath);
            continue;
         }

         auto initFn = reinterpret_cast<LibraryInitFn>(initSym);
         initFn(exportSymbols);

         auto destroyFn = reinterpret_cast<LibraryDestroyFn>(destroySim);
         destroyFns.push_back(destroyFn);
      }
      engine->destroyFns = std::move(destroyFns);

      // Callback to create the object layer with symbol resolution to current
      // process and dynamically linked libraries.
      auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession& session,
                                           const llvm::Triple& tt) {
         auto objectLayer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
            session, [sectionMemoryMapper = options.sectionMemoryMapper]() {
               return std::make_unique<llvm::SectionMemoryManager>(sectionMemoryMapper);
            });

         // Register JIT event listeners if they are enabled.
         if (engine->gdbListener)
            objectLayer->registerJITEventListener(*engine->gdbListener);
         if (engine->perfListener)
            objectLayer->registerJITEventListener(*engine->perfListener);

         // COFF format binaries (Windows) need special handling to deal with
         // exported symbol visibility.
         // cf llvm/lib/ExecutionEngine/Orc/LLJIT.cpp LLJIT::createObjectLinkingLayer
         llvm::Triple targetTriple(llvm::Twine(llvmModule->getTargetTriple()));
         if (targetTriple.isOSBinFormatCOFF()) {
            objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
            objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
         }

         // Resolve symbols from shared libraries.
         for (auto& libPath : jitDyLibPaths) {
            auto mb = llvm::MemoryBuffer::getFile(libPath);
            if (!mb) {
               llvm::errs() << "Failed to create MemoryBuffer for: " << libPath
                            << "\nError: " << mb.getError().message() << "\n";
               continue;
            }
            auto& jd = session.createBareJITDylib(std::string(libPath));
            auto loaded = llvm::orc::DynamicLibrarySearchGenerator::Load(
               libPath.str().c_str(), dataLayout.getGlobalPrefix());
            if (!loaded) {
               llvm::errs() << "Could not load " << libPath << ":\n  " << loaded.takeError()
                            << "\n";
               continue;
            }
            jd.addGenerator(std::move(*loaded));
            llvm::cantFail(objectLayer->add(jd, std::move(mb.get())));
         }

         return objectLayer;
      };

      // Callback to inspect the cache and recompile on demand. This follows Lang's
      // LLJITWithObjectCache example.
      auto compileFunctionCreator = [&](llvm::orc::JITTargetMachineBuilder jtmb)
         -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
         if (options.jitCodeGenOptLevel)
            jtmb.setCodeGenOptLevel(*options.jitCodeGenOptLevel);
         return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(tm),
                                                                    engine->cache.get());
      };

      // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
      auto jit =
         cantFail(llvm::orc::LLJITBuilder()
                     .setCompileFunctionCreator(compileFunctionCreator)
                     .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                     .setDataLayout(dataLayout)
                     .create());

      // Add a ThreadSafemodule to the engine and return.
      llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
      if (options.transformer)
         llvm::cantFail(tsm.withModuleDo(
            [&](llvm::Module& module) { return options.transformer(&module); }));
      llvm::cantFail(jit->addIRModule(std::move(tsm)));
      engine->jit = std::move(jit);

      // Resolve symbols that are statically linked in the current process.
      llvm::orc::JITDylib& mainJD = engine->jit->getMainJITDylib();
      mainJD.addGenerator(
         llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            dataLayout.getGlobalPrefix())));

      // Build a runtime symbol map from the exported symbols and register them.
      auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
         auto symbolMap = llvm::orc::SymbolMap();
         for (auto& exportSymbol : exportSymbols)
            symbolMap[interner(exportSymbol.getKey())] = {
               llvm::orc::ExecutorAddr::fromPtr(exportSymbol.getValue()),
               llvm::JITSymbolFlags::Exported};
         return symbolMap;
      };
      engine->registerSymbols(runtimeSymbolMap);

      engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
         auto symbolMap = llvm::orc::SymbolMap();
         util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         lingodb::execution::visitBareFunctions([&](std::string s, void* ptr) {
            symbolMap[interner(s)] = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported);
         });
         return symbolMap;
      });

      // Execute the global constructors from the module being processed.
      // TODO: Allow JIT initialize for AArch64. Currently there's a bug causing a
      // crash for AArch64 see related issue #71963.
      if (!engine->jit->getTargetTriple().isAArch64())
         cantFail(engine->jit->initialize(engine->jit->getMainJITDylib()));

      return std::move(engine);
   }

   /// Looks up the original function with the given name and returns a
   /// pointer to it. This is not necesarily a packed function. Propagates
   /// errors in case of failure.
   llvm::Expected<void*> lookup(llvm::StringRef name) const {
      auto expectedSymbol = jit->lookup(name);

      // JIT lookup may return an Error referring to strings stored internally by
      // the JIT. If the Error outlives the LLVMBackend, it would want have a
      // dangling reference, which is currently caught by an assertion inside JIT
      // thanks to hand-rolled reference counting. Rewrap the error message into a
      // string before returning. Alternatively, ORC JIT should consider copying
      // the string into the error message.
      if (!expectedSymbol) {
         std::string errorMessage;
         llvm::raw_string_ostream os(errorMessage);
         llvm::handleAllErrors(expectedSymbol.takeError(),
                               [&os](llvm::ErrorInfoBase& ei) { ei.log(os); });
         return makeStringError(errorMessage);
      }

      if (void* fptr = expectedSymbol->toPtr<void*>())
         return fptr;
      return makeStringError("looked up function is null");
   }

   /// Set the target triple and the data layout for the input module based on
   /// the input TargetMachine. This is implicitly done when creating the
   /// engine.
   static void setupTargetTripleAndDataLayout(llvm::Module* llvmModule,
                                              llvm::TargetMachine* tm) {
      llvmModule->setDataLayout(tm->createDataLayout());
      llvmModule->setTargetTriple(tm->getTargetTriple().getTriple());
   }

   /// Dump object code to output file `filename`.
   void dumpToObjectFile(llvm::StringRef filename) {
      if (cache == nullptr) {
         llvm::errs() << "cannot dump ExecutionEngine object code to file: "
                         "object cache is disabled\n";
         return;
      }
      cache->dumpToObjectFile(filename);
   }

   private:
   /// Register symbols with this LLVMBackend.
   void registerSymbols(llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap) {
      auto& mainJitDylib = jit->getMainJITDylib();
      cantFail(mainJitDylib.define(
         absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
            mainJitDylib.getExecutionSession(), jit->getDataLayout())))));
   }

   private:
   /// Ordering of llvmContext and jit is important for destruction purposes: the
   /// jit must be destroyed before the context.
   llvm::LLVMContext llvmContext;

   /// Underlying LLJIT.
   std::unique_ptr<llvm::orc::LLJIT> jit;

   /// Underlying cache.
   std::unique_ptr<mlir::SimpleObjectCache> cache;

   /// Names of functions that may be looked up.
   std::vector<std::string> functionNames;

   /// GDB notification listener.
   llvm::JITEventListener* gdbListener;

   /// Perf notification listener.
   llvm::JITEventListener* perfListener;

   /// Destroy functions in the libraries loaded by the LLVMBackend that are
   /// called when this LLVMBackend is destructed.
   llvm::SmallVector<LibraryDestroyFn> destroyFns;
};

} // namespace

static bool lowerToLLVMDialect(mlir::ModuleOp& moduleOp, std::shared_ptr<execution::SnapshotState> serializationState, bool verify) {
   if (runLoweringPasses.getValue()) {
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
      lingodb::execution::addLingoDBInstrumentation(pm2, serializationState);
      pm2.addPass(mlir::createConvertSCFToCFPass());
      pm2.addPass(util::createUtilToLLVMPass());
      pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
      pm2.addPass(mlir::createArithToLLVMConversionPass());
      pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
      pm2.addPass(mlir::createArithToLLVMConversionPass());
      pm2.addPass(mlir::createConvertFuncToLLVMPass());
      pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
      pm2.addPass(lingodb::execution::createEnforceCABI());
      pm2.addPass(mlir::createCSEPass());
      if (mlir::failed(pm2.run(moduleOp))) {
         return false;
      }
      return true;
   } else {
      return true;
   }
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
   utility::Tracer::Trace trace(llvmOpt);
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

static void linkStatic(LLVMBackend* engine, execution::Error& error, execution::mainFnType& mainFunc, execution::setExecutionContextFnType& setExecutionContextFn) {
   auto currPath = std::filesystem::current_path();
   std::ofstream symbolfile("symbolfile");
   util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) {
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

   // pm2.enableIRPrinting();

   pm2.addPass(util::createUtilToLLVMPass());
   pm2.addPass(mlir::createConvertSCFToCFPass());
   // pm2.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
   pm2.addPass(mlir::createConvertIndexToLLVMPass());
   // pm2.addPass(mlir::createGpuKernelOutliningPass());
   llvm::SmallVector<std::string> linkFiles = {std::string(GPU_BC_DIR) + "/RTDeviceFuncs.bc"};
   pm2.addPass(mlir::createGpuNVVMAttachTarget(mlir::GpuNVVMAttachTargetOptions{
      .chip = lingodb::runtime::gpu::getChipStr(),
      .optLevel = 3,
      .linkLibs = linkFiles,
   }));
   pm2.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createStripDebugInfoPass());
   pm2.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps());
   pm2.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertIndexToLLVMPass());
   pm2.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createReconcileUnrealizedCastsPass());
   pm2.addPass(mlir::createGpuToLLVMConversionPass());
   pm2.addPass(mlir::createGpuModuleToBinaryPass(mlir::GpuModuleToBinaryPassOptions{.cmdOptions = ""}));

   pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
   pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
   pm2.addPass(mlir::createArithToLLVMConversionPass());
   pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
   pm2.addPass(mlir::createConvertFuncToLLVMPass());
   pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
   pm2.addPass(execution::createEnforceCABI());
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
         return res;
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
      // Libraries that we'll pass to the LLVMBackend for loading.
      llvm::SmallVector<llvm::StringRef, 4> requiredLibs = {CUDA_RUNTIME_LIB};

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

      auto maybeEngine = LLVMBackend::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .sharedLibPaths = requiredLibs, .enableObjectDump = false});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());
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
};
#endif

class DefaultCPULLVMBackend : public execution::ExecutionBackend {
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      mlir::registerBuiltinDialectTranslation(*moduleOp->getContext());
      mlir::registerLLVMDialectTranslation(*moduleOp->getContext());
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      auto startLowerToLLVM = std::chrono::high_resolution_clock::now();
      if (!lowerToLLVMDialect(moduleOp, getSerializationState(), verify)) {
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
         return res;
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
      utility::Tracer::Trace traceCodeGen(llvmCodeGen);

      auto maybeEngine = LLVMBackend::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .enableObjectDump = false});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());

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
      traceCodeGen.stop();
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
};

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
      auto subProgramAttr = mlir::LLVM::DISubprogramAttr::get(module->getContext(), id, compileUnitAt, fileAttr, funcOp.getNameAttr(), funcOp.getNameAttr(), fileAttr, 0, 0, subprogramFlags, subroutineType, {}, {});
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
      if (!lowerToLLVMDialect(moduleOp, getSerializationState(), verify)) {
         error.emit() << "Could not lower module to llvm dialect";
         return;
      }
      addLLVMExecutionContextFuncs(moduleOp);
      auto endLowerToLLVM = std::chrono::high_resolution_clock::now();
      if (auto moduleFileLineLoc = mlir::dyn_cast_or_null<mlir::FileLineColLoc>(moduleOp.getLoc())) {
         addDebugInfo(moduleOp, moduleFileLineLoc.getFilename().str());
      }
      timing["lowerToLLVM"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerToLLVM - startLowerToLLVM).count() / 1000.0;
      auto convertFn = [&](mlir::Operation* module, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
         return translateModuleToLLVMIR(module, context, "LLVMDialectModule", true);
      };
      //do not optimize in debug mode
      auto optimizeFn = [&](llvm::Module* module) -> llvm::Error { return llvm::Error::success(); };
      //first step: use LLVMBackend
      auto maybeEngine = LLVMBackend::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .enableObjectDump = true});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());
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
};

class CPULLVMProfilingBackend : public execution::ExecutionBackend {
   pid_t runPerfRecord() {
      pid_t childPid = 0;
      auto parentPid = std::to_string(getpid());
      auto perfBinaryStr = perfBinary.getValue();
      auto perfFileStr = perfFile.getValue();
      const char* argV[] = {perfBinaryStr.c_str(), "record", "-o", perfFileStr.c_str(), "-c", "5000", "-p", parentPid.c_str(), nullptr};
      auto status = posix_spawn(&childPid, perfBinaryStr.c_str(), nullptr, nullptr, const_cast<char**>(argV), environ);
      sleep(10);
      if (status != 0)
         error.emit() << "Launching of perf failed" << status;
      return childPid;
   }
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      mlir::registerBuiltinDialectTranslation(*moduleOp->getContext());
      mlir::registerLLVMDialectTranslation(*moduleOp->getContext());
      LLVMInitializeX86AsmParser();
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
      if (!lowerToLLVMDialect(moduleOp, getSerializationState(), verify)) {
         error.emit() << "Could not lower module to llvm dialect";
         return;
      }
      addLLVMExecutionContextFuncs(moduleOp);
      auto endLowerToLLVM = std::chrono::high_resolution_clock::now();
      timing["lowerToLLVM"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerToLLVM - startLowerToLLVM).count() / 1000.0;
      if (auto moduleFileLineLoc = mlir::dyn_cast_or_null<mlir::FileLineColLoc>(moduleOp.getLoc())) {
         addDebugInfo(moduleOp, moduleFileLineLoc.getFilename().str());
      }
      double translateToLLVMIRTime;
      auto convertFn = [&](mlir::Operation* module, llvm::LLVMContext& context) -> std::unique_ptr<llvm::Module> {
         auto startTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         auto res = translateModuleToLLVMIR(module, context, "LLVMDialectModule", true);
         auto endTranslationToLLVMIR = std::chrono::high_resolution_clock::now();
         translateToLLVMIRTime = std::chrono::duration_cast<std::chrono::microseconds>(endTranslationToLLVMIR - startTranslationToLLVMIR).count() / 1000.0;
         return res;
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

      auto maybeEngine = LLVMBackend::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default, .enableObjectDump = true});
      if (!maybeEngine) {
         error.emit() << "Could not create execution engine";
         return;
      }
      auto engine = std::move(maybeEngine.get());
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

      std::vector<double> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         executionContext->reset();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      //finish profiling
      kill(pid, SIGINT);
      sleep(2);

      timing["toLLVMIR"] = translateToLLVMIRTime;
      timing["llvmOptimize"] = llvmPassesTime;
      timing["llvmCodeGen"] = totalJITTime;
      timing["executionTime"] = (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]);
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