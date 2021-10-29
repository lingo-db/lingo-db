#include "runner/jit.h"

#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

namespace {
using JIT = runner::JIT;

static void optimizeModule(llvm::Module& module) {
   llvm::legacy::PassManager modulePMInline;
   modulePMInline.add(llvm::createAlwaysInlinerLegacyPass());
   modulePMInline.run(module);
   // Create a function pass manager
   llvm::legacy::PassManager modulePM;
   llvm::legacy::FunctionPassManager funcPM(&module);
   funcPM.add(llvm::createInstructionCombiningPass());
   funcPM.add(llvm::createReassociatePass());
   funcPM.add(llvm::createGVNPass());
   funcPM.add(llvm::createCFGSimplificationPass());
   funcPM.add(llvm::createAggressiveDCEPass());
   funcPM.add(llvm::createCFGSimplificationPass());

   funcPM.doInitialization();
   for (auto& func : module) {
      funcPM.run(func);
   }
   funcPM.doFinalization();
   modulePM.add(llvm::createAlwaysInlinerLegacyPass());
   modulePM.add(llvm::createGlobalDCEPass());
   modulePM.run(module);
}

} // namespace
static llvm::orc::JITDylib& checkAndGet(llvm::Expected<llvm::orc::JITDylib&> lib) {
   lib.operator bool();
   return lib.get();
}
static std::unique_ptr<llvm::orc::SelfExecutorProcessControl> createEPC() {
   auto epc = llvm::orc::SelfExecutorProcessControl::Create();
   if (!epc) {
      return nullptr;
   }
   return std::move(epc.get());
}
static std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
   auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
   if (!tmBuilderOrError) {
      llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
      return nullptr;
   }
   auto tmOrError = tmBuilderOrError->createTargetMachine();
   if (!tmOrError) {
      llvm::errs() << "Failed to create a TargetMachine for the host\n";
      return nullptr;
   }
   return std::move(tmOrError.get());
}
JIT::JIT(llvm::orc::ThreadSafeContext& ctx)
   : targetMachine(createTargetMachine()),
     dataLayout(targetMachine->createDataLayout()),
     executionSession(std::move(createEPC())),
     context(ctx),
     objectLinkingLayer(executionSession, []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
     compileLayer(executionSession, objectLinkingLayer, std::make_unique<llvm::orc::SimpleCompiler>(*targetMachine)),
     optimizeLayer(executionSession, compileLayer, [](llvm::orc::ThreadSafeModule m, const llvm::orc::MaterializationResponsibility&) { optimizeModule(*m.getModuleUnlocked()); return m; }),
     mainDylib(checkAndGet(executionSession.createJITDylib("<main>"))) {
   //objectLinkingLayer.registerJITEventListener(*llvm::JITEventListener::createPerfJITEventListener());
   // Lookup symbols in host process

   auto generator = llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      dataLayout.getGlobalPrefix(),
      [](auto&) { return true; }));
   mainDylib.addGenerator(move(generator));
}

llvm::Error JIT::addModule(std::unique_ptr<llvm::Module> module) {
   llvm::legacy::PassManager modulePM;
   modulePM.add(llvm::createInternalizePass([&](const llvm::GlobalValue& gv) { return gv.getName() == "main" || gv.getName() == "rt_set_execution_context" || gv.isDeclaration(); }));
   modulePM.run(*module);
   return optimizeLayer.add(mainDylib, llvm::orc::ThreadSafeModule{move(module), context});
}

void* JIT::getPointerToFunction(const std::string& name) {
   auto sym = executionSession.lookup(&mainDylib, name);
   return sym ? reinterpret_cast<void*>(static_cast<uintptr_t>(sym->getAddress())) : nullptr;
}