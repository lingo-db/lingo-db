#include "runner/jit.h"

#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

namespace {
using JIT=runner::JIT;

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

}  // namespace
static llvm::orc::JITDylib& checkAndGet(llvm::Expected<llvm::orc::JITDylib&> lib){
   lib.operator bool();
   return lib.get();
}
JIT::JIT(llvm::orc::ThreadSafeContext& ctx)
   : target_machine(llvm::EngineBuilder().selectTarget()),
     data_layout(target_machine->createDataLayout()),
     execution_session(),
     context(ctx),
     object_layer(execution_session, []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
     compile_layer(execution_session, object_layer, std::make_unique<llvm::orc::SimpleCompiler>(*target_machine)),
     optimize_layer(execution_session, compile_layer, [] (llvm::orc::ThreadSafeModule m, const llvm::orc::MaterializationResponsibility&) { optimizeModule(*m.getModuleUnlocked()); return m; }),
     mainDylib(checkAndGet(execution_session.createJITDylib("<main>"))) {

   // Lookup symbols in host process
   auto generator = llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      data_layout.getGlobalPrefix(),
      [](auto&) { return true; }));
   mainDylib.addGenerator(move(generator));
}

llvm::Error JIT::addModule(std::unique_ptr<llvm::Module> module) {
   llvm::legacy::PassManager modulePM;
   modulePM.add(llvm::createInternalizePass([&](const llvm::GlobalValue& gv){return gv.getName()=="main"||gv.isDeclaration();}));
   modulePM.run(*module);
   return optimize_layer.add(mainDylib, llvm::orc::ThreadSafeModule{move(module), context});
}

void* JIT::getPointerToFunction(const std::string& name) {
   auto sym = execution_session.lookup(&mainDylib, name);
   return sym ? reinterpret_cast<void*>(static_cast<uintptr_t>(sym->getAddress())) : nullptr;
}