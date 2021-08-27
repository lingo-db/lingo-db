#ifndef RUNNER_JIT_H
#define RUNNER_JIT_H

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Mangler.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace runner {

class JIT {
   private:
   /// The target machine.
   std::unique_ptr<llvm::TargetMachine> targetMachine;
   /// The data layout.
   const llvm::DataLayout dataLayout;
   /// The execution session
   llvm::orc::ExecutionSession executionSession;
   /// The context
   llvm::orc::ThreadSafeContext& context;

   /// The object layer.
   llvm::orc::RTDyldObjectLinkingLayer objectLinkingLayer;
   /// The compile layer.
   llvm::orc::IRCompileLayer compileLayer;
   /// The optimize layer.
   llvm::orc::IRTransformLayer optimizeLayer;
   /// The main JITDylib
   llvm::orc::JITDylib& mainDylib;

   public:
   /// The constructor.
   explicit JIT(llvm::orc::ThreadSafeContext& ctx);

   /// Get the target machine.
   auto& getTargetMachine() { return *targetMachine; }
   /// Add a module.
   llvm::Error addModule(std::unique_ptr<llvm::Module> module);

   /// Get pointer to function.
   void* getPointerToFunction(const std::string& name);
   ~JIT() {
      if (executionSession.endSession()) {
         assert(false);
      }
   }
};

} // namespace runner

#endif // RUNNER_JIT_H
