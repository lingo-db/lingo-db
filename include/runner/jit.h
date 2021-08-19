#ifndef INCLUDE_MODERNDBS_CODEGEN_JIT_H
#define INCLUDE_MODERNDBS_CODEGEN_JIT_H

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
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
   std::unique_ptr<llvm::TargetMachine> target_machine;
   /// The data layout.
   const llvm::DataLayout data_layout;
   /// The execution session
   llvm::orc::ExecutionSession execution_session;
   /// The context
   llvm::orc::ThreadSafeContext& context;

   /// Optimization function using OptimizeFunction = std::function<std::unique_ptr<llvm::Module>(std::unique_ptr<llvm::Module>)>;

   /// The symbol resolver
   //std::shared_ptr<llvm::orc::SymbolResolver> resolver;
   /// The object layer.
   llvm::orc::RTDyldObjectLinkingLayer object_layer;
   /// The compile layer.
   llvm::orc::IRCompileLayer compile_layer;
   /// The optimize layer.
   llvm::orc::IRTransformLayer optimize_layer;
   /// The main JITDylib
   llvm::orc::JITDylib& mainDylib;

   public:
   /// The module handle.
  // using ModuleHandle = llvm::orc::VModuleKey;

   /// The constructor.
   explicit JIT(llvm::orc::ThreadSafeContext& ctx);

   /// Get the target machine.
   auto& getTargetMachine() { return *target_machine; }
   /// Add a module.
   llvm::Error addModule(std::unique_ptr<llvm::Module> module);

   /// Get pointer to function.
   void* getPointerToFunction(const std::string& name);
   ~JIT() {
      if(execution_session.endSession()){
         assert(false);
      }
   }
};

}  // namespace runner

#endif
