#include <csignal>
#include <filesystem>
#include <fstream>
#include <spawn.h>

#include "dlfcn.h"
#include "unistd.h"

#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"

#include "frontend/SQL/Parser.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

#include <execution/runner.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/util/UtilTypes.h>

#include <execution/BackendPasses.h>
#include <sched.h>

#include <iostream>
namespace runner {
static void initializeContext(mlir::MLIRContext& context) {
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::tuples::TupleStreamDialect>();
   registry.insert<mlir::subop::SubOperatorDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<mlir::relalg::RelAlgDialect>();
   context.disableMultithreading();
   mlir::registerLLVMDialectTranslation(context);
}
class MLIRFrontend : public Frontend {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   void loadFromFile(std::string fileName) override {
      initializeContext(context);
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
         llvm::MemoryBuffer::getFileOrSTDIN(fileName);
      if (std::error_code ec = fileOrErr.getError()) {
         error.emit() << "Could not open input file: " << ec.message();
         return;
      }
      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
      module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
      if (!module) {
         error.emit() << "Error can't load file " << fileName << "\n";
         return;
      }
   }
   void loadFromString(std::string data) override {
      initializeContext(context);
      module = mlir::parseSourceString<mlir::ModuleOp>(data, &context);
      if (!module) {
         error.emit() << "Error can't load module\n";
      }
   }
   mlir::ModuleOp* getModule() override {
      assert(module);
      return module.operator->();
   }
};
class SQLFrontend : public Frontend {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   void loadFromString(std::string sql) override {
      initializeContext(context);

      mlir::OpBuilder builder(&context);

      mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
      if (!database) {
         error.emit() << "Database must be attached for parsing SQL";
      }
      frontend::sql::Parser translator(sql, *database, moduleOp);

      builder.setInsertionPointToStart(moduleOp.getBody());
      auto* queryBlock = new mlir::Block;
      std::vector<mlir::Type> returnTypes;
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(queryBlock);
         auto val = translator.translate(builder);
         if (val.has_value()) {
            builder.create<mlir::subop::SetResultOp>(builder.getUnknownLoc(), 0, val.value());
         }
         builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
      }
      mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
      funcOp.getBody().push_back(queryBlock);
      module = moduleOp;
   }
   void loadFromFile(std::string fileName) override {
      std::ifstream istream{fileName};
      if (!istream) {
         error.emit() << "Error can't load file " << fileName;
      }
      std::stringstream buffer;
      buffer << istream.rdbuf();
      std::string sqlQuery = buffer.str();
      loadFromString(sqlQuery);
   }
   mlir::ModuleOp* getModule() override {
      assert(module);
      return module.operator->();
   }
};

class DefaultQueryOptimizer : public QueryOptimizer {
   void optimize(mlir::ModuleOp& moduleOp) override {
      auto start = std::chrono::high_resolution_clock::now();
      mlir::PassManager pm(moduleOp.getContext());
      pm.enableVerifier(verify);
      pm.addPass(mlir::createInlinerPass());
      pm.addPass(mlir::createSymbolDCEPass());
      mlir::relalg::createQueryOptPipeline(pm, database);
      if (mlir::failed(pm.run(moduleOp))) {
         error.emit() << " Query Optimization failed";
      }
      auto end = std::chrono::high_resolution_clock::now();
      timing["QOpt"] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
   }
};
class RelAlgLoweringStep : public LoweringStep {
   void implement(mlir::ModuleOp& moduleOp) override {
      auto startLowerRelAlg = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerRelAlgPm(moduleOp->getContext());
      lowerRelAlgPm.enableVerifier(verify);
      mlir::relalg::createLowerRelAlgToSubOpPipeline(lowerRelAlgPm);
      if (mlir::failed(lowerRelAlgPm.run(moduleOp))) {
         error.emit() << "Lowering of RelAlg to Sub-Operators failed";
         return;
      }
      auto endLowerRelAlg = std::chrono::high_resolution_clock::now();
      timing["lowerRelAlg"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerRelAlg - startLowerRelAlg).count() / 1000.0;
   }
};
class SubOpLoweringStep : public LoweringStep {
   void implement(mlir::ModuleOp& moduleOp) override {
      auto startLowerSubOp = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerSubOpPm(moduleOp->getContext());
      lowerSubOpPm.enableVerifier(verify);
      mlir::subop::createLowerSubOpPipeline(lowerSubOpPm);
      if (mlir::failed(lowerSubOpPm.run(moduleOp))) {
         error.emit() << "Lowering of Sub-Operators to imperative operations failed";
         return;
      }
      auto endLowerSubOp = std::chrono::high_resolution_clock::now();
      timing["lowerSubOp"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerSubOp - startLowerSubOp).count() / 1000.0;
   }
};
class DefaultImperativeLowering : public LoweringStep {
   void implement(mlir::ModuleOp& moduleOp) override {
      auto startLowerDB = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerDBPm(moduleOp->getContext());
      lowerDBPm.enableVerifier(verify);
      mlir::db::createLowerDBPipeline(lowerDBPm);
      if (mlir::failed(lowerDBPm.run(moduleOp))) {
         error.emit() << "Lowering of imperative db operations failed";
         return;
      }
      auto endLowerDB = std::chrono::high_resolution_clock::now();
      auto startLowerDSA = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerDSAPm(moduleOp->getContext());
      lowerDSAPm.enableVerifier(verify);
      lowerDSAPm.addPass(mlir::dsa::createLowerToStdPass());
      lowerDSAPm.addPass(mlir::createCanonicalizerPass());
      lowerDSAPm.addPass(mlir::createLoopInvariantCodeMotionPass());
      lowerDSAPm.addPass(mlir::createCSEPass());
      if (mlir::failed(lowerDSAPm.run(moduleOp))) {
         error.emit() << "Lowering of dsa failed";
         return;
      }
      auto endLowerDSA = std::chrono::high_resolution_clock::now();
      timing["lowerDB"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerDB - startLowerDB).count() / 1000.0;
      timing["lowerDSA"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerDSA - startLowerDSA).count() / 1000.0;
   }
};
static bool lowerToLLVMDialect(mlir::ModuleOp& moduleOp, bool verify) {
   mlir::PassManager pm2(moduleOp->getContext());
   pm2.enableVerifier(verify);
   pm2.addPass(mlir::createConvertSCFToCFPass());
   pm2.addPass(mlir::util::createUtilToLLVMPass());
   pm2.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
   pm2.addPass(mlir::createMemRefToLLVMConversionPass());
   pm2.addPass(mlir::createArithToLLVMConversionPass());
   pm2.addPass(mlir::createConvertFuncToLLVMPass());
   pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
   pm2.addNestedPass<mlir::LLVM::LLVMFuncOp>(runner::createEnforceCABI());
   pm2.addPass(mlir::createCSEPass());
   if (mlir::failed(pm2.run(moduleOp))) {
      return false;
   }
   return true;
}
static void addLLVMExecutionContextFuncs(mlir::ModuleOp& moduleOp) {
   mlir::OpBuilder builder(moduleOp->getContext());
   builder.setInsertionPointToStart(moduleOp.getBody());
   auto pointerType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
   auto globalOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), builder.getI64Type(), false, mlir::LLVM::Linkage::Private, "execution_context", builder.getI64IntegerAttr(0));
   auto setExecContextFn = builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), "rt_set_execution_context", mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(builder.getContext()), builder.getI64Type()), mlir::LLVM::Linkage::External);
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto* block = setExecContextFn.addEntryBlock();
      auto execContext = block->getArgument(0);
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      builder.create<mlir::LLVM::StoreOp>(builder.getUnknownLoc(), execContext, ptr);
      builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
   }
   if (auto getExecContextFn = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(moduleOp.lookupSymbol("rt_get_execution_context"))) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto* block = getExecContextFn.addEntryBlock();
      builder.setInsertionPointToStart(block);
      auto ptr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), globalOp);
      auto execContext = builder.create<mlir::LLVM::LoadOp>(builder.getUnknownLoc(), ptr);
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

static void linkStatic(mlir::ExecutionEngine* engine, Error& error, runner::mainFnType& mainFunc, runner::setExecutionContextFnType& setExecutionContextFn) {
   auto currPath = std::filesystem::current_path();

   engine->dumpToObjectFile("llvm-jit-static.o");
   std::string cmd = "g++ -shared -fPIC -o llvm-jit-static.so llvm-jit-static.o";
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
   mainFunc = reinterpret_cast<runner::mainFnType>(dlsym(handle, "main"));
   dlsymError = dlerror();
   if (dlsymError) {
      dlclose(handle);
      error.emit() << "Could not load symbol for main function: " << std::string(dlsymError);
      return;
   }
   setExecutionContextFn = reinterpret_cast<runner::setExecutionContextFnType>(dlsym(handle, "rt_set_execution_context"));
   dlsymError = dlerror();
   if (dlsymError) {
      dlclose(handle);
      error.emit() << "Could not load symbol for rt_set_execution_context function: " << std::string(dlsymError);
      return;
   }
   return;
}

class DefaultCPULLVMBackend : public ExecutionBackend {
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
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

      auto maybeEngine = mlir::ExecutionEngine::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOpt::Level::Default, .enableObjectDump = false});
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
      auto mainFunc = reinterpret_cast<mainFnType>(mainFnLookupResult.get());
      auto setExecutionContextFunc = reinterpret_cast<setExecutionContextFnType>(setExecutionContextLookup.get());
      auto endJIT = std::chrono::high_resolution_clock::now();
      setExecutionContextFunc(executionContext);
      auto totalJITTime = std::chrono::duration_cast<std::chrono::microseconds>(endJIT - startJIT).count() / 1000.0;
      totalJITTime -= translateToLLVMIRTime;
      totalJITTime -= llvmPassesTime;

      std::vector<size_t> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
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
static void snapshot(mlir::ModuleOp moduleOp, Error& error, std::string fileName) {
   mlir::PassManager pm(moduleOp->getContext());
   mlir::OpPrintingFlags flags;
   flags.enableDebugInfo(false);
   pm.addPass(mlir::createLocationSnapshotPass(flags, fileName));
   if (pm.run(moduleOp).failed()) {
      error.emit() << "Snapshotting failed";
   }
}
static void addDebugInfo(mlir::ModuleOp module, std::string lastSnapShotFile) {
   auto fileAttr = mlir::LLVM::DIFileAttr::get(module->getContext(), lastSnapShotFile, std::filesystem::current_path().string());
   auto compileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(module->getContext(), llvm::dwarf::DW_LANG_C, fileAttr, mlir::StringAttr::get(module->getContext(), "LingoDB"), true, mlir::LLVM::DIEmissionKind::Full);
   module->walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
      auto subroutineType = mlir::LLVM::DISubroutineTypeAttr::get(module->getContext(), {});
      auto subProgramAttr = mlir::LLVM::DISubprogramAttr::get(compileUnitAttr, fileAttr, funcOp.getName(), funcOp.getName(), fileAttr, 0, 0, mlir::LLVM::DISubprogramFlags::Definition | mlir::LLVM::DISubprogramFlags::Optimized, subroutineType);
      funcOp->setLoc(mlir::FusedLocWith<mlir::LLVM::DIScopeAttr>::get(funcOp->getLoc(), subProgramAttr, module->getContext()));
   });
}
class CPULLVMDebugBackend : public ExecutionBackend {
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
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
      auto maybeEngine = mlir::ExecutionEngine::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOpt::Level::None, .enableObjectDump = true});
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
      mainFnType mainFunc;
      setExecutionContextFnType setExecutionContextFunc;
      linkStatic(engine.get(), error, mainFunc, setExecutionContextFunc);
      if (error) {
         return;
      }
      setExecutionContextFunc(executionContext);

      std::vector<size_t> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      timing["executionTime"] = (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]);
   }
   bool requiresSnapshotting() override {
      return true;
   }
};

class CPULLVMProfilingBackend : public ExecutionBackend {
   static inline void assignToThisCore(int coreId) {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(coreId, &mask);
      sched_setaffinity(0, sizeof(mask), &mask);
   }

   static pid_t runPerfRecord() {
      assignToThisCore(0);
      pid_t childPid = 0;
      auto parentPid = std::to_string(getpid());
      const char* argV[] = {"perf", "record", "-R", "-e", "ibs_op//p", "-c", "5000", "--intr-regs=r15", "-C", "0", nullptr};
      auto status = posix_spawn(&childPid, "/usr/bin/perf", nullptr, nullptr, const_cast<char**>(argV), environ);
      sleep(5);
      assignToThisCore(0);
      if (status != 0)
         std::cerr << "Launching application Failed: " << status << std::endl;
      return childPid;
   }
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      LLVMInitializeX86AsmParser();
      reserveLastRegister = true;
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      auto targetTriple = llvm::sys::getDefaultTargetTriple();
      std::string errorMessage;
      const auto* target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
      if (!target) {
         error.emit()<<"Could not lookup target";
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
      pm.addPass(createAnnotateProfilingDataPass());
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

      auto maybeEngine = mlir::ExecutionEngine::create(moduleOp, {.llvmModuleBuilder = convertFn, .transformer = optimizeFn, .jitCodeGenOptLevel = llvm::CodeGenOpt::Level::Default, .enableObjectDump = true});
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
      mainFnType mainFunc;
      setExecutionContextFnType setExecutionContextFunc;
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
      uint64_t r15DefaultValue = 0xbadeaffe;
      __asm__ __volatile__("mov %0, %%r15\n\t"
                           : /* no output */
                           : "a"(r15DefaultValue)
                           : "%r15");
      std::vector<size_t> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
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

RunMode getRunMode() {
   RunMode runMode;
   if (RUN_QUERIES_WITH_PERF) {
      runMode = RunMode::PERF;
   } else {
      runMode = RunMode::DEFAULT;
   }
   if (const char* mode = std::getenv("LINGODB_RUN_MODE")) {
      if (std::string(mode) == "PERF") {
         runMode = RunMode::PERF;
      } else if (std::string(mode) == "DEFAULT") {
         runMode = RunMode::DEFAULT;
      } else if (std::string(mode) == "DEBUGGING") {
         runMode = RunMode::DEBUGGING;
      } else if (std::string(mode) == "SPEED") {
         std::cout << "using speed mode" << std::endl;
         runMode = RunMode::SPEED;
      }
   }
   return runMode;
}

class DefaultQueryExecuter : public QueryExecuter {
   size_t snapShotCounter = 0;
   void handleError(std::string phase, Error& e) {
      if (e) {
         std::cerr << phase << ": " << e.getMessage() << std::endl;
         exit(1);
      }
   }
   void handleTiming(const std::unordered_map<std::string, double>& timing) {
      if (queryExecutionConfig->timingProcessor) {
         queryExecutionConfig->timingProcessor->addTiming(timing);
      }
   }
   void performSnapShot(mlir::ModuleOp moduleOp, std::string fileName = "") {
      if (queryExecutionConfig->executionBackend->requiresSnapshotting()) {
         if (fileName.empty()) {
            fileName = "snapshot-" + std::to_string(snapShotCounter++) + ".mlir";
         }
         Error e;
         snapshot(moduleOp, e, fileName);
         handleError("SNAPSHOT", e);
      }
   }

   public:
   using QueryExecuter::QueryExecuter;
   void execute() override {
      if (!executionContext) {
         std::cerr << "Execution Context is missing" << std::endl;
         exit(1);
      }
      runtime::Database* database = executionContext->getDatabase();

      if (!queryExecutionConfig->frontend) {
         std::cerr << "Frontend is missing" << std::endl;
         exit(1);
      }
      if (!queryExecutionConfig->executionBackend) {
         std::cerr << "Execution Backend is missing" << std::endl;
         exit(1);
      }
      auto& frontend = *queryExecutionConfig->frontend;

      frontend.setDatabase(database);
      if (data) {
         frontend.loadFromString(data.value());
      } else if (file) {
         frontend.loadFromFile(file.value());
      } else {
         std::cerr << "Must provide file or string!" << std::endl;
         exit(1);
      }
      handleError("FRONTEND", frontend.getError());
      mlir::ModuleOp& moduleOp = *queryExecutionConfig->frontend->getModule();
      performSnapShot(moduleOp, "input.mlir");
      if (queryExecutionConfig->queryOptimizer) {
         auto& queryOptimizer = *queryExecutionConfig->queryOptimizer;
         queryOptimizer.setDatabase(database);
         queryOptimizer.optimize(moduleOp);
         handleError("OPTIMIZER", queryOptimizer.getError());
         handleTiming(queryOptimizer.getTiming());
         performSnapShot(moduleOp);
      }
      for (auto& loweringStepPtr : queryExecutionConfig->loweringSteps) {
         auto& loweringStep = *loweringStepPtr;
         loweringStep.setDatabase(database);
         loweringStep.implement(moduleOp);
         handleError("LOWERING", loweringStep.getError());
         handleTiming(loweringStep.getTiming());
         performSnapShot(moduleOp);
      }

      auto& executionBackend = *queryExecutionConfig->executionBackend;
      executionBackend.setSnapShotCounter(snapShotCounter);
      executionBackend.execute(moduleOp, executionContext);
      handleError("BACKEND", executionBackend.getError());
      handleTiming(executionBackend.getTiming());
      if (queryExecutionConfig->resultProcessor) {
         auto& resultProcessor = *queryExecutionConfig->resultProcessor;
         resultProcessor.process(executionContext);
      }
      if (queryExecutionConfig->timingProcessor) {
         queryExecutionConfig->timingProcessor->process();
      }
   }
};
std::unique_ptr<QueryExecutionConfig> createQueryExecutionConfig(runner::RunMode runMode, bool sqlInput) {
   auto config = std::make_unique<QueryExecutionConfig>();
   if (sqlInput) {
      config->frontend = std::make_unique<SQLFrontend>();
   } else {
      config->frontend = std::make_unique<MLIRFrontend>();
   }
   config->queryOptimizer = std::make_unique<DefaultQueryOptimizer>();
   config->loweringSteps.emplace_back(std::make_unique<RelAlgLoweringStep>());
   config->loweringSteps.emplace_back(std::make_unique<SubOpLoweringStep>());
   config->loweringSteps.emplace_back(std::make_unique<DefaultImperativeLowering>());
   if (runMode == RunMode::DEBUGGING) {
      config->executionBackend = std::make_unique<CPULLVMDebugBackend>();
   } else if (runMode == RunMode::PERF) {
      config->executionBackend = std::make_unique<CPULLVMProfilingBackend>();
   } else {
      config->executionBackend = std::make_unique<DefaultCPULLVMBackend>();
   }
   config->resultProcessor = runner::createTablePrinter();
   if (runMode == RunMode::SPEED) {
      config->queryOptimizer->disableVerification();
      config->executionBackend->disableVerification();
      for (auto& loweringStep : config->loweringSteps) {
         loweringStep->disableVerification();
      }
      config->executionBackend->disableVerification();
   }
   return config;
}
std::unique_ptr<QueryExecuter> QueryExecuter::createDefaultExecuter(std::unique_ptr<QueryExecutionConfig> queryExecutionConfig) {
   return std::make_unique<DefaultQueryExecuter>(std::move(queryExecutionConfig));
}

} // namespace runner
