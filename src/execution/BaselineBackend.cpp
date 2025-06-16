#if BASELINE_ENABLED == 1
#if !defined(__linux__)
#error "Baseline backend is only supported on Linux systems."
#endif

#include "lingodb/execution/BaselineBackend.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/utility/Setting.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Transforms/Passes.h>

#include <tpde/CompilerBase.hpp>
#include <tpde/x64/CompilerX64.hpp>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <sstream>

#include <dlfcn.h>

namespace {
lingodb::utility::GlobalSetting<std::string> baselineObjectFileOut("system.compilation.baseline_object_out", "");

using namespace lingodb::compiler;
using namespace lingodb::execution;

class SpdLogSpoof {
   // storage for the log messages
   std::ostringstream oss;
   std::shared_ptr<spdlog::sinks::ostream_sink_mt> ostream_sink;
   std::shared_ptr<spdlog::logger> logger;
   std::shared_ptr<spdlog::logger> old_logger;

   public:
   SpdLogSpoof() : ostream_sink(std::make_shared<spdlog::sinks::ostream_sink_mt>(oss)), logger(std::make_shared<spdlog::logger>("string_logger", ostream_sink)) {
      old_logger = spdlog::default_logger();
      spdlog::set_default_logger(logger);
   }

   ~SpdLogSpoof() {
      spdlog::set_default_logger(old_logger);
   }

   std::string drain_logs() {
      std::string ret = oss.str();
      oss.clear();
      return ret;
   }
};

// adaptor mlir -> tpde
struct IRAdaptor {
   using IRFuncRef = mlir::func::FuncOp;
   using IRBlockRef = mlir::Block*;
   using IRInstRef = mlir::Operation*;
   using IRValueRef = mlir::Value;

   IRFuncRef INVALID_FUNC_REF = nullptr;
   IRBlockRef INVALID_BLOCK_REF = nullptr;
   IRValueRef INVALID_VALUE_REF = nullptr;

   static constexpr bool TPDE_PROVIDES_HIGHEST_VAL_IDX = false;
   static constexpr bool TPDE_LIVENESS_VISIT_ARGS = true;

   using IR = mlir::ModuleOp;

   IR* module;
   IRFuncRef cur_func = INVALID_FUNC_REF;
   Error& error;

   struct ValInfo {
      tpde::ValLocalIdx local_idx;
   };

   llvm::DenseMap<IRBlockRef, std::pair<uint32_t, uint32_t>> blockInfoMap;
   llvm::DenseMap<IRValueRef, ValInfo> valueMap;

   IRAdaptor(mlir::ModuleOp* module, Error& error) : module(module), error(error) {}

   Error& getError() { return error; }

   auto funcs() const noexcept {
      return llvm::map_range(module->getOps<mlir::func::FuncOp>(), [](mlir::func::FuncOp func) {
         return cast<IRFuncRef>(func);
      });
   }

   uint32_t func_count() const noexcept {
      const auto it = funcs();
      return std::distance(it.begin(), it.begin());
   }

   auto funcs_to_compile() const noexcept {
      return llvm::make_filter_range(funcs(), [](mlir::func::FuncOp func) {
         return !func.isExternal() && !func.isDeclaration();
      });
   }

   std::string_view func_link_name(IRFuncRef func) const noexcept {
      return func.getSymName();
   }

   bool func_extern(IRFuncRef func) const noexcept {
      return func.isExternal();
   }

   bool func_only_local(IRFuncRef func) const noexcept {
      return func.isPrivate();
   }

   static bool func_has_weak_linkage(IRFuncRef func) noexcept {
      return false; // IR does not support weak linkage
   }

   static bool cur_needs_unwind_info() noexcept {
      return false; // we do not want to support exceptions
   }

   static bool cur_is_vararg() noexcept {
      return false; // we do not support varargs
   }

   auto cur_args() const noexcept {
      mlir::FunctionOpInterface interface = dyn_cast<mlir::FunctionOpInterface>(cur_func->getParentOp());
      return std::views::all(interface.getArguments()) |
         std::views::transform([](mlir::BlockArgument arg) {
                return dyn_cast<mlir::Value>(arg);
             });
   }

   static bool cur_arg_is_byval(uint32_t) noexcept { return false; }
   static uint32_t cur_arg_byval_align(uint32_t) noexcept { return 0; }
   static uint32_t cur_arg_byval_size(uint32_t) noexcept { return 0; }
   static bool cur_arg_is_sret(uint32_t) noexcept { return false; }

   auto cur_static_allocas() const noexcept {
      mlir::FunctionOpInterface interface = mlir::dyn_cast<mlir::FunctionOpInterface>(cur_func->getParentOp());
      return interface.getFunctionBody().getOps<dialect::util::AllocaOp>(); // we do not have dynamic allocas in the IR
   }

   static bool cur_has_dynamic_alloca() noexcept {
      // the IR does not support dynamic stack allocations
      return false;
   }

   IRBlockRef cur_entry_block() noexcept {
      auto interface = mlir::cast<mlir::FunctionOpInterface>(cur_func->getParentOp());
      return &interface.getFunctionBody().getBlocks().front();
   }

   auto cur_blocks() const noexcept {
      mlir::FunctionOpInterface interface = dyn_cast<mlir::FunctionOpInterface>(cur_func->getParentOp());
      return interface.getFunctionBody().getBlocks() |
         std::views::transform([](mlir::Block& block) {
                return &block;
             });
   }

   auto block_succs(IRBlockRef block) const noexcept {
      return block->getSuccessors();
   }

   auto block_insts(IRBlockRef block) const noexcept {
      return block->getOperations() | std::views::transform([](mlir::Operation& op) {
         return &op;
      });
   }

   auto block_phis(IRBlockRef block) const noexcept {
      return block->getArguments();
   }

   uint32_t block_info(IRBlockRef block) noexcept {
      return blockInfoMap[block].first;
   }

   void block_set_info(IRBlockRef block, const uint32_t info) noexcept {
      blockInfoMap[block].first = info;
   }

   uint32_t block_info2(IRBlockRef block) noexcept {
      return blockInfoMap[block].second;
   }

   void block_set_info2(IRBlockRef block, const uint32_t info) noexcept {
      blockInfoMap[block].second = info;
   }

   std::string block_fmt_ref(IRBlockRef block) const noexcept {
      return block->getParentOp()->getName().getStringRef().str();
   }

   tpde::ValLocalIdx val_local_idx(IRValueRef val) noexcept {
      return valueMap[val].local_idx;
   }

   bool val_ignore_in_liveness_analysis(IRValueRef val) const noexcept {
      return !mlir::isa<mlir::BlockArgument>(val); // TODO: refine this
   }

   bool val_is_phi(IRValueRef val) const noexcept {
      return mlir::isa<mlir::BlockArgument>(val);
   }

   struct PHIRef {
      mlir::BlockArgument arg;

      uint32_t incoming_count() const noexcept {
         const auto preds = arg.getOwner()->getPredecessors();
         return std::distance(preds.begin(), preds.end()); // TODO: this is O(n), can we du better?
      }

      IRBlockRef incoming_block_for_slot(const uint32_t slot) const noexcept {
         assert(slot < incoming_count());
         const auto preds = arg.getOwner()->getPredecessors();
         return *std::next(preds.begin(), slot);
      }

      IRValueRef incoming_val_for_block(IRBlockRef predecessor) const noexcept {
         const auto preds = arg.getOwner()->getPredecessors();
         uint32_t slot = std::distance(preds.begin(), std::find(preds.begin(), preds.end(), predecessor));
         mlir::Operation* terminator = predecessor->getTerminator();
         const mlir::OpResult incomingVal = terminator->getOpResult(slot);
         assert(incomingVal && "Invalid slot for incoming value");
         assert(incomingVal.getType() == arg.getType() && "Incoming value type mismatch");
         return cast<IRValueRef>(incomingVal);
      }

      // looks roughly the same as the above, but does not need to calculate the slot index
      IRValueRef incoming_val_for_slot(const uint32_t slot) const noexcept {
         mlir::Block* predecessor = incoming_block_for_slot(slot);
         mlir::Operation* terminator = predecessor->getTerminator();
         const mlir::OpResult incomingVal = terminator->getOpResult(slot);
         assert(incomingVal && "Invalid slot for incoming value");
         assert(incomingVal.getType() == arg.getType() && "Incoming value type mismatch");
         return cast<IRValueRef>(incomingVal);
      }
   };

   PHIRef val_as_phi(IRValueRef val) const noexcept {
      assert(mlir::isa<mlir::BlockArgument>(val) && "Value is not a phi node");
      return PHIRef{cast<mlir::BlockArgument>(val)};
   }

   uint32_t val_alloca_size(IRValueRef val) const noexcept {
      assert(mlir::isa<dialect::util::AllocaOp>(val.getDefiningOp()) && "Value is not an alloca operation");
      auto allocaOp = cast<dialect::util::AllocaOp>(val.getDefiningOp());
      if (const auto size = allocaOp.getSize()) {
         auto op = mlir::cast_or_null<mlir::arith::ConstantIntOp>(size.getDefiningOp());
         if (!op) {
            error.emit() << "Value is not an arith int constant";
            abort();
         }
         return static_cast<uint32_t>(op.value());
      }
      return 1; // default size for an alloca without a size is 1 byte
   }

   uint32_t val_alloca_align(IRValueRef val) const noexcept {
      return 0;
   }

   std::string value_fmt_ref(IRValueRef val) const noexcept {
      return val.getDefiningOp()->getName().getStringRef().str();
   }

   auto inst_operands(IRInstRef inst) const noexcept {
      auto operands = inst->getOperands();
      if (operands.empty()) {
         return mlir::OperandRange{nullptr, 0};
      }
      return operands;
   }

   auto inst_results(IRInstRef inst) const noexcept {
      return inst->getResults();
   }

   static bool inst_fused(IRInstRef) noexcept {
      return false;
   }

   std::string inst_fmt_ref(IRInstRef inst) const noexcept {
      return inst->getName().getStringRef().str();
   }

   static void start_compile() {
      // pass
   }

   static void end_compile() {
      // pass
   }

   bool switch_func(IRFuncRef func) noexcept {
      cur_func = func;
      return true;
   }

   void reset() {
      // technically, we don't need to do anything
      cur_func = INVALID_FUNC_REF;
   }
};

// we will use the default config
struct CompilerConfig : tpde::x64::PlatformConfig {
};

// cross-platform compiler base class
template <typename Adapter, typename Derived, typename Config>
struct IRCompilerBase : tpde::CompilerBase<IRAdaptor, Derived, Config> {
   using Base = tpde::CompilerBase<IRAdaptor, Derived, Config>;
   using IR = IRAdaptor::IR;
   using ValuePartRef = typename Base::ValuePartRef;

   IRAdaptor* adaptor;
   Error error;

   IRCompilerBase(IRAdaptor* adaptor) : Base{adaptor} {
      static_assert(tpde::Compiler<Derived, Config>);
      static_assert(std::is_same_v<Adapter, IRAdaptor>, "Adapter must be IRAdaptor");
   }

   Error& getError() { return error; }

   // shortcuts to access the derived class later
   Derived* derived() noexcept { return static_cast<Derived*>(this); }
   const Derived* derived() const noexcept {
      return static_cast<Derived*>(this);
   }

   const IR* ir() const noexcept {
      return this->adaptor->module;
   }

   bool cur_func_may_emit_calls() {
      assert(!this->adaptor->cur_func.getOps().empty());
      return this->adaptor->cur_func.getOps<mlir::func::CallOp>().empty() && this->adaptor->cur_func.getOps<mlir::func::CallIndirectOp>().empty();
   }

   static typename CompilerConfig::Assembler::SymRef cur_personality_func() {
      // we do not support exceptions, so we do not need a personality function
      return {};
   }

   bool try_force_fixed_assignment(IRAdaptor::IRValueRef) const noexcept {
      return false;
   }

   struct ValueParts {
      mlir::Type valType;

      uint32_t count() const noexcept {
         valType.dump();
         return 0;
      }
      uint32_t size_bytes(uint32_t) const noexcept {
         valType.dump();
         return 0;
      }
      tpde::RegBank reg_bank(uint32_t) const noexcept {
         valType.dump();
         return CompilerConfig::GP_BANK;
      }
   };

   static ValueParts val_parts(IRAdaptor::IRValueRef value) { return ValueParts{value.getType()}; }

   struct ValRefSpecial {
      uint8_t mode = 4;
      uint64_t value;
   };

   static std::optional<ValRefSpecial> val_ref_special(IRAdaptor::IRValueRef val) {
      if (auto constOp = mlir::cast_or_null<mlir::arith::ConstantOp>(val.getDefiningOp())) {
         if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return ValRefSpecial{.value = intAttr.getInt()};
         }
         abort();
         return std::nullopt;
      }
      return std::nullopt;
   }

   ValuePartRef val_part_ref_special(ValRefSpecial& ref, uint32_t part) noexcept {
      assert(part == 0);
      return ValuePartRef(this, ref.value, 8, Config::GP_BANK);
   }

   static bool arg_is_int128(IRAdaptor::IRValueRef val) noexcept {
      return mlir::isa<dialect::util::VarLen32Type>(val.getType());
   }

   static bool arg_allow_split_reg_stack_passing(IRAdaptor::IRValueRef val) noexcept {
      return !arg_is_int128(val);
   }

   static void define_func_idx(IRAdaptor::IRFuncRef, uint32_t) {
      // pass
   }

   bool compile_inst(IRAdaptor::IRInstRef inst, typename Base::InstRange remaining) noexcept {
      return llvm::TypeSwitch<IRAdaptor::IRInstRef, bool>(inst).Default([&](IRAdaptor::IRInstRef op) {
         error.emit() << "Encountered unimplemented instruction: " << op->getName().getStringRef().str() << "\n";
         op->dump();
         return false;
      });
   }
};

// x86_64 target specific compiler
struct IRCompilerX64 : tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig> {
   using Base = CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;

   explicit IRCompilerX64(IRAdaptor* adaptor) : Base(adaptor) {
      static_assert(tpde::Compiler<IRCompilerX64, tpde::x64::PlatformConfig>);
   }

   void reset() noexcept {
      Base::reset();
   }

   Error& getError() { return Base::getError(); }
};

class BaselineBackend : public lingodb::execution::ExecutionBackend {
   // lower mlir IR to a form that can be compiled by tpde
   // currently mostly does a SCF to CF conversion
   bool lower(mlir::ModuleOp& moduleOp, std::shared_ptr<lingodb::execution::SnapshotState> serializationState) {
      mlir::PassManager pm2(moduleOp->getContext());
      pm2.enableVerifier(verify);
      lingodb::execution::addLingoDBInstrumentation(pm2, serializationState);
      pm2.addPass(mlir::createConvertSCFToCFPass());
      pm2.addPass(mlir::createCSEPass()); // TODO: evaluate whether we need this
      if (mlir::failed(pm2.run(moduleOp))) {
         return false;
      }
      return true;
   }

   void execute(mlir::ModuleOp& moduleOp, lingodb::runtime::ExecutionContext* executionContext) override {
      auto startLowering = std::chrono::high_resolution_clock::now();
      if (!lower(moduleOp, getSerializationState())) {
         error.emit() << "Could not lower module for baseline compilation";
         return;
      }
      auto endLowering = std::chrono::high_resolution_clock::now();
      timing["baselineLowering"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowering - startLowering).count() / 1000.0;

      SpdLogSpoof logSpoof;
      IRAdaptor adaptor{&moduleOp, error};

#if defined(__x86_64__)
      IRCompilerX64 compiler{&adaptor};
#else
#error "Baseline backend is only supported on x86_64 architecture."
#endif

      if (!compiler.compile()) {
         error.emit() << "Could not compile query module:\n"
                      << logSpoof.drain_logs() << "\n"
                      << adaptor.getError() << "\n"
                      << compiler.getError() << "\n";
      }
      std::vector<uint8_t> objFileBytes = compiler.assembler.build_object_file();

      std::FILE* outFile;
      std::string outFileName;
      if (baselineObjectFileOut.getValue().empty()) {
         outFile = std::tmpfile();
         // be careful, this only works on Linux!
         outFileName = std::filesystem::read_symlink(
            std::filesystem::path("/proc/self/fd") / std::to_string(fileno(outFile)));
      } else {
         outFileName = baselineObjectFileOut.getValue();
         outFile = std::fopen(baselineObjectFileOut.getValue().c_str(), "wb");
         if (!outFile) {
            error.emit() << "Could not open output file for baseline object: " << baselineObjectFileOut.getValue() << " (" << strerror(errno) << ")\n";
            return;
         }
      }
      if (std::fwrite(objFileBytes.data(), 1, objFileBytes.size(), outFile) != objFileBytes.size()) {
         error.emit() << "Could not write object file to output file: " << baselineObjectFileOut.getValue() << " (" << strerror(errno) << ")\n";
         std::fclose(outFile);
         return;
      }

      void* handle = dlopen(outFileName.c_str(), RTLD_LAZY);
      const char* dlsymError = dlerror();
      if (dlsymError) {
         error.emit() << "Can not open static library: " << std::string(dlsymError) << "\nerror:" << strerror(errno) << "\n";
         return;
      }
      auto mainFunc = reinterpret_cast<lingodb::execution::mainFnType>(dlsym(handle, "main"));
      dlsymError = dlerror();
      if (dlsymError) {
         dlclose(handle);
         error.emit() << "Could not load symbol for main function: " << std::string(dlsymError) << "\nerror:" << strerror(errno) << "\n";
         return;
      }

      std::vector<size_t> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      dlclose(handle);
      std::fclose(outFile);
   }
};
}

std::unique_ptr<lingodb::execution::ExecutionBackend> lingodb::execution::createBaselineBackend() {
   using tpde::CompilerBase;
   using tpde::x64::CompilerX64;
   return {nullptr};
}
#endif