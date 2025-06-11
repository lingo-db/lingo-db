#if BASELINE_ENABLED == 1
#if !defined(__linux__)
#error "Baseline backend is only supported on Linux systems."
#endif

#include "lingodb/execution/BaselineBackend.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/utility/Setting.h"
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <sstream>
#include <tpde/CompilerBase.hpp>
#include <tpde/x64/CompilerX64.hpp>
#include <dlfcn.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

namespace {
lingodb::utility::GlobalSetting<std::string> baselineObjectFileOut("system.compilation.baseline_object_out", "");

using namespace lingodb::compiler;

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

class BaselineBackend : public lingodb::execution::ExecutionBackend {
   // adaptor mlir -> tpde
   struct IRAdaptor {
      using IRFuncRef = mlir::func::FuncOp;
      using IRBlockRef = mlir::Block*;
      using IRInstRef = mlir::Operation;
      using IRValueRef = mlir::Value;

      IRFuncRef INVALID_FUNC_REF = nullptr;
      IRBlockRef INVALID_BLOCK_REF = nullptr;
      IRValueRef INVALID_VALUE_REF = nullptr;

      static constexpr bool TPDE_PROVIDES_HIGHEST_VAL_IDX = false;
      static constexpr bool TPDE_LIVENESS_VISIT_ARGS = true;

      using IR = mlir::ModuleOp;

      mlir::ModuleOp* moduleOp;
      IRAdaptor(mlir::ModuleOp* moduleOp) : moduleOp(moduleOp) {}
      IR* module;
      IRFuncRef cur_func = INVALID_FUNC_REF;

      uint32_t func_count() const noexcept {
         const auto it = funcs();
         return std::distance(it.begin(), it.begin());
      }

      auto funcs() const noexcept {
         return module->getOps<mlir::func::FuncOp>();
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
         mlir::FunctionOpInterface interface = dyn_cast<mlir::FunctionOpInterface>(cur_func);
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
         mlir::FunctionOpInterface interface = dyn_cast<mlir::FunctionOpInterface>(cur_func);
         return llvm::make_filter_range(interface.getFunctionBody().getOps<dialect::util::AllocaOp>(),
                                        [](dialect::util::AllocaOp alloca) {
                                           return dyn_cast<mlir::Value>(alloca.getResult());
                                        });
      }

      static bool cur_has_dynamic_alloca() noexcept {
         // the IR does not support dynamic stack allocations
         return false;
      }

      IRBlockRef cur_entry_block() const noexcept {
         mlir::FunctionOpInterface interface = dyn_cast<mlir::FunctionOpInterface>(cur_func);
         return &interface.getFunctionBody().getBlocks().front();
      }

      auto cur_blocks() const noexcept {
         mlir::FunctionOpInterface interface = dyn_cast<mlir::FunctionOpInterface>(cur_func);
         return interface.getFunctionBody().getBlocks() |
            std::views::transform([](mlir::Block& block) {
                   return &block;
                });
      }

      auto block_succs(IRBlockRef block) const noexcept {
         return block->getSuccessors();
      }

      auto& block_insts(IRBlockRef block) const noexcept {
         return block->getOperations();
      }

      auto& block_phis(IRBlockRef block) const noexcept {
         // return block->phiNodes;
      }

      uint32_t block_info(IRBlockRef block) const noexcept {
         // return block->block_info;
      }

      void block_set_info(IRBlockRef block, uint32_t info) noexcept {
         // block->block_info = info;
      }

      uint32_t block_info2(IRBlockRef block) const noexcept {
         // return block->block_info2;
      }

      void block_set_info2(IRBlockRef block, uint32_t info) noexcept {
         // block->block_info2 = info;
      }

      std::string_view block_fmt_ref(IRBlockRef block) const noexcept {
         // return block->name;
      }

      tpde::ValLocalIdx val_local_idx(IRValueRef val) const noexcept {
         // return static_cast<tpde::ValLocalIdx>(val->id);
      }

      bool val_ignore_in_liveness_analysis(IRValueRef val) const noexcept {
         // return false;
      }

      bool val_is_phi(IRValueRef val) const noexcept {
         // return val->op.type == Operation::Type::PhiNode;
      }

      struct PHIRef {
         IRInstRef* phi_op;

         uint32_t incoming_count() const noexcept {
            // return phi_op->args.phiOperands.size();
         }

         IRValueRef incoming_val_for_slot(uint32_t slot) const noexcept {
            // return phi_op->args.phiOperands[slot].second;
         }

         IRBlockRef incoming_block_for_slot(uint32_t slot) const noexcept {
            // return phi_op->args.phiOperands[slot].first;
         }

         IRValueRef incoming_val_for_block(IRBlockRef block) const noexcept {
            // for (const auto& pair : phi_op->args.phiOperands) {
            //    if (pair.first == block) {
            //       return pair.second;
            //    }
            // }
            // return INVALID_VALUE_REF;
         }
      };

      PHIRef val_as_phi(IRValueRef val) const noexcept {
         // return PHIRef{&val->op};
      }

      uint32_t val_alloca_size(IRValueRef val) const noexcept {
         // return 0;
      }

      uint32_t val_alloca_align(IRValueRef val) const noexcept {
         // return 0;
      }

      std::string_view value_fmt_ref(IRValueRef val) const noexcept {
         // return val->name;
      }

      auto inst_operands(IRInstRef inst) const noexcept {
         // if (inst->args.operands.empty() || inst->type == Operation::Type::Const || inst->type == Operation::Type::PhiNode) {
         //    return std::ranges::subrange<IRValueRef*>(nullptr, nullptr);
         // }
         // return std::ranges::subrange<IRValueRef*>(
         //    inst->args.operands.data(),
         //    inst->args.operands.data() + inst->args.operands.size());
      }

      auto inst_results(IRInstRef inst) const noexcept {
         // if (inst->type == Operation::Type::Const || inst->type == Operation::Type::PhiNode) {
         //    return std::ranges::subrange<IRValueRef*>(nullptr, nullptr);
         // }
         // return std::ranges::subrange<IRValueRef*>(&inst->result, &inst->result + 1);
      }

      static bool inst_fused(IRInstRef) noexcept {
         // return false;
      }

      std::string_view inst_fmt_ref(IRInstRef inst) const noexcept {
         // return inst->name;
      }

      static void start_compile() {
         // pass
      }

      static void end_compile() {
         // pass
      }

      bool switch_func(IRFuncRef func) noexcept {
         // cur_func = func;
         // return true;
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
   template <typename, typename Derived, typename Config>
   struct IRCompilerBase : tpde::CompilerBase<IRAdaptor, Derived, Config> {
      using Base = tpde::CompilerBase<IRAdaptor, Derived, Config>;
      using IR = IRAdaptor::IR;

      IRCompilerBase(IRAdaptor* adaptor) : Base{adaptor} {
         static_assert(tpde::Compiler<Derived, Config>);
      }

      // shortcuts to access the derived class later
      Derived* derived() noexcept { return static_cast<Derived*>(this); }
      const Derived* derived() const noexcept {
         return static_cast<Derived*>(this);
      }

      const IR* ir() const noexcept {
         return this->adaptor->module;
      }

      bool cur_func_may_emit_calls() {
         // we can grab this directly from the IR
         // return false;
      }

      static typename CompilerConfig::Assembler::SymRef cur_personality_func() {
         // as the IR has no means of handling exceptions or specifying unwind actions
         // there are no personality functions
         // return {};
      }

      bool try_force_fixed_assignment(IRAdaptor::IRValueRef) const noexcept {
         // our example IR has a flag to try to force a fixed assignment on a value.
         // you likely don't want to do this
         // return false;
      }

      struct ValueParts {
         // all values use one register
         static uint32_t count() { return 1; }
         // all values are 64-bit
         static uint32_t size_bytes(uint32_t /* part_idx */) { return 8; }
         // all values are integers and therefore use GP registers
         static tpde::RegBank reg_bank(uint32_t /* part_idx */) {
            return CompilerConfig::GP_BANK;
         }
      };

      static ValueParts val_parts(IRAdaptor::IRValueRef) { return ValueParts{}; }

      static std::optional<typename Base::ValRefSpecial> val_ref_special(IRAdaptor::IRValueRef value_ref) {
         // if (value_ref->op.type == Operation::Type::Const) {
         //    return typename Base::ValRefSpecial{
         //       .const_data = std::bit_cast<uint64_t>(value_ref->op.args.constantValue)};
         // }
         // return std::nullopt;
      }

      static void define_func_idx(IRAdaptor::IRFuncRef func, uint32_t idx) {
         // pass
      }

      bool compile_inst(IRAdaptor::IRInstRef inst, Base::InstRange remaining) noexcept {
         // assert(inst->type != Operation::Type::PhiNode);
         //
         // // simply switch over the opcode and compile
         // switch (inst->type) {
         //    using enum Operation::Type;
         //    case Add: return derived()->compile_add(inst);
         //    case Sub: return derived()->compile_sub(inst);
         //    case Unreachable: return this->compile_unreachable();
         //    case Return: return this->compile_ret(inst);
         //    case Const: return derived()->compile_const(inst);
         //    default:
         //       TPDE_LOG_ERR("encountered unimplemented instruction");
         //       return false;
         // }
      }

      bool compile_unreachable() noexcept {
         // // this will restore callee-saved registers and return
         // // and is implemented in tpde::x64::CompilerX64 or tpde::a64::CompilerA64
         // derived()->gen_func_epilog();
         //
         // // need to do this after return
         // this->release_regs_after_return();
         // return true;
      }

      bool compile_ret(IRAdaptor::IRInstRef inst) noexcept {
         // // a return simply has to move the value to be returned into the return register.
         // // since we only have single-register integer values, this is very easy to implement
         // // in the base class using the calling convention information from the derived class.
         //
         // const Value& value = *inst->result;
         // auto ret_op = inst->args.operands.front();
         //
         // // Create the RetBuilder
         // typename Base::RetBuilder rb{*derived(), *this->cur_cc_assigner()};
         // rb.add(ret_op);
         //
         // // generate the return
         // rb.ret();
         // return true;
      }

      bool compile_sub(IRAdaptor::IRInstRef inst) noexcept {
         // Value* value = inst->result;
         //
         // const auto lhs_idx = inst->args.operands[0];
         // const auto rhs_idx = inst->args.operands[1];
         //
         // ValueRef lhs_ref = this->val_ref(lhs_idx);
         // ValueRef rhs_ref = this->val_ref(rhs_idx);
         // ValueRef res_ref = this->result_ref(value);
         //
         // ScratchReg res_scratch{this};
         // if (!derived()->encode_subi64(lhs_ref.part(0), rhs_ref.part(0), res_scratch)) {
         //    return false;
         // }
         // this->set_value(res_ref.part(0), res_scratch);
         // return true;
      }

      Base::ValuePartRef val_part_ref_special(Base::ValRefSpecial& ref, uint32_t part) noexcept {
         // return typename Base::ValuePartRef(this, ref.const_data, 8, Config::GP_BANK);
      }

      static bool arg_is_int128(Base::IRValueRef) noexcept {
         // return false;
      }

      static bool arg_allow_split_reg_stack_passing(Base::IRValueRef) noexcept {
         // return false;
      }
   };

   // x86_64 target specific compiler
   struct IRCompilerX64 : tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig> {
      using Base = tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;

      explicit IRCompilerX64(IRAdaptor* adaptor)
         : Base(adaptor) {
         static_assert(tpde::Compiler<IRCompilerX64, tpde::x64::PlatformConfig>);
      }

      void reset() noexcept {
         Base::reset();
      }
   };

   void execute(mlir::ModuleOp& moduleOp, lingodb::runtime::ExecutionContext* executionContext) override {
      SpdLogSpoof logSpoof;
      IRAdaptor adaptor{&moduleOp};

#if defined(__x86_64__)
      IRCompilerX64 compiler{&adaptor};
#else
#error "Baseline backend is only supported on x86_64 architecture."
#endif

      if (!compiler.compile()) {
         error.emit() << "Could not compile query module:\n"
                      << logSpoof.drain_logs() << "\n";
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