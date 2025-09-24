#pragma once

#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/execution/Error.h"

#include "utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

#include <tpde/ValLocalIdx.hpp>

#include <format>
#include <ranges>
#include <sstream>

namespace lingodb::execution::baseline {
// NOLINTBEGIN(readability-identifier-naming)

// adaptor mlir -> tpde
struct IRAdaptor {
   using IRFuncRef = mlir::func::FuncOp;
   using IRBlockRef = mlir::Block*;
   using IRInstRef = mlir::Operation*;
   using IRValueRef = mlir::Value;

   [[maybe_unused]] static IRFuncRef INVALID_FUNC_REF;
   [[maybe_unused]] static constexpr mlir::Block* INVALID_BLOCK_REF = nullptr;
   [[maybe_unused]] static IRValueRef INVALID_VALUE_REF;

   [[maybe_unused]] static constexpr bool TPDE_PROVIDES_HIGHEST_VAL_IDX = true;
   [[maybe_unused]] static constexpr bool TPDE_LIVENESS_VISIT_ARGS = true;

   using IR = mlir::ModuleOp;

   IR* module;
   IRFuncRef cur_func = INVALID_FUNC_REF;
   Error& error;

   struct ValInfo {
      tpde::ValLocalIdx local_idx;
      bool fused;
   };

   llvm::DenseMap<IRBlockRef, std::pair<uint32_t, uint32_t>> blockInfoMap;
   llvm::DenseMap<IRValueRef, ValInfo> values;
   llvm::DenseSet<IRInstRef> fused_instr;

   IRAdaptor(mlir::ModuleOp* module, Error& error) : module(module), error(error) {
   }

   Error& getError() const noexcept { return error; }

   [[maybe_unused]] auto funcs() const noexcept {
      return llvm::map_range(module->getOps<mlir::func::FuncOp>(), [](mlir::func::FuncOp func) {
         return cast<IRFuncRef>(func);
      });
   }

   [[maybe_unused]] uint32_t func_count() const noexcept {
      const auto it = funcs();
      return std::distance(it.begin(), it.begin());
   }

   // Then, in funcs_to_compile():
   auto funcs_to_compile() const noexcept {
      return llvm_iter_range_to_tpde_iter(llvm::make_filter_range(funcs(), [](mlir::func::FuncOp func) {
         return !func.isExternal() && !func.isDeclaration();
      }));
   }

   [[maybe_unused]] std::string_view func_link_name(IRFuncRef func) const noexcept {
      return func.getSymName();
   }

   [[maybe_unused]] bool func_extern(IRFuncRef func) const noexcept {
      return func.isExternal();
   }

   [[maybe_unused]] bool func_only_local(IRFuncRef func) const noexcept {
      return func.isPrivate() && !func.isExternal();
   }

   [[maybe_unused]] static bool func_has_weak_linkage(IRFuncRef) noexcept {
      return false; // IR does not support weak linkage
   }

   [[maybe_unused]] static bool cur_needs_unwind_info() noexcept {
      return true;
   }

   [[maybe_unused]] static bool cur_is_vararg() noexcept {
      return false; // we do not support varargs
   }

   [[maybe_unused]] auto cur_args() noexcept {
      assert(cur_func && cur_func != INVALID_FUNC_REF && "No current function set");
      return cur_func.getArguments() |
         std::views::transform([](mlir::BlockArgument arg) {
                return dyn_cast<mlir::Value>(arg);
             });
   }

   [[maybe_unused]] static bool cur_arg_is_byval(uint32_t) noexcept { return false; }
   [[maybe_unused]] static uint32_t cur_arg_byval_align(uint32_t) noexcept { return 0; }
   [[maybe_unused]] static uint32_t cur_arg_byval_size(uint32_t) noexcept { return 0; }
   [[maybe_unused]] static bool cur_arg_is_sret(uint32_t) noexcept { return false; }

   [[maybe_unused]] auto cur_static_allocas() noexcept {
      // we do not have dynamic allocas in the IR
      return llvm_iter_range_to_tpde_iter(cur_func.getFunctionBody().getOps<dialect::util::AllocaOp>());
   }

   [[maybe_unused]] static bool cur_has_dynamic_alloca() noexcept {
      // the IR does not support dynamic stack allocations
      return false;
   }

   [[maybe_unused]] uint32_t cur_highest_val_idx() const noexcept {
      return values.size();
   }

   [[maybe_unused]] IRBlockRef cur_entry_block() noexcept {
      return &cur_func.getFunctionBody().getBlocks().front();
   }

   [[maybe_unused]] auto cur_blocks() noexcept {
      return cur_func.getFunctionBody().getBlocks() |
         std::views::transform([](mlir::Block& block) {
                return &block;
             });
   }

   [[maybe_unused]] auto block_succs(IRBlockRef block) const noexcept {
      return block->getSuccessors();
   }

   [[maybe_unused]] auto block_insts(IRBlockRef block) const noexcept {
      return block->getOperations() | std::views::transform([](mlir::Operation& op) {
                return &op;
             });
   }

   [[maybe_unused]] auto block_phis(IRBlockRef block) const noexcept {
      return block->getArguments();
   }

   [[maybe_unused]] uint32_t block_info(IRBlockRef block) noexcept {
      return blockInfoMap[block].first;
   }

   [[maybe_unused]] void block_set_info(IRBlockRef block, const uint32_t info) noexcept {
      blockInfoMap[block].first = info;
   }

   [[maybe_unused]] uint32_t block_info2(IRBlockRef block) noexcept {
      return blockInfoMap[block].second;
   }

   [[maybe_unused]] void block_set_info2(IRBlockRef block, const uint32_t info) noexcept {
      blockInfoMap[block].second = info;
   }

   [[maybe_unused]] std::string block_fmt_ref(IRBlockRef block) const noexcept {
      return block->getParentOp()->getName().getStringRef().str();
   }

   [[maybe_unused]] tpde::ValLocalIdx val_local_idx(IRValueRef val) noexcept {
      return values[val].local_idx;
   }

   [[maybe_unused]] bool val_ignore_in_liveness_analysis(IRValueRef) const noexcept {
      return false;
   }

   struct PHIRef {
      mlir::BlockArgument arg;

      [[maybe_unused]] uint32_t incoming_count() const noexcept {
         const auto preds = arg.getOwner()->getPredecessors();
         return std::distance(preds.begin(), preds.end()); // TODO: this is O(n), can we du better?
      }

      [[maybe_unused]] IRBlockRef incoming_block_for_slot(const uint32_t slot) const noexcept {
         assert(slot < incoming_count());
         const auto preds = arg.getOwner()->getPredecessors();
         return *std::next(preds.begin(), slot);
      }

      [[maybe_unused]] IRValueRef incoming_val_for_block(IRBlockRef predecessor) const noexcept {
#ifndef NDEBUG
         const auto preds = arg.getOwner()->getPredecessors();
         const auto matching_pred = std::find(preds.begin(), preds.end(), predecessor);
         assert(matching_pred != preds.end() && "Predecessor block not found in predecessors");
         assert(
            std::count(preds.begin(), preds.end(), predecessor) == 1 &&
            "Predecessor block found multiple times in predecessors. While this is allowed for MLIR, this is currently not supported by TPDE, especially with different edge-values");
#endif
         mlir::Operation* terminator = predecessor->getTerminator();
         const mlir::Value incomingVal = mlir::TypeSwitch<mlir::Operation*, mlir::Value>(terminator)
                                            .Case<mlir::cf::BranchOp>([&](mlir::cf::BranchOp br) {
                                               return br.getDestOperands()[arg.getArgNumber()];
                                            })
                                            .Case<mlir::cf::CondBranchOp>([&](mlir::cf::CondBranchOp br) {
                                               if (br.getTrueDest() == arg.getOwner()) {
                                                  return br.getTrueDestOperands()[arg.getArgNumber()];
                                               }
                                               if (br.getFalseDest() == arg.getOwner()) {
                                                  return br.getFalseDestOperands()[arg.getArgNumber()];
                                               }
                                               assert(0 && "Predecessor block not found in branch operands");
                                               return mlir::Value();
                                            })
                                            .Default([](mlir::Operation* op) {
                                               op->dump();
                                               assert(0);
                                               return mlir::Value();
                                            });
         assert(incomingVal && "Invalid slot for incoming value");
         assert(incomingVal.getType() == arg.getType() && "Incoming value type mismatch");
         return cast<IRValueRef>(incomingVal);
      }

      // looks roughly the same as the above, but does not need to calculate the slot index
      [[maybe_unused]] IRValueRef incoming_val_for_slot(const uint32_t slot) const noexcept {
         mlir::Block* predecessor = incoming_block_for_slot(slot);
         return incoming_val_for_block(predecessor);
      }
   };

   [[maybe_unused]] bool val_is_phi(IRValueRef val) const noexcept {
      return mlir::isa<mlir::BlockArgument>(val);
   }

   [[maybe_unused]] PHIRef val_as_phi(IRValueRef val) const noexcept {
      assert(mlir::isa<mlir::BlockArgument>(val) && "Value is not a phi node");
      return PHIRef{cast<mlir::BlockArgument>(val)};
   }

   [[maybe_unused]] uint32_t val_alloca_size(IRValueRef val) {
      assert(mlir::isa<dialect::util::AllocaOp>(val.getDefiningOp()) && "Value is not an alloca operation");
      auto allocaOp = cast<dialect::util::AllocaOp>(val.getDefiningOp());
      unsigned count = 1; // default size for an alloca without a size is 1
      if (const auto size = allocaOp.getSize()) {
         if (auto const_int = mlir::dyn_cast<mlir::arith::ConstantIntOp>(size.getDefiningOp())) {
            count = const_int.value();
         } else if (auto const_index = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(size.getDefiningOp())) {
            count = const_index.value();
         } else {
            getError().emit() << "Value is not an arith int constant\n";
         }
      }

      const mlir::TypedValue<dialect::util::RefType> res_ptr = allocaOp.getRef();
      const mlir::Type element_type = res_ptr.getType().getElementType();
      const auto size = get_size(element_type);
      if (!size) {
         getError().emit() << std::format("Value is not a supported type for alloca size calculation: {}\n",
                                          element_type.getAbstractType().getName().str());
         return 1;
      }
      return *size * count;
   }

   [[maybe_unused]] uint32_t val_alloca_align(IRValueRef val) const noexcept {
      return 1;
   }

   [[maybe_unused]] std::string value_fmt_ref(IRValueRef val) const noexcept {
      if (auto* const op = val.getDefiningOp()) {
         return op->getName().getStringRef().str();
      }
      if (const mlir::BlockArgument arg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
         return std::format("block_arg_{}", arg.getArgNumber());
      }
      return "";
   }

   [[maybe_unused]] auto inst_operands(IRInstRef op) const noexcept {
      return llvm::TypeSwitch<IRInstRef, mlir::OperandRange>(op)
         .Case<mlir::cf::BranchOp>([](auto) {
            // direct branches only have block arguments as operands, their usage is counted via phi nodes
            return mlir::OperandRange{nullptr, 0};
         })
         .Case<mlir::cf::CondBranchOp>([](const auto branch) {
            return mlir::OperandRange{&branch->getOpOperand(0), 1};
         })
         .Default([](const auto op) {
            auto operands = op->getOperands();
            if (operands.empty()) {
               return mlir::OperandRange{nullptr, 0};
            }
            return operands;
         });
   }

   [[maybe_unused]] static auto inst_results(const IRInstRef inst) noexcept {
      return inst->getResults();
   }

   [[maybe_unused]] bool inst_fused(const IRInstRef inst) noexcept {
      return fused_instr.contains(inst);
   }

   [[maybe_unused]] void inst_set_fused(const IRInstRef inst, const bool fused) noexcept {
      fused_instr.insert(inst);
   }

   [[maybe_unused]] std::string inst_fmt_ref(IRInstRef inst) const noexcept {
      return inst->getName().getStringRef().str();
   }

   [[maybe_unused]] static void start_compile() {
      // pass
   }

   [[maybe_unused]] static void end_compile() {
      // pass
   }

   [[maybe_unused]] bool switch_func(IRFuncRef func) noexcept {
      cur_func = func;
      values.clear();
      fused_instr.clear();
      for (auto& block : func.getFunctionBody()) {
         for (auto arg : block.getArguments())
            values[arg] = ValInfo{
               .local_idx = static_cast<tpde::ValLocalIdx>(values.size()), .fused = false};
         for (auto& op : block) {
            for (auto result : op.getResults())
               values[result] = ValInfo{
                  .local_idx = static_cast<tpde::ValLocalIdx>(values.size()), .fused = false};
         }
      }
      return true;
   }

   [[maybe_unused]] void reset() {
      cur_func = INVALID_FUNC_REF;
      values.clear();
      fused_instr.clear();
   }
};
// NOLINTEND(readability-identifier-naming)
}
