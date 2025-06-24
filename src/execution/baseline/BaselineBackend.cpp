#if BASELINE_ENABLED == 1
#if !defined(__linux__)
#error "Baseline backend is only supported on Linux systems."
#endif

#include "lingodb/execution/BaselineBackend.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/utility/Setting.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Dialect/Transform/IR/TransformTypes.h.inc>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Operation.h>
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

#include "snippet_encoders_x64.hpp"

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

class TupleHelper {
   mlir::TupleType tupleType;

   public:
   explicit TupleHelper(mlir::TupleType tupleType) : tupleType(tupleType) {}

   [[nodiscard]] unsigned getElementOffset(uint32_t index) const noexcept {
      size_t offset = 0;
      assert(index < tupleType.getTypes().size() && "Index out of bounds for tuple type");
      for (const mlir::Type elem : tupleType.getTypes() | std::views::take(index - 1)) {
         unsigned elemAlign = 0;
         size_t elemSize = 0;
         llvm::TypeSwitch<mlir::Type>(elem)
            .Case<dialect::util::RefType>([&](dialect::util::RefType t) {
               elemSize = sizeof(void*);
               elemAlign = alignof(void*);
            })
            .Case<mlir::IntegerType>([&](mlir::IntegerType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (elemSize) {
                  case 1: elemAlign = alignof(int8_t); break;
                  case 2: elemAlign = alignof(int16_t); break;
                  case 4: elemAlign = alignof(int32_t); break;
                  case 8: elemAlign = alignof(int64_t); break;
                  case 16: elemAlign = alignof(__int128); break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::FloatType>([&](mlir::FloatType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (t.getIntOrFloatBitWidth()) {
                  case 32:
                     static_assert(sizeof(float) == 4);
                     elemAlign = alignof(float);
                     break;
                  case 64:
                     static_assert(sizeof(double) == 8);
                     elemAlign = alignof(double);
                     break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::TupleType>([&](mlir::TupleType tupleType) {
               auto [size, align] = TupleHelper{tupleType}.sizeAndPadding();
               elemSize = size;
               elemAlign = align;
            })
            .Case<mlir::IndexType>([&](auto) {
               elemSize = sizeof(uint64_t);
               elemAlign = alignof(uint64_t);
            })
            .Default([](mlir::Type) {
               assert(false && "Cannot calculate size for unsupported type.");
               return 0;
            });
         // enforce alignment requirement
         offset += offset % elemAlign;
         // add offset of element
         offset += elemSize;
      }
      return offset;
   }

   [[nodiscard]] std::pair<size_t, unsigned> sizeAndPadding() const noexcept {
      size_t offset = 0;
      unsigned maxAlign = 0;
      for (const mlir::Type elem : tupleType.getTypes()) {
         unsigned elemAlign = 0;
         size_t elemSize = 0;
         llvm::TypeSwitch<mlir::Type>(elem)
            .Case<dialect::util::RefType>([&](dialect::util::RefType t) {
               elemSize = sizeof(void*);
               elemAlign = alignof(void*);
            })
            .Case<mlir::IntegerType>([&](mlir::IntegerType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (elemSize) {
                  case 1: elemAlign = alignof(int8_t); break;
                  case 2: elemAlign = alignof(int16_t); break;
                  case 4: elemAlign = alignof(int32_t); break;
                  case 8: elemAlign = alignof(int64_t); break;
                  case 16: elemAlign = alignof(__int128); break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::FloatType>([&](mlir::FloatType t) {
               elemSize = t.getIntOrFloatBitWidth() / 8;
               switch (t.getIntOrFloatBitWidth()) {
                  case 32:
                     static_assert(sizeof(float) == 4);
                     elemAlign = alignof(float);
                     break;
                  case 64:
                     static_assert(sizeof(double) == 8);
                     elemAlign = alignof(double);
                     break;
                  default:
                     assert(false && "Unsupported integer type width for alignment calculation.");
                     elemAlign = 1; // fallback to 1 byte alignment
               }
            })
            .Case<mlir::TupleType>([&](mlir::TupleType tupleType) {
               auto [size, align] = TupleHelper{tupleType}.sizeAndPadding();
               elemSize = size;
               elemAlign = align;
            })
            .Case<mlir::IndexType>([&](auto) {
               elemSize = sizeof(uint64_t);
               elemAlign = alignof(uint64_t);
            })
            .Default([](mlir::Type) {
               assert(false && "Cannot calculate size for unsupported type.");
               return 0;
            });
         // enforce alignment requirement
         offset += offset % elemAlign;
         // add offset of element
         offset += elemSize;
         maxAlign = std::max(maxAlign, elemAlign);
      }
      // adjust entire struct size to the maximum alignment (for array usage)
      offset += offset % maxAlign;
      return {offset, maxAlign};
   }
};

// adaptor mlir -> tpde
// NOLINTBEGIN(readability-identifier-naming)
struct IRAdaptor {
   using IRFuncRef = mlir::func::FuncOp;
   using IRBlockRef = mlir::Block*;
   using IRInstRef = mlir::Operation*;
   using IRValueRef = mlir::Value;

   [[maybe_unused]] static IRFuncRef INVALID_FUNC_REF;
   [[maybe_unused]] static IRBlockRef INVALID_BLOCK_REF;
   [[maybe_unused]] static IRValueRef INVALID_VALUE_REF;

   [[maybe_unused]] static constexpr bool TPDE_PROVIDES_HIGHEST_VAL_IDX = true;
   [[maybe_unused]] static constexpr bool TPDE_LIVENESS_VISIT_ARGS = true;

   using IR = mlir::ModuleOp;

   IR* module;
   IRFuncRef cur_func = INVALID_FUNC_REF;
   Error& error;

   struct ValInfo {
      tpde::ValLocalIdx local_idx;
   };

   llvm::DenseMap<IRBlockRef, std::pair<uint32_t, uint32_t>> blockInfoMap;
   llvm::DenseMap<IRValueRef, ValInfo> values;

   IRAdaptor(mlir::ModuleOp* module, Error& error) : module(module), error(error) {}

   Error& getError() { return error; }

   [[maybe_unused]] auto funcs() const noexcept {
      return llvm::map_range(module->getOps<mlir::func::FuncOp>(), [](mlir::func::FuncOp func) {
         return cast<IRFuncRef>(func);
      });
   }

   [[maybe_unused]] uint32_t func_count() const noexcept {
      const auto it = funcs();
      return std::distance(it.begin(), it.begin());
   }

   [[maybe_unused]] auto funcs_to_compile() const noexcept {
      return llvm::make_filter_range(funcs(), [](mlir::func::FuncOp func) {
         return !func.isExternal() && !func.isDeclaration();
      });
   }

   [[maybe_unused]] std::string_view func_link_name(IRFuncRef func) const noexcept {
      return func.getSymName();
   }

   [[maybe_unused]] bool func_extern(IRFuncRef func) const noexcept {
      return func.isExternal();
   }

   [[maybe_unused]] bool func_only_local(IRFuncRef func) const noexcept {
      return func.isPrivate();
   }

   [[maybe_unused]] static bool func_has_weak_linkage(IRFuncRef func) noexcept {
      return false; // IR does not support weak linkage
   }

   [[maybe_unused]] static bool cur_needs_unwind_info() noexcept {
      return false; // we do not want to support exceptions
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
      return cur_func.getFunctionBody().getOps<dialect::util::AllocaOp>(); // we do not have dynamic allocas in the IR
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

   [[maybe_unused]] bool val_ignore_in_liveness_analysis(IRValueRef val) const noexcept {
      // return !mlir::isa<mlir::BlockArgument>(val); // TODO: refine this
      return false;
   }

   [[maybe_unused]] bool val_is_phi(IRValueRef val) const noexcept {
      return mlir::isa<mlir::BlockArgument>(val);
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
         const auto preds = arg.getOwner()->getPredecessors();
         const auto matching_pred = std::find(preds.begin(), preds.end(), predecessor);
         assert(matching_pred != preds.end() && "Predecessor block not found in predecessors");
#ifndef NDEBUG
         assert(std::count(preds.begin(), preds.end(), predecessor) == 1 && "Predecessor block found multiple times in predecessors. While this is allowed for MLIR, this is currently not supported by TPDE, especially with different edge-values");
#endif
         const uint32_t slot = std::distance(preds.begin(), matching_pred);
         mlir::Operation* terminator = predecessor->getTerminator();
         const mlir::OpResult incomingVal = mlir::TypeSwitch<mlir::Operation, mlir::OpResult>(*terminator)
                                               .Case([&](mlir::cf::BranchOp br) {
                                                  return br.getDestOperands()[slot];
                                               })
                                               .Case([&](mlir::cf::CondBranchOp br) {
                                                  if (br.getTrueDest() == arg.getOwner()) {
                                                     return br.getTrueDestOperands()[slot];
                                                  }
                                                  if (br.getFalseDest() == arg.getOwner()) {
                                                     return br.getFalseDestOperands()[slot];
                                                  }
                                                  assert(0 && "Predecessor block not found in branch operands");
                                                  return mlir::OpResult();
                                               })
                                               .Default([](auto op) {
                                                  op.dump();
                                                  assert(0);
                                                  return mlir::OpResult();
                                               });
         assert(incomingVal && "Invalid slot for incoming value");
         assert(incomingVal.getType() == arg.getType() && "Incoming value type mismatch");
         return cast<IRValueRef>(incomingVal);
      }

      // looks roughly the same as the above, but does not need to calculate the slot index
      [[maybe_unused]] IRValueRef incoming_val_for_slot(const uint32_t slot) const noexcept {
         mlir::Block* predecessor = incoming_block_for_slot(slot);
         mlir::Operation* terminator = predecessor->getTerminator();
         assert(terminator && "Predecessor block does not have a terminator");
         assert(arg.getArgNumber() < terminator->getNumOperands() && "block arg num out of range for operands of predecessor terminator");
         const mlir::Value incomingVal = terminator->getOperand(arg.getArgNumber());
         assert(incomingVal.getType() == arg.getType() && "Incoming value type mismatch");
         return incomingVal;
      }
   };

   [[maybe_unused]] PHIRef val_as_phi(IRValueRef val) const noexcept {
      assert(mlir::isa<mlir::BlockArgument>(val) && "Value is not a phi node");
      return PHIRef{cast<mlir::BlockArgument>(val)};
   }

   [[maybe_unused]] uint32_t val_alloca_size(IRValueRef val) const noexcept {
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

   [[maybe_unused]] uint32_t val_alloca_align(IRValueRef val) const noexcept {
      return 0;
   }

   [[maybe_unused]] std::string value_fmt_ref(IRValueRef val) const noexcept {
      return val.getDefiningOp()->getName().getStringRef().str();
   }

   [[maybe_unused]] auto inst_operands(IRInstRef inst) const noexcept {
      auto operands = inst->getOperands();
      if (operands.empty()) {
         return mlir::OperandRange{nullptr, 0};
      }
      return operands;
   }

   [[maybe_unused]] auto inst_results(IRInstRef inst) const noexcept {
      return inst->getResults();
   }

   [[maybe_unused]] static bool inst_fused(IRInstRef) noexcept {
      return false;
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
      for (auto& block : func.getFunctionBody()) {
         for (auto arg : block.getArguments())
            values[arg] = ValInfo{.local_idx = tpde::ValLocalIdx(values.size())};
         for (auto& op : block) {
            for (auto result : op.getResults())
               values[result] = ValInfo{.local_idx = tpde::ValLocalIdx(values.size())};
         }
      }
      return true;
   }

   [[maybe_unused]] void reset() {
      cur_func = INVALID_FUNC_REF;
      values.clear();
   }
};
// NOLINTEND(readability-identifier-naming)
IRAdaptor::IRFuncRef IRAdaptor::INVALID_FUNC_REF = nullptr;
IRAdaptor::IRBlockRef IRAdaptor::INVALID_BLOCK_REF = nullptr;
IRAdaptor::IRValueRef IRAdaptor::INVALID_VALUE_REF = mlir::Value();

// we will use the default config
// NOLINTBEGIN(readability-identifier-naming)
struct CompilerConfig : tpde::x64::PlatformConfig {
};
// NOLINTEND(readability-identifier-naming)

// cross-platform compiler base class
// NOLINTBEGIN(readability-identifier-naming)
template <typename Adapter, typename Derived, typename Config>
struct IRCompilerBase : tpde::CompilerBase<IRAdaptor, Derived, Config> {
   using Base = tpde::CompilerBase<IRAdaptor, Derived, Config>;
   using IR = IRAdaptor::IR;
   using ValuePartRef = typename Base::ValuePartRef;
   using GenericValuePart = typename Base::GenericValuePart;
   using Expr = typename Base::GenericValuePart::Expr;
   using ScratchReg = typename Base::ScratchReg;
   using ValueRef = typename Base::ValueRef;
   using InstRange = typename Base::InstRange;
   using IRInstRef = IRAdaptor::IRInstRef;
   using AsmReg = Base::AsmReg;

   Error error;
   mlir::DenseMap<mlir::StringRef, uint32_t> funcMap;

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
      return true;
   }

   static CompilerConfig::Assembler::SymRef cur_personality_func() {
      // we do not support exceptions, so we do not need a personality function
      return {};
   }

   bool try_force_fixed_assignment(IRAdaptor::IRValueRef) const noexcept {
      return false;
   }

   struct ValueParts {
      mlir::Type valType;

      uint32_t count() const noexcept {
         assert(!mlir::isa<mlir::TupleType>(valType) && "Tuple types are not supported yet");
         return mlir::TypeSwitch<mlir::Type, uint32_t>(valType)
            .Case<mlir::IntegerType>([](auto intType) { return (intType.getIntOrFloatBitWidth() + 64 - 1) / 64; })
            .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) { return 2; })
            .Default([](mlir::Type t) { return 1; });
      }
      uint32_t size_bytes(uint32_t part_idx) const noexcept {
         assert(!mlir::isa<mlir::TupleType>(valType) && "Tuple types are not supported yet");
         return mlir::TypeSwitch<mlir::Type, uint32_t>(valType)
            .Case<mlir::IntegerType>([&](auto intType) {
               assert(part_idx < (intType.getIntOrFloatBitWidth() + 64 - 1) / 64 && "Part index out of range for integer type");
               // integer types are sized by their bit width
               return (intType.getIntOrFloatBitWidth() % 65 + 8 - 1) / 8;
            })
            .template Case<mlir::IndexType, dialect::util::RefType>([](auto) { return 8; })
            .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
               assert(part_idx < 2 && "VarLen32 and Buffer types are made of two parts");
               // these container types are 2x64-bit
               return 8;
            })
            .Default([](mlir::Type t) {
               t.dump();
               assert(0 && "invalid type");
               return 0;
            }); // all other types are not supported yet
      }
      tpde::RegBank reg_bank(uint32_t) const noexcept {
         assert(!mlir::isa<mlir::TupleType>(valType) && "Tuple types are not supported yet");
         // we do not support floats
         return CompilerConfig::GP_BANK;
      }
   };

   static ValueParts val_parts(IRAdaptor::IRValueRef value) { return ValueParts{value.getType()}; }

   struct ValRefSpecial {
      uint8_t mode = 4;
      uint64_t value;
   };

   static std::optional<ValRefSpecial> val_ref_special(IRAdaptor::IRValueRef val) {
      if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(val.getDefiningOp())) {
         if (const auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constOp.getValue())) {
            return ValRefSpecial{.mode = 4, .value = std::bit_cast<uint64_t>(intAttr.getInt())};
         }
         assert(0 && "Unsupported constant type in val_ref_special");
         return std::nullopt;
      }
      if (const auto utilSizeOfOp = mlir::dyn_cast_or_null<dialect::util::SizeOfOp>(val.getDefiningOp())) {
         const mlir::TypeAttr type_attr = mlir::cast<mlir::TypeAttr>(utilSizeOfOp->getAttr("type"));
         const size_t tuple_size = TupleHelper{mlir::cast<mlir::TupleType>(type_attr.getValue())}.sizeAndPadding().first;
         return ValRefSpecial{.mode = 4, .value = static_cast<uint64_t>(tuple_size)};
      }
      return std::nullopt;
   }

   ValuePartRef val_part_ref_special(ValRefSpecial& ref, uint32_t part) noexcept {
      assert(part == 0);
      return ValuePartRef(this, ref.value, 8, Config::GP_BANK);
   }

   static bool arg_is_int128(IRAdaptor::IRValueRef val) noexcept {
      // TODO: implement this properly
      return mlir::isa<dialect::util::VarLen32Type>(val.getType()) || (mlir::isa<mlir::IntegerType>(val.getType()) && val.getType().getIntOrFloatBitWidth() == 128) || mlir::isa<dialect::util::BufferType>(val.getType());
   }

   static bool arg_allow_split_reg_stack_passing(IRAdaptor::IRValueRef val) noexcept {
      return false; // we do not support split register stack passing
   }

   void define_func_idx(IRAdaptor::IRFuncRef func, uint32_t idx) {
      funcMap[func.getSymName()] = idx;
   }

   bool compile_arith_binary_op(IRInstRef op) {
      auto lhs_vr = this->val_ref(op->getOperand(0));
      auto rhs_vr = this->val_ref(op->getOperand(1));
      auto res_type = op->getResult(0).getType();

      unsigned op_width;
      switch (res_type.getIntOrFloatBitWidth()) {
         case 32:
         case 64:
            op_width = res_type.getIntOrFloatBitWidth();
            break;
         case 128:
            op_width = 128;
            break;
         case 1:
            op_width = 32; // we use 32-bit operations for boolean values
            break;
         default:
            op->dump();
            assert(0 && "Unsupported integer type width for arithmetic operation");
            return false;
      }

      if (op_width == 128) {
         std::unordered_map<std::string, bool (Derived::*)(GenericValuePart&&, GenericValuePart&&, GenericValuePart&&, GenericValuePart&&, ScratchReg&, ScratchReg&)> encoder_lookup = {
            {"arith.addi", &Derived::encode_arith_add_i128},
            {"arith.andi", &Derived::encode_arith_land_i128},
         };
#ifndef NDEBUG
         if (!encoder_lookup.contains(op->getName().getStringRef().str().c_str())) {
            op->dump();
            std::cerr << op->getName().getStringRef().str().c_str() << " is not supported by the baseline backend\n";
            assert(0);
         }
#endif
         const auto encoder = encoder_lookup[op->getName().getStringRef().str().c_str()];

         ScratchReg res_scratch_high{derived()};
         ScratchReg res_scratch_low{derived()};
         auto res = this->result_ref(op->getResult(0));
         auto res_low = res.part(0);
         auto res_high = res.part(1);
         (derived()->*encoder)(std::move(lhs_vr.part(0)), std::move(lhs_vr.part(1)), std::move(rhs_vr.part(0)), std::move(rhs_vr.part(1)), res_scratch_low, res_scratch_high);
         this->set_value(res_low, res_scratch_low);
         this->set_value(res_high, res_scratch_high);
         return true;
      } else {
         auto lhs_pr = lhs_vr.part(0);
         auto rhs_pr = rhs_vr.part(0);
         // move constant operands to the right side
         if ((mlir::isa<mlir::arith::AddIOp, mlir::arith::MulIOp, mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp>(op)) && lhs_pr.is_const() && !rhs_pr.is_const()) {
            std::swap(lhs_vr, rhs_vr);
            std::swap(lhs_pr, rhs_pr);
         }
         auto lhs_op = GenericValuePart{std::move(lhs_pr)};
         auto rhs_op = GenericValuePart{std::move(rhs_pr)};

         ScratchReg res_scratch{derived()};

         // encode functions for 32/64 bit operations
         // TODO: replace this map with something more efficient
         std::unordered_map<std::string, std::array<bool (Derived::*)(GenericValuePart&&, GenericValuePart&&, ScratchReg&), 2>> encoder_lookup = {
            {"arith.addi", {&Derived::encode_arith_add_i32, &Derived::encode_arith_add_i64}},
            {"arith.andi", {&Derived::encode_arith_land_i32, &Derived::encode_arith_land_i64}},
         };
#ifndef NDEBUG
         if (!encoder_lookup.contains(op->getName().getStringRef().str().c_str())) {
            op->dump();
            std::cerr << op->getName().getStringRef().str().c_str() << " is not supported by the baseline backend\n";
            assert(0);
         }
#endif
         const auto encoders = encoder_lookup[op->getName().getStringRef().str().c_str()];
         const auto sub_encoder_idx = op_width == 64 ? 1 : 0;
         (derived()->*encoders[sub_encoder_idx])(std::move(lhs_op), std::move(rhs_op), res_scratch);
         auto [res_vr, res_pr] = this->result_ref_single(op->getResult(0));
         this->set_value(res_pr, res_scratch);
         return true;
      }
   }

   bool compile_cf_br_op(mlir::cf::BranchOp op) {
      auto spilled = this->spill_before_branch();
      this->begin_branch_region();

      derived()->generate_branch_to_block(Derived::Jump::jmp, op.getDest(), false, true);

      this->end_branch_region();
      derived()->release_spilled_regs(spilled);
      return true;
   }

   bool compile_util_sizeof_op(dialect::util::SizeOfOp op) {
      // the value of the sizeof operation can be statically calculated => treat value as constant, no codegen required
      assert(this->val_ref(op.getResult()).part(0).is_const());
      return true;
   }

   bool compile_arith_const_op(mlir::arith::ConstantOp op) {
      // ValueRef val = this->val_ref(op.getResult());
      // ValuePartRef val_pr = val.part(0);
      // assert(val_pr.is_const());
      //
      // auto [_, res_ref] = this->result_ref_single(op.getResult());
      // ScratchReg res_scratch{this};
      // AsmReg res_reg = res_scratch.alloc_gp();
      // ASM(MOV64ri, res_reg, val_pr.const_data()[0]);
      assert(this->val_ref(op.getResult()).part(0).is_const());
      return true;
   }

   bool compile_util_generic_memref_cast_op(dialect::util::GenericMemrefCastOp op) {
      const mlir::TypedValue<dialect::util::RefType> src = op.getVal();
      const mlir::TypedValue<dialect::util::RefType> dst = op.getRes();
      assert(val_parts(src).count() == 1);
      assert(val_parts(dst).count() == 1);

      auto [_, src_ref] = this->val_ref_single(src);
      ValueRef res_ref = this->result_ref(op);

      if (val_parts(src).reg_bank(0) == val_parts(dst).reg_bank(0)) {
         res_ref.part(0).set_value(std::move(src_ref));
         return true;
      }
      return false;
   }

   bool compile_util_tuple_element_ptr_op(dialect::util::TupleElementPtrOp op) {
      const mlir::TypedValue<dialect::util::RefType> base_ref = op.getRef();
      const mlir::TupleType tuple_type = mlir::cast<mlir::TupleType>(base_ref.getType().getElementType());

      // calc the byte-offset of the element in the tuple (same address layout as C++ structs for target)
      unsigned elementOffset = TupleHelper{tuple_type}.getElementOffset(op.getIdx());

      const auto dst = op->getResult(0);
      assert(val_parts(base_ref).count() == 1);
      assert(val_parts(dst).count() == 1);
      auto [_, base_vr] = this->val_ref_single(base_ref);
      auto [_, res_vr] = this->result_ref_single(dst);

      // create a base + offset expression
      AsmReg base_reg = base_vr.load_to_reg();
      GenericValuePart addr = typename GenericValuePart::Expr{base_reg, elementOffset};

      // load value to register (e.g. mov + add / lea for x86_64)
      AsmReg res_reg = derived()->gval_expr_as_reg(addr);
      ScratchReg res_scratch{derived()};
      derived()->mov(res_scratch.alloc_gp(), res_reg, 8);
      this->set_value(res_vr, res_scratch);
      return true;
   }

   bool compile_util_load_op(dialect::util::LoadOp op) {
      const mlir::TypedValue<dialect::util::RefType> ptr = op.getRef();
      const mlir::TypedValue<mlir::IndexType> idx = op.getIdx();
      const mlir::Type loaded_type = op.getVal().getType();

      auto [_, ptr_ref] = this->val_ref_single(ptr);
      GenericValuePart ptr_part;
      if (idx) {
         if (auto idx_op = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(idx.getDefiningOp())) {
            AsmReg ptr_reg = ptr_ref.alloc_reg();
            ptr_part = GenericValuePart{Expr{std::move(ptr_reg), idx_op.value()}};
         } else {
            error.emit() << "Index must be a constant index operation";
            return false;
         }
      } else {
         ptr_part = GenericValuePart{std::move(ptr_ref)};
      }

      auto res = this->result_ref(op.getVal());
      ScratchReg res_scratch{this};
      return mlir::TypeSwitch<mlir::Type, bool>(loaded_type)
         .Case([&](const mlir::IntegerType t) {
            switch (t.getIntOrFloatBitWidth()) {
               case 64: derived()->encode_util_load_i64(std::move(ptr_part), res_scratch); break;
               case 32: derived()->encode_util_load_i32(std::move(ptr_part), res_scratch); break;
               case 128: {
                  ScratchReg res_scratch_high{derived()};
                  auto res_low = res.part(0);
                  auto res_high = res.part(1);
                  derived()->encode_loadi128(std::move(ptr_part), res_scratch, res_scratch_high);
                  this->set_value(res_low, res_scratch);
                  this->set_value(res_high, res_scratch_high);
                  return true;
               }
               default:
                  assert(false && "Unsupported integer type width for load operation");
                  return false;
            }
            ValuePartRef res_ref = res.part(0);
            this->set_value(res_ref, res_scratch);
            return true;
         })
         .template Case<dialect::util::RefType, mlir::IndexType>([&](auto) {
            derived()->encode_util_load_i64(std::move(ptr_part), res_scratch);
            ValuePartRef res_ref = res.part(0);
            this->set_value(res_ref, res_scratch);
            return true;
         })
         .Default([](mlir::Type t) {
            t.dump();
            assert(false && "Unsupported load type");
            return false;
         });
   }

   bool compile_func_constant_op(mlir::func::ConstantOp) {
      // we do not support function constants in the IR, so we just return true
      // this is a no-op
      return true;
   }

   // TODO: this is veeeery brittle -> test!
   bool compile_func_call_op(mlir::func::CallOp op) {
      const mlir::FlatSymbolRefAttr callee_attr = op.getCalleeAttr();
      mlir::func::FuncOp callee_func = mlir::cast<mlir::func::FuncOp>(op.resolveCallable());

      // we only call into the runtime => use C-CallConv (yes, this is a hack since the runtime is actually C++ code. works for now)
      auto call_conv_assigner = tpde::x64::CCAssignerSysV(false /* is_vararg */);
      auto builder = typename Derived::CallBuilder(*derived(), call_conv_assigner);

      for (size_t i = 0; i < op.getArgOperands().size(); ++i) {
         const mlir::Value arg = op.getArgOperands()[i];
         auto flag = Base::CallArg::Flag::none;
         if (const auto attrs = callee_func.getArgAttrs(); attrs.has_value()) {
            mlir::ArrayAttr attr = attrs.value();
            if (auto string_attr = mlir::dyn_cast<mlir::StringAttr>(attr[i]); string_attr && string_attr.getValue() == "llvm.zeroext") {
               flag = Base::CallArg::Flag::zext;
            } else {
               assert(0 && "Unsupported call argument attribute");
               return false;
            }
         }
         builder.add_arg(typename Base::CallArg{arg, flag, 0, 0}); // TODO: check byval_align and byval_size
      }

      assert(funcMap.contains(callee_func.getSymName()) && "Function not found in function map");
      uint32_t func_idx = funcMap[callee_func.getSymName()];
      assert(func_idx < this->func_syms.size() && "Function index out of bounds");
      typename Base::Assembler::SymRef func_ref = this->func_syms[func_idx];
      builder.call(func_ref);

      assert(op.getNumResults() == 1 && "Function call must have exactly one result in the IR");
      ValueRef res = this->result_ref(op.getResult(0));
      builder.add_ret(res);
      return true;
   }

   bool compile_inst(IRInstRef inst, InstRange) noexcept {
      return mlir::TypeSwitch<IRInstRef, bool>(inst)
         .Case<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp, mlir::arith::DivSIOp, mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp, mlir::arith::ShLIOp, mlir::arith::ShRUIOp>([&](auto op) {
            return compile_arith_binary_op(op);
         })
         .template Case<mlir::arith::CmpIOp>([&](auto op) {
            return derived()->compile_arith_cmp_int_op(op);
         })
         .template Case<mlir::cf::BranchOp>([&](auto op) {
            return compile_cf_br_op(op);
         })
         .template Case<mlir::cf::CondBranchOp>([&](auto op) {
            return derived()->compile_cf_cond_br_op(op);
         })
         .template Case<mlir::arith::ConstantOp>([&](auto op) {
            return compile_arith_const_op(op);
         })
         .template Case<dialect::util::SizeOfOp>([&](auto op) {
            return compile_util_sizeof_op(op);
         })
         .template Case<dialect::util::GenericMemrefCastOp>([&](auto op) {
            return compile_util_generic_memref_cast_op(op);
         })
         .template Case<dialect::util::TupleElementPtrOp>([&](auto op) {
            return derived()->compile_util_tuple_element_ptr_op(op);
         })
         .template Case<mlir::func::ConstantOp>([&](auto op) {
            return compile_func_constant_op(op);
         })
         .template Case<dialect::util::LoadOp>([&](auto op) {
            return compile_util_load_op(op);
         })
         .template Case<mlir::func::CallOp>([&](auto op) {
            return compile_func_call_op(op);
         })
         .Default([&](IRInstRef op) {
            error.emit() << "Encountered unimplemented instruction: " << op->getName().getStringRef().str() << "\n";
            op->dump();
            return false;
         });
   }
};
// NOLINTEND(readability-identifier-naming)

// x86_64 target specific compiler
// NOLINTBEGIN(readability-identifier-naming)
struct IRCompilerX64 : tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>, tpde_encodegen::EncodeCompiler<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig> {
   using Base = tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;
   using EncCompiler = tpde_encodegen::EncodeCompiler<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;

   std::unique_ptr<IRAdaptor> adaptor;

   explicit IRCompilerX64(std::unique_ptr<IRAdaptor>&& adaptor) : Base{adaptor.get()}, adaptor(std::move(adaptor)) {
      static_assert(tpde::Compiler<IRCompilerX64, tpde::x64::PlatformConfig>);
   }

   bool compile_arith_cmp_int_op(mlir::arith::CmpIOp op) {
      mlir::Type ty = op.getLhs().getType();
      assert(mlir::isa<mlir::IntegerType>(ty) && "Expected integer type for comparison operation");
      unsigned int_width = ty.getIntOrFloatBitWidth();
      Jump jump;

      bool is_signed = false;
      switch (op.getPredicate()) {
         case mlir::arith::CmpIPredicate::eq: jump = Jump::je; break;
         case mlir::arith::CmpIPredicate::ne: jump = Jump::jne; break;
         case mlir::arith::CmpIPredicate::ugt: jump = Jump::jg; break;
         case mlir::arith::CmpIPredicate::uge: jump = Jump::jge; break;
         case mlir::arith::CmpIPredicate::ult: jump = Jump::jl; break;
         case mlir::arith::CmpIPredicate::ule: jump = Jump::jle; break;
         case mlir::arith::CmpIPredicate::sgt: jump = Jump::jg; break;
         case mlir::arith::CmpIPredicate::sge: jump = Jump::jge; break;
         case mlir::arith::CmpIPredicate::slt: jump = Jump::jl; break;
         case mlir::arith::CmpIPredicate::sle: jump = Jump::jle; break;
         default: assert(0); return false;
      }
      switch (op.getPredicate()) {
         case mlir::arith::CmpIPredicate::sgt:
         case mlir::arith::CmpIPredicate::sge:
         case mlir::arith::CmpIPredicate::slt:
         case mlir::arith::CmpIPredicate::sle:
            is_signed = true;
            break;
         default: break;
      }

      // TODO: check if the result can be fused with a subsequent br instruction

      auto lhs = this->val_ref(op.getLhs());
      auto rhs = this->val_ref(op.getRhs());
      ScratchReg res_scratch{this};

      ValuePartRef lhs_pr = lhs.part(0);
      ValuePartRef rhs_pr = rhs.part(0);

      if (lhs_pr.is_const() && !rhs_pr.is_const()) {
         std::swap(lhs_pr, rhs_pr);
         jump = swap_jump(jump);
      }

      if (int_width != 32 && int_width != 64) {
         unsigned ext_bits = tpde::util::align_up(int_width, 32);
         lhs_pr = std::move(lhs_pr).into_extended(is_signed, int_width, ext_bits);
         rhs_pr = std::move(rhs_pr).into_extended(is_signed, int_width, ext_bits);
      }

      AsmReg lhs_reg = lhs_pr.has_reg() ? lhs_pr.cur_reg() : lhs_pr.load_to_reg();
      if (rhs_pr.is_const()) {
         uint64_t rhs_const = rhs_pr.const_data()[0];
         switch (int_width) {
            case 8: ASM(CMP8ri, lhs_reg, static_cast<uint8_t>(rhs_const)); break;
            case 16: ASM(CMP16ri, lhs_reg, static_cast<uint16_t>(rhs_const)); break;
            case 32: ASM(CMP32ri, lhs_reg, static_cast<uint32_t>(rhs_const)); break;
            case 64:
               // test if the constant fits into a signed 32-bit integer
               if (static_cast<int64_t>(static_cast<int32_t>(rhs_const)) == static_cast<int64_t>(rhs_const)) {
                  ASM(CMP64ri, lhs_reg, static_cast<int32_t>(rhs_const));
               } else {
                  ScratchReg scratch3{this};
                  auto tmp = scratch3.alloc_gp();
                  ASM(MOV64ri, tmp, rhs_const);
                  ASM(CMP64rr, lhs_reg, tmp);
               }
               break;
            default: assert(0); return false;
         }
      } else {
         auto rhs_reg = rhs_pr.load_to_reg();
         switch (int_width) {
            case 8: ASM(CMP8rr, lhs_reg, rhs_reg); break;
            case 16: ASM(CMP16rr, lhs_reg, rhs_reg); break;
            case 32: ASM(CMP32rr, lhs_reg, rhs_reg); break;
            case 64: ASM(CMP64rr, lhs_reg, rhs_reg); break;
            default: assert(0); return false;
         }
      }
      // TODO: why does this not work?
      // lhs.reset();
      // rhs.reset();

      auto [_, res_ref] = result_ref_single(op);
      generate_raw_set(jump, res_scratch.alloc_gp());
      set_value(res_ref, res_scratch);
      return true;
   }

   bool compile_cf_cond_br_op(mlir::cf::CondBranchOp op) {
      const auto true_block = op.getTrueDest();
      const auto false_block = op.getFalseDest();

      auto [_, cond_ref] = this->val_ref_single(op.getCondition());
      const auto cond_reg = cond_ref.load_to_reg();
      ASM(TEST32ri, cond_reg, 1);

      const auto next_block = this->analyzer.block_ref(this->next_block());

      const auto true_needs_split = this->branch_needs_split(true_block);
      const auto false_needs_split = this->branch_needs_split(false_block);

      const auto spilled = this->spill_before_branch();
      this->begin_branch_region();

      if (next_block == true_block || (next_block != false_block && true_needs_split)) {
         generate_branch_to_block(Jump::je, false_block, false_needs_split, false);
         generate_branch_to_block(Jump::jmp, true_block, false, true);
      } else if (next_block == false_block) {
         generate_branch_to_block(Jump::jne, true_block, true_needs_split, false);
         generate_branch_to_block(Jump::jmp, false_block, false, true);
      } else {
         assert(!true_needs_split);
         this->generate_branch_to_block(Jump::jne, true_block, false, false);
         this->generate_branch_to_block(Jump::jmp, false_block, false, true);
      }
      this->end_branch_region();
      this->release_spilled_regs(spilled);
   }

   void reset() noexcept {
      Base::reset();
      EncCompiler::reset();
   }

   Error& getError() { return Base::getError(); }
};
// NOLINTEND(readability-identifier-naming)

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
#if defined(__x86_64__)
      IRCompilerX64 compiler{std::make_unique<IRAdaptor>(&moduleOp, error)};
#else
#error "Baseline backend is only supported on x86_64 architecture."
#endif
      if (!compiler.compile()) {
         error.emit() << "Could not compile query module:\n"
                      << logSpoof.drain_logs() << "\n"
                      << compiler.adaptor->getError().emit().str() << "\n"
                      << compiler.getError().emit().str() << "\n";
         return;
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
   return std::make_unique<BaselineBackend>();
}
#endif