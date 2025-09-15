#pragma once

#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/execution/Backend.h"
#include "lingodb/execution/baseline/utils.hpp"
#include "lingodb/catalog/FunctionCatalogEntry.h"

#include "Adaptor.hpp"
#include "CompilerConfig.hpp"

#include <tpde/CompilerBase.hpp>

#include <iostream>

namespace lingodb::execution::baseline {
// NOLINTBEGIN(readability-identifier-naming)

// cross-platform compiler base class
template <typename Adaptor, typename Derived, typename Config>
struct IRCompilerBase : tpde::CompilerBase<IRAdaptor, Derived, Config> {
   using Base = tpde::CompilerBase<IRAdaptor, Derived, Config>;
   using IR = IRAdaptor::IR;
   using ValuePartRef = typename Base::ValuePartRef;
   using GenericValuePart = typename Base::GenericValuePart;
   using Expr = typename Base::GenericValuePart::Expr;
   using ScratchReg = typename Base::ScratchReg;
   using ValueRef = typename Base::ValueRef;
   using ValuePart = typename Base::ValuePart;
   using InstRange = typename Base::InstRange;
   using IRInstRef = IRAdaptor::IRInstRef;
   using IRValueRef = IRAdaptor::IRValueRef;
   using AsmReg = typename Base::AsmReg;
   using Assembler = typename Base::Assembler;
   using SymRef = tpde::SymRef;

   class BuiltinFuncStorage {
      public:
      enum class Type : uint8_t {
         divti3,
         udivti3,
         modti3,
         umodti3,
         MAX, // sentinel counter-element
      };

      BuiltinFuncStorage()
         : funcs() { std::fill(funcs.begin(), funcs.end(), SymRef{}); }

      SymRef get_symbol(IRCompilerBase* compiler, Type type) {
         if (SymRef ref = funcs[static_cast<unsigned>(type)]; ref != SymRef{}) {
            return ref;
         } else {
            std::string name;
            // https://gcc.gnu.org/onlinedocs/gccint/Integer-library-routines.html
            switch (type) {
               case Type::divti3:
                  name = "__divti3";
                  break;
               case Type::udivti3:
                  name = "__udivti3";
                  break;
               case Type::modti3:
                  name = "__modti3";
                  break;
               case Type::umodti3:
                  name = "__umodti3";
                  break;
               default: __builtin_unreachable();
            }
            auto sym = compiler->assembler.sym_add_undef(name, Assembler::SymBinding::GLOBAL);
            funcs[static_cast<size_t>(type)] = sym;
            return sym;
         }
      }

      private:
      std::array<SymRef, static_cast<size_t>(Type::MAX)> funcs;
   };

   BuiltinFuncStorage builtins;
   llvm::BumpPtrAllocator allocator;

   Error error;
   // non-external function name -> idx lookup map
   llvm::StringMap<uint32_t> localFuncMap;
   // external function name -> ptr lookup map
   llvm::StringMap<void*> externFuncMap;

   IRCompilerBase(IRAdaptor* adaptor)
      : Base{adaptor} {
      static_assert(tpde::Compiler<Derived, Config>);
      static_assert(std::is_same_v<Adaptor, IRAdaptor>, "Adaptor must be IRAdaptor");

      dialect::util::FunctionHelper::visitAllFunctions([&](std::string s, void* ptr) { externFuncMap[s] = ptr; });
      execution::visitBareFunctions([&](std::string s, void* ptr) { externFuncMap[s] = ptr; });
      catalog::visitUDFFunctions1([&](std::string s, void* ptr) { externFuncMap[s] = ptr; });

   }

   Error& getError() { return error; }

   // shortcuts to access the derived class later
   Derived* derived() noexcept { return static_cast<Derived*>(this); }

   const Derived* derived() const noexcept { return static_cast<Derived*>(this); }

   const IR* ir() const noexcept { return this->adaptor->module; }

   static bool cur_func_may_emit_calls() { return true; }

   static tpde::SymRef cur_personality_func() {
      // we do not support exceptions, so we do not need a personality function
      return {};
   }

   static bool try_force_fixed_assignment(IRAdaptor::IRValueRef) noexcept { return false; }

   struct ValueParts {
      mlir::Type valType;

      uint32_t count() const noexcept {
         return mlir::TypeSwitch<mlir::Type, uint32_t>(valType)
            .Case<mlir::IntegerType>([](auto intType) {
               return (intType.getIntOrFloatBitWidth() + 64 - 1) / 64;
            })
            .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) { return 2; })
            .template Case<mlir::TupleType>([](const mlir::TupleType t) { return TupleHelper::numSlots(t); })
            .Default([](mlir::Type t) { return 1; });
      }

      uint32_t size_bytes(uint32_t part_idx) const noexcept {
         mlir::Type typeAtSlot;
         if (mlir::isa<mlir::TupleType>(valType)) {
            typeAtSlot = TupleHelper::typeAtSlot(mlir::cast<mlir::TupleType>(valType), part_idx);
            part_idx = 0; // reset idx for later following logic
         } else {
            typeAtSlot = valType;
         }
         return mlir::TypeSwitch<mlir::Type, uint32_t>(typeAtSlot)
            .Case<mlir::IntegerType, mlir::FloatType>([&](auto numType) {
               assert(
                  part_idx < (numType.getIntOrFloatBitWidth() + 64 - 1) / 64 &&
                  "Part index out of range for integer or float type");
               // integer and float types are sized by their bit width
               return (numType.getIntOrFloatBitWidth() % 65 + 8 - 1) / 8;
            })
            .template Case<mlir::IndexType, dialect::util::RefType, mlir::FunctionType>([](auto) {
               return 8;
            })
            .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
               assert(part_idx < 2 && "VarLen32 and Buffer types are made of two parts");
               // these container types are 2x64-bit
               return 8;
            })
            .Default([](const mlir::Type t) {
               t.dump();
               assert(0 && "invalid type");
               return 0;
            }); // all other types are not supported yet
      }

      tpde::RegBank reg_bank(uint32_t part_idx) const noexcept {
         mlir::Type typeAtSlot;
         if (mlir::isa<mlir::TupleType>(valType)) {
            typeAtSlot = TupleHelper::typeAtSlot(mlir::cast<mlir::TupleType>(valType), part_idx);
         } else {
            typeAtSlot = valType;
         }
         return mlir::TypeSwitch<mlir::Type, tpde::RegBank>(typeAtSlot)
            .Case<mlir::IntegerType, mlir::IndexType, dialect::util::RefType, mlir::FunctionType,
                  dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
               return Config::GP_BANK;
            })
            .template Case<mlir::FloatType>([](auto) { return Config::FP_BANK; })
            .Default([](const mlir::Type t) {
               t.dump();
               assert(0 && "invalid type");
               return Config::GP_BANK;
            });
      }
   };

   static ValueParts val_parts(IRAdaptor::IRValueRef value) { return ValueParts{value.getType()}; }

   struct ValRefSpecial {
      uint8_t mode = 4;
      IRAdaptor::IRValueRef value;
   };

   static std::optional<ValRefSpecial> val_ref_special(IRAdaptor::IRValueRef val) {
      if (const auto* op = val.getDefiningOp()) {
         return mlir::TypeSwitch<const mlir::Operation*, std::optional<ValRefSpecial>>(op)
            .template Case<mlir::arith::ConstantOp, dialect::util::SizeOfOp, dialect::util::UndefOp>(
               [&](auto) { return ValRefSpecial{.mode = 4, .value = val}; })
            .template Case<dialect::util::CreateConstVarLen>(
               [&](auto constVarLenOp) -> std::optional<ValRefSpecial> {
                  const mlir::StringRef content = mlir::cast<mlir::StringAttr>(
                                                     constVarLenOp->getAttr(constVarLenOp.getStrAttrName()))
                                                     .getValue();
                  if (content.size() <= 12) {
                     return ValRefSpecial{.mode = 4, .value = val};
                  } else {
                     // long string require a pointer to the content, making them non-constant
                     return std::nullopt;
                  }
               })
            .Default([&](auto) { return std::nullopt; });
      }
      return std::nullopt;
   }

   ValuePart val_part_ref_special(ValRefSpecial& ref, const uint32_t part) noexcept {
      if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(ref.value.getDefiningOp())) {
         if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constOp.getValue())) {
            const mlir::APInt containedInt = intAttr.getValue();
            unsigned bitWidth;
            if (mlir::isa<mlir::IndexType>(intAttr.getType())) {
               bitWidth = 64;
            } else {
               bitWidth = intAttr.getType().getIntOrFloatBitWidth();
            }
            if (bitWidth <= 64) {
               assert(part == 0);
               return ValuePartRef(this, containedInt.getRawData()[0], std::max<uint32_t>(1, (bitWidth + 7) / 8), Config::GP_BANK);
            }
            if (bitWidth == 128) {
               assert(part < 2 && "Part index out of range for 128-bit integer");
               return ValuePartRef(this, containedInt.getRawData()[part], 8, Config::GP_BANK);
            }
            assert(0 && "Unsupported integer type in val_ref_special");
            return ValuePartRef{this};
         }
         if (const auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(constOp.getValue())) {
            assert(part == 0);
            uint64_t containedFloat = floatAttr.getValue().bitcastToAPInt().getRawData()[0];
            return ValuePartRef(this, containedFloat, floatAttr.getType().getIntOrFloatBitWidth() / 8,
                                Config::FP_BANK);
         }
         assert(0 && "Unsupported constant type in val_ref_special");
         return ValuePartRef{this};
      }
      if (const auto utilSizeOfOp = mlir::dyn_cast_or_null<dialect::util::SizeOfOp>(ref.value.getDefiningOp())) {
         assert(part == 0);
         const mlir::TypeAttr type_attr = mlir::cast<mlir::TypeAttr>(utilSizeOfOp->getAttr("type"));
         const uint64_t tuple_size = TupleHelper::sizeAndPadding(mlir::cast<mlir::TupleType>(type_attr.getValue())).first;
         return ValuePartRef(this, tuple_size, 8, Config::GP_BANK);
      }
      if (const auto constVarLenOp = mlir::dyn_cast_or_null<dialect::util::CreateConstVarLen>(
             ref.value.getDefiningOp())) {
         assert(constVarLenOp->getAttrs().size() == 1);
         const mlir::StringRef content = mlir::cast<mlir::StringAttr>(
                                            constVarLenOp->getAttrs().front().getValue())
                                            .getValue();
         const size_t len = content.size();

         /* VarLen32 is stored in a 128-bit int:
                 * The first 4 bytes of the content are used to store the length of the content.
                 * The next 4 bytes are used to store the first 4 bytes (chars) of the content.
                 * This is done to efficiently store short strings.
                 * The next 8 bytes are either a pointer to the full string or 8 chars if the string <= 12 chars
                 */
         if (part == 0) {
            uint64_t first4 = 0;
            memcpy(&first4, content.data(), std::min(4ul, len));
            uint64_t c1 = (first4 << 32) | len;
            return ValuePartRef(this, c1, 8, Config::GP_BANK);
         }
         if (part == 1) {
            if (len <= 12) {
               uint64_t last8 = 0;
               if (len > 4) { memcpy(&last8, content.data() + 4, std::min(8ul, len - 4)); }
               return ValuePartRef(this, last8, 8, Config::GP_BANK);
            } else {
               assert(0 && "VarLen32 type with part index 1 should not be used for long strings");
               return ValuePartRef{this};
            }
         }
         assert(0 && "Part index out of range for VarLen32 type");
         return ValuePartRef{this};
      }
      if (auto undef_op = mlir::dyn_cast<dialect::util::UndefOp>(ref.value.getDefiningOp())) {
         auto ret_type = undef_op.getRes().getType();
         if (mlir::isa<mlir::IntegerType>(ret_type)) {
            const unsigned width = ret_type.getIntOrFloatBitWidth();
            const unsigned register_width = width == 128 ? 8 : std::max<unsigned>(1, (width + 7) / 8);
            return ValuePartRef(this, 0, register_width, Config::GP_BANK);
         } else if (mlir::isa<mlir::FloatType>(ret_type)) {
            return ValuePartRef(this, 0, ret_type.getIntOrFloatBitWidth() / 8, Config::FP_BANK);
         } else if (mlir::isa<dialect::util::VarLen32Type, dialect::util::BufferType>(ret_type)) {
            return ValuePartRef(this, 0, 8, Config::GP_BANK);
         } else {
            assert(0);
            return ValuePartRef{this};
         }
      }
      ref.value.dump();
      assert(0 && "Unsupported value in val_ref_special");
      return ValuePartRef{this};
   }

   static bool arg_is_int128(IRAdaptor::IRValueRef val) noexcept {
      return mlir::isa<dialect::util::VarLen32Type, dialect::util::BufferType>(val.getType()) || (mlir::isa<mlir::IntegerType>(val.getType()) && val.getType().getIntOrFloatBitWidth() == 128);
   }

   static bool arg_allow_split_reg_stack_passing(IRAdaptor::IRValueRef) noexcept {
      // we do not support split register stack passing
      return false;
   }

   void define_func_idx(IRAdaptor::IRFuncRef func, uint32_t idx) {
      localFuncMap[func.getSymName()] = idx;
   }

   // [ptr + idx * size_of(elem_t)]
   std::optional<GenericValuePart> create_idx_offset_expr(ValuePartRef base, IRValueRef idx,
                                                          const mlir::Type elem_t) {
      const auto elem_size = get_size(elem_t);
      if (!elem_size) {
         assert(0 && "Unsupported type for store operation");
         error.emit() << "Unsupported type for store operation.";
         return std::nullopt;
      }

      if (idx) {
         if (auto idx_op = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(idx.getDefiningOp())) {
            AsmReg ptr_reg = base.load_to_reg();
            return GenericValuePart{
               Expr{
                  std::move(ptr_reg), static_cast<int64_t>(static_cast<size_t>(idx_op.value()) * (*elem_size))}};
         }
         auto [_, idx_ref] = this->val_ref_single(idx);
         auto generic_ptr_expr = Expr{base.load_to_reg()};
         generic_ptr_expr.index = idx_ref.load_to_reg();
         generic_ptr_expr.scale = *elem_size;
         return GenericValuePart{std::move(generic_ptr_expr)};
      }
      return GenericValuePart{std::move(base)};
   }

   bool compile_arith_binary_op(IRInstRef op) {
      const auto res_type = op->getResult(0).getType();
      unsigned res_width;
      unsigned op_width;
      if (mlir::isa<mlir::IndexType>(res_type)) {
         op_width = 64;
         res_width = 64;
      } else if (mlir::isa<mlir::IntegerType>(res_type)) {
         res_width = res_type.getIntOrFloatBitWidth();
         switch (res_width) {
            case 1:
            case 8:
            case 16:
            case 32:
               op_width = 32;
               break;
            case 64:
               op_width = 64;
               break;
            case 128:
               op_width = 128;
               break;
            default:
               op->dump();
               assert(0 && "Unsupported integer type width for arithmetic operation");
               return false;
         }
      } else if (mlir::isa<mlir::FloatType>(res_type)) {
         res_width = res_type.getIntOrFloatBitWidth();
         switch (res_width) {
            case 32:
               op_width = 32;
               break;
            case 64:
               op_width = 64;
               break;
            default:
               res_type.dump();
               assert(0 && "Unsupported float type width for arithmetic operation");
               return false;
         }
      } else {
         assert(0 && "Unsupported type for arithmetic operation");
         return false;
      }

      if (op_width == 128) {
         auto res = this->result_ref(op->getResult(0));
         auto builtin_func = mlir::TypeSwitch<mlir::Operation*, std::optional<SymRef>>(op)
                                .Case([&](mlir::arith::DivUIOp) {
                                   return builtins.get_symbol(derived(), BuiltinFuncStorage::Type::udivti3);
                                })
                                .Case([&](mlir::arith::DivSIOp) {
                                   return builtins.get_symbol(derived(), BuiltinFuncStorage::Type::divti3);
                                })
                                .Case([&](mlir::arith::RemUIOp) {
                                   return builtins.get_symbol(derived(), BuiltinFuncStorage::Type::umodti3);
                                })
                                .Case([&](mlir::arith::RemSIOp) {
                                   return builtins.get_symbol(derived(), BuiltinFuncStorage::Type::modti3);
                                })
                                .Default([&](auto) {
                                   return std::nullopt;
                                });
         if (builtin_func.has_value()) {
            std::array args{op->getOperand(0), op->getOperand(1)};
            derived()->create_helper_call(args, &res, *builtin_func);
            return true;
         }

         // the shifts (even when written as a snippet) do not take a 2-part 128-bit int as the shift amount
         if (op->getName().getStringRef().str() == "arith.shrui") {
            auto lhs_vr = this->val_ref(op->getOperand(0));
            auto rhs_vr = this->val_ref(op->getOperand(1));
            // special case: only take rhs_vr.part(0) as the shift amount, ignore part(1)
            return derived()->encode_arith_shr_u128(std::move(lhs_vr.part(0)), std::move(lhs_vr.part(1)), std::move(rhs_vr.part(0)), res.part(0), res.part(1));
         }

         static llvm::SmallDenseMap<llvm::StringRef, bool (Derived::*)(GenericValuePart&&, GenericValuePart&&, GenericValuePart&&, GenericValuePart&&, ValuePart&&, ValuePart&&), 8> encoder_lookup = {
            {"arith.addi", &Derived::encode_arith_add_i128},
            {"arith.andi", &Derived::encode_arith_land_i128},
            {"arith.ori", &Derived::encode_arith_lor_i128},
            {"arith.xori", &Derived::encode_arith_lxor_i128},
            {"arith.muli", &Derived::encode_arith_mul_i128},
            {"arith.subi", &Derived::encode_arith_sub_i128},
         };
#ifndef NDEBUG
         if (!encoder_lookup.contains(op->getName().getStringRef())) {
            op->dump();
            error.emit() << op->getName().getStringRef().str().c_str() << " is not supported by the baseline backend\n";
            return false;
         }
#endif
         const auto encoder = encoder_lookup[op->getName().getStringRef()];

         auto lhs_vr = this->val_ref(op->getOperand(0));
         auto rhs_vr = this->val_ref(op->getOperand(1));
         (derived()->*encoder)(std::move(lhs_vr.part(0)), std::move(lhs_vr.part(1)), std::move(rhs_vr.part(0)), std::move(rhs_vr.part(1)), res.part(0), res.part(1));
         return true;
      } else {
         auto lhs_vr = this->val_ref(op->getOperand(0));
         auto rhs_vr = this->val_ref(op->getOperand(1));
         auto lhs_pr = lhs_vr.part(0);
         auto rhs_pr = rhs_vr.part(0);
         // move constant operands to the right side
         if ((mlir::isa<mlir::arith::AddIOp, mlir::arith::MulIOp, mlir::arith::AndIOp, mlir::arith::OrIOp,
                        mlir::arith::XOrIOp>(op)) &&
             lhs_pr.is_const() && !rhs_pr.is_const()) {
            std::swap(lhs_vr, rhs_vr);
            std::swap(lhs_pr, rhs_pr);
         }

         if (op_width > res_width) {
            lhs_pr = std::move(lhs_pr).into_extended(true, res_width, op_width);
            rhs_pr = std::move(rhs_pr).into_extended(true, res_width, op_width);
         }

         auto lhs_op = GenericValuePart{std::move(lhs_pr)};
         auto rhs_op = GenericValuePart{std::move(rhs_pr)};

         auto [res_vr, res_pr] = this->result_ref_single(op->getResult(0));

         // encode functions for 32/64 bit operations
         static llvm::SmallDenseMap<llvm::StringRef, std::array<bool (Derived::*)(GenericValuePart&&, GenericValuePart&&, ValuePart&&), 2>, 16> encoder_lookup_int;
         static llvm::SmallDenseMap<llvm::StringRef, std::array<bool (Derived::*)(GenericValuePart&&, GenericValuePart&&, ValuePart&&), 2>, 4> encoder_lookup_float;

         auto encoder_call = [&](auto encoder_table) {
#ifndef NDEBUG
            if (!encoder_table.contains(op->getName().getStringRef())) {
               op->dump();
               std::cerr << op->getName().getStringRef().str().c_str() << " is not supported by the baseline backend\n";
               assert(0);
               return false;
            }
#endif
            const auto encoders = encoder_table[op->getName().getStringRef()];
            const auto sub_encoder_idx = op_width == 64 ? 1 : 0;
            return (derived()->*encoders[sub_encoder_idx])(std::move(lhs_op), std::move(rhs_op), std::move(res_pr));
         };
         if (mlir::isa<mlir::FloatType>(res_type)) {
            if (encoder_lookup_float.empty()) {
               encoder_lookup_float = {
                  {"arith.addf", {&Derived::encode_arith_add_f32, &Derived::encode_arith_add_f64}},
                  {"arith.subf", {&Derived::encode_arith_sub_f32, &Derived::encode_arith_sub_f64}},
                  {"arith.mulf", {&Derived::encode_arith_mul_f32, &Derived::encode_arith_mul_f64}},
                  {"arith.divf", {&Derived::encode_arith_div_f32, &Derived::encode_arith_div_f64}},
               };
            }
            return encoder_call(encoder_lookup_float);
         } else {
            if (encoder_lookup_int.empty()) {
               encoder_lookup_int = {
                  {"arith.addi", {&Derived::encode_arith_add_i32, &Derived::encode_arith_add_i64}},
                  {"arith.subi", {&Derived::encode_arith_sub_i32, &Derived::encode_arith_sub_i64}},
                  {"arith.muli", {&Derived::encode_arith_mul_i32, &Derived::encode_arith_mul_i64}},
                  {"arith.divsi", {&Derived::encode_arith_sdiv_i32, &Derived::encode_arith_sdiv_i64}},
                  {"arith.divui", {&Derived::encode_arith_udiv_i32, &Derived::encode_arith_udiv_i64}},
                  {"arith.remsi", {&Derived::encode_arith_srem_i32, &Derived::encode_arith_srem_i64}},
                  {"arith.remui", {&Derived::encode_arith_urem_i32, &Derived::encode_arith_urem_i64}},
                  {"arith.ori", {&Derived::encode_arith_lor_i32, &Derived::encode_arith_lor_i64}},
                  {"arith.xori", {&Derived::encode_arith_lxor_i32, &Derived::encode_arith_lxor_i64}},
                  {"arith.andi", {&Derived::encode_arith_land_i32, &Derived::encode_arith_land_i64}},
                  {"arith.shrui", {&Derived::encode_arith_shr_u32, &Derived::encode_arith_shr_u64}},
                  {"arith.shli", {&Derived::encode_arith_shl_i32, &Derived::encode_arith_shl_i64}},
                  {"arith.minsi", {&Derived::encode_arith_minsi_i32, &Derived::encode_arith_minsi_i64}},
                  {"arith.maxsi", {&Derived::encode_arith_maxsi_i32, &Derived::encode_arith_maxsi_i64}},
                  {"arith.minui", {&Derived::encode_arith_minui_i32, &Derived::encode_arith_minui_i64}},
                  {"arith.maxui", {&Derived::encode_arith_maxui_i32, &Derived::encode_arith_maxui_i64}},
               };
            }
            return encoder_call(encoder_lookup_int);
         }
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
      unsigned elementOffset = TupleHelper::getElementOffset(tuple_type, op.getIdx());

      const auto dst = op->getResult(0);
      assert(val_parts(base_ref).count() == 1);
      assert(val_parts(dst).count() == 1);
      auto base_vr = this->val_ref(base_ref);
      auto res_vr = this->result_ref(dst);

      // create a base + offset expression
      AsmReg base_reg = base_vr.part(0).load_to_reg();
      GenericValuePart addr = typename GenericValuePart::Expr{base_reg, elementOffset};

      // load value to register (e.g. mov + add / lea for x86_64)
      AsmReg res_reg = derived()->gval_expr_as_reg(addr);
      ScratchReg res_scratch{derived()};
      derived()->mov(res_scratch.alloc_gp(), res_reg, 8);
      res_vr.part(0).set_value(std::move(res_scratch));
      return true;
   }

   bool compile_util_buffer_cast_op(dialect::util::BufferCastOp op) {
      auto src_vr = this->val_ref(op.getVal());
      auto res_vr = this->result_ref(op.getRes());
      {
         ScratchReg res_scratch{derived()};
         derived()->mov(res_scratch.alloc_gp(), src_vr.part(0).load_to_reg(), 8);
         res_vr.part(0).set_value(std::move(res_scratch));
      }
      {
         ScratchReg res_scratch{derived()};
         derived()->mov(res_scratch.alloc_gp(), src_vr.part(1).load_to_reg(), 8);
         res_vr.part(1).set_value(std::move(res_scratch));
      }
      return true;
   }

   bool compile_util_buffer_get_element_ref_op(dialect::util::BufferGetElementRef op) {
      const mlir::TypedValue<dialect::util::BufferType> buf = op.getBuffer();
      const mlir::TypedValue<mlir::IndexType> idx = op.getIdx();
      const mlir::Type elem_type = buf.getType().getT();

      const auto dst = op->getResult(0);
      assert(val_parts(buf).count() == 2);
      assert(val_parts(dst).count() == 1);
      auto buf_vr = this->val_ref(buf);

      auto offset_expr = create_idx_offset_expr(std::move(buf_vr.part(1)), idx, elem_type);
      if (!offset_expr) {
         return false;
      }

      // load value to register (e.g. mov + add / lea for x86_64)
      AsmReg res_reg = derived()->gval_expr_as_reg(*offset_expr);
      ScratchReg res_scratch{derived()};
      derived()->mov(res_scratch.alloc_gp(), res_reg, 8);

      auto [_, res_vr] = this->result_ref_single(dst);
      res_vr.set_value(std::move(res_scratch));
      return true;
   }

   bool compile_util_store_op(dialect::util::StoreOp op) {
      const mlir::Value in = op.getVal();
      const mlir::TypedValue<dialect::util::RefType> ptr = op.getRef();
      const mlir::TypedValue<mlir::IndexType> idx = op.getIdx();
      const mlir::Type stored_type = in.getType();

      auto [_, ptr_ref] = this->val_ref_single(ptr);
      auto offset_expr = create_idx_offset_expr(std::move(ptr_ref), idx, stored_type);
      if (!offset_expr) {
         return false;
      }

      auto in_vr = this->val_ref(in);
      return mlir::TypeSwitch<mlir::Type, bool>(stored_type)
         .Case([&](const mlir::IntegerType t) {
            switch (t.getIntOrFloatBitWidth()) {
               case 1: return derived()->encode_util_store_i1(std::move(*offset_expr), in_vr.part(0));
               case 8: return derived()->encode_util_store_i8(std::move(*offset_expr), in_vr.part(0));
               case 16: return derived()->encode_util_store_i16(std::move(*offset_expr), in_vr.part(0));
               case 32: return derived()->encode_util_store_i32(std::move(*offset_expr), in_vr.part(0));
               case 64: return derived()->encode_util_store_i64(std::move(*offset_expr), in_vr.part(0));
               case 128: return derived()->encode_store_i128(
                  std::move(*offset_expr), in_vr.part(0), in_vr.part(1));
               default:
                  assert(0 && "Unsupported integer type width for store operation");
                  return false;
            }
         })
         .Case([&](const mlir::FloatType t) {
            switch (t.getIntOrFloatBitWidth()) {
               case 32: return derived()->encode_util_store_f32(std::move(*offset_expr), in_vr.part(0));
               case 64: return derived()->encode_util_store_f64(std::move(*offset_expr), in_vr.part(0));
               default:
                  assert(0 && "Unsupported float type width for store operation");
                  return false;
            }
         })
         .template Case<dialect::util::RefType, mlir::IndexType>([&](auto) {
            return derived()->encode_util_store_i64(std::move(*offset_expr), in_vr.part(0));
         })
         .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
            return derived()->encode_store_i128(std::move(*offset_expr), in_vr.part(0), in_vr.part(1));
         })
         .Default([](mlir::Type t) {
            t.dump();
            assert(false && "Unsupported load type");
            return false;
         });
   }

   bool compile_util_load_op(dialect::util::LoadOp op) {
      const mlir::TypedValue<dialect::util::RefType> ptr = op.getRef();
      const mlir::TypedValue<mlir::IndexType> idx = op.getIdx();
      const mlir::Type loaded_type = op.getVal().getType();

      auto [_, ptr_ref] = this->val_ref_single(ptr);
      auto offset_expr = create_idx_offset_expr(std::move(ptr_ref), idx, loaded_type);
      if (!offset_expr) {
         return false;
      }

      auto res = this->result_ref(op.getVal());
      return mlir::TypeSwitch<mlir::Type, bool>(loaded_type)
         .Case([&](const mlir::IntegerType t) {
            switch (t.getIntOrFloatBitWidth()) {
               case 1: return derived()->encode_util_load_i1(std::move(*offset_expr), res.part(0));
               case 8: return derived()->encode_util_load_i8(std::move(*offset_expr), res.part(0));
               case 16: return derived()->encode_util_load_i16(std::move(*offset_expr), res.part(0));
               case 32: return derived()->encode_util_load_i32(std::move(*offset_expr), res.part(0));
               case 64: return derived()->encode_util_load_i64(std::move(*offset_expr), res.part(0));
               case 128: return derived()->encode_load_i128(std::move(*offset_expr), res.part(0), res.part(1));
               default:
                  assert(false && "Unsupported integer type width for load operation");
                  return false;
            }
         })
         .Case([&](const mlir::FloatType t) {
            switch (t.getIntOrFloatBitWidth()) {
               case 32: return derived()->encode_util_load_f32(std::move(*offset_expr), res.part(0));
               case 64: return derived()->encode_util_load_f64(std::move(*offset_expr), res.part(0));
               default:
                  assert(false && "Unsupported float type width for load operation");
                  return false;
            }
         })
         .template Case<dialect::util::RefType, mlir::IndexType>([&](auto) {
            return derived()->encode_util_load_i64(std::move(*offset_expr), res.part(0));
         })
         .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
            return derived()->encode_load_i128(std::move(*offset_expr), res.part(0), res.part(1));
         })
         .Default([](const mlir::Type t) {
            t.dump();
            assert(false && "Unsupported load type");
            return false;
         });
   }

   bool compile_func_call_op(mlir::func::CallOp op) {
      auto callee_func = mlir::cast<mlir::func::FuncOp>(op.resolveCallable());

      // we only call into the runtime => use C-CallConv (yes, this is a hack since the runtime is actually C++ code. works for now)
      auto builder = derived()->create_call_builder();
      for (size_t i = 0; i < op.getArgOperands().size(); ++i) {
         const mlir::Value arg = op.getArgOperands()[i];
         auto flag = Base::CallArg::Flag::none;
         if (const auto attrs = callee_func.getArgAttrs(); attrs.has_value()) {
            mlir::ArrayAttr attr = attrs.value();
            if (auto string_attr = mlir::dyn_cast<mlir::StringAttr>(attr[i]);
                string_attr && string_attr.getValue() == "llvm.zeroext") { flag = Base::CallArg::Flag::zext; }
         }
         builder.add_arg(typename Base::CallArg{arg, flag, 0, 0});
      }

      if (callee_func.isExternal()) {
         assert(
            externFuncMap.contains(callee_func.getSymName()) && "Function not found in external function map");
         ValuePart funcPtrRef{
            std::bit_cast<uint64_t>(externFuncMap[callee_func.getSymName()]), 8, Config::GP_BANK};
         builder.call(std::move(funcPtrRef));
      } else {
         assert(localFuncMap.contains(callee_func.getSymName()) && "Function not found in local function map");
         uint32_t func_idx = localFuncMap[callee_func.getSymName()];
         assert(func_idx < this->func_syms.size() && "Function index out of bounds");
         SymRef func_ref = this->func_syms[func_idx];
         builder.call(std::move(func_ref));
      }

      for (size_t i = 0; i < op.getNumResults(); ++i) {
         ValueRef res = this->result_ref(op.getResult(i));
         builder.add_ret(res);
      }
      return true;
   }

   bool compile_func_return_op(const mlir::func::ReturnOp op) {
      assert(!this->adaptor->cur_func.getResAttrs() && "we do not support return attributes yet");
      typename Derived::RetBuilder rb{*derived(), *derived()->cur_cc_assigner()};
      for (size_t i = 0; i < op->getNumOperands(); ++i) {
         rb.add(op->getOperand(i));
      }
      rb.ret();
      return true;
   }

   // zext and sext operations
   bool compile_arith_exti_op(const auto op, const bool sign) {
      mlir::Value src_val = op->getOperand(0);
      mlir::Value res_val = op->getResult(0);
      assert(
         mlir::isa<mlir::IntegerType>(src_val.getType()) &&
         "Source value must be an integer type for zext operation");
      unsigned src_width = src_val.getType().getIntOrFloatBitWidth();
      unsigned dst_width = res_val.getType().getIntOrFloatBitWidth();
      assert(
         src_width < dst_width &&
         "Source width must be less than or equal to destination width for zext operation");
      assert(
         src_width <= 64 && (dst_width <= 64 || dst_width == 128) &&
         "Source and destination widths must be less than or equal to 64 bits for zext operation");

      auto [_, src_vpr] = this->val_ref_single(src_val);
      auto res = this->result_ref(res_val);
      if (dst_width != 128) {
         src_vpr = std::move(src_vpr).into_extended(sign, src_width, dst_width);
         res.part(0).set_value(std::move(src_vpr));
      } else {
         if (sign) {
            switch (src_width) {
               case 1:
               case 8:
                  derived()->encode_arith_sext_i8_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               case 16:
                  derived()->encode_arith_sext_i16_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               case 32:
                  derived()->encode_arith_sext_i32_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               case 64:
                  derived()->encode_arith_sext_i64_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               default:
                  return false;
            }
         } else {
            switch (src_width) {
               case 1:
               case 8:
                  derived()->encode_arith_zext_i8_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               case 16:
                  derived()->encode_arith_zext_i16_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               case 32:
                  derived()->encode_arith_zext_i32_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               case 64:
                  derived()->encode_arith_zext_i64_i128(std::move(src_vpr), res.part(0), res.part(1));
                  break;
               default:
                  return false;
            }
         }
      }
      return true;
   }

   bool compile_arith_extf_op(const auto op) {
      mlir::Value src_val = op->getOperand(0);
      mlir::Value res_val = op->getResult(0);
      assert(mlir::isa<mlir::FloatType>(src_val.getType()) && mlir::isa<mlir::FloatType>(res_val.getType()) &&
             "Source and destination value must be a float type for extend operation");
      unsigned src_width = src_val.getType().getIntOrFloatBitWidth();
      unsigned dst_width = res_val.getType().getIntOrFloatBitWidth();
      assert(src_width == 32 && dst_width == 64 && "For extend operation, source must be float and destination must be double");
      auto [_, src_vpr] = this->val_ref_single(src_val);
      auto res = this->result_ref(res_val);
      return derived()->encode_arith_extf_f32_f64(std::move(src_vpr), res.part(0));
   }

   bool compile_arith_select_op(mlir::arith::SelectOp op) {
      mlir::Value cond = op.getCondition();
      mlir::Value lhs = op.getTrueValue();
      mlir::Value rhs = op.getFalseValue();
      auto cond_vr = this->val_ref(cond);
      auto lhs_vr = this->val_ref(lhs);
      auto rhs_vr = this->val_ref(rhs);
      auto res = this->result_ref(op.getResult());

      assert(lhs.getType() == rhs.getType() && "Both operands of select must be of the same type");
      bool success = true;
      const auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(lhs.getType());
      const size_t loopIterations = tupleType ? TupleHelper::numSlots(tupleType) : 1;
      size_t i = 0;
      while (i < loopIterations) {
         auto cond_vpr = loopIterations > 1 ? cond_vr.part_unowned(0) : cond_vr.part(0);
         success &= llvm::TypeSwitch<mlir::Type, bool>(tupleType ? TupleHelper::typeAtSlot(tupleType, i) : lhs.getType())
                       .template Case<mlir::IntegerType>([&](const mlir::IntegerType t) {
                          switch (t.getIntOrFloatBitWidth()) {
                             case 1:
                             case 8:
                             case 16:
                             case 32: return derived()->encode_arith_select_i32(std::move(cond_vpr), lhs_vr.part(i), rhs_vr.part(i), res.part(i));
                             case 64: return derived()->encode_arith_select_i64(std::move(cond_vpr), lhs_vr.part(i), rhs_vr.part(i), res.part(i));
                             case 128: {
                                const auto ret = derived()->encode_arith_select_i128(
                                   std::move(cond_vpr),
                                   lhs_vr.part(i), lhs_vr.part(i + 1),
                                   rhs_vr.part(i), rhs_vr.part(i + 1),
                                   res.part(i), res.part(i + 1));
                                i++;
                                return ret;
                             }
                             default:
                                assert(0 && "Unsupported integer type width for select operation");
                                return false;
                          }
                       })
                       .template Case<mlir::FloatType>([&](const mlir::FloatType t) {
                          switch (t.getIntOrFloatBitWidth()) {
                             case 32: return derived()->encode_arith_select_f32(std::move(cond_vpr), lhs_vr.part(i), rhs_vr.part(i), res.part(i));
                             case 64: return derived()->encode_arith_select_f64(std::move(cond_vpr), lhs_vr.part(i), rhs_vr.part(i), res.part(i));
                             default:
                                assert(0 && "Unsupported float type width for select operation");
                                return false;
                          }
                       })
                       .template Case<dialect::util::RefType, mlir::IndexType>([&](auto) {
                          return derived()->encode_arith_select_i64(std::move(cond_vpr), lhs_vr.part(i), rhs_vr.part(i), res.part(i));
                       })
                       .template Case<dialect::util::BufferType, dialect::util::VarLen32Type>([&](auto) {
                          const auto ret = derived()->encode_arith_select_i128(
                             std::move(cond_vpr),
                             lhs_vr.part(i), lhs_vr.part(i + 1),
                             rhs_vr.part(i), rhs_vr.part(i + 1),
                             res.part(i), res.part(i + 1));
                          i++;
                          return ret;
                       })
                       .Default([&](const auto t) {
                          t.dump();
                          assert(0 && "Unsupported type for select operation");
                          return false;
                       });
         i++;
      }
      return success;
   }

   bool compile_arith_index_cast_op(const mlir::arith::IndexCastOp op) {
      mlir::Value src = op->getOperand(0);
      mlir::Value res = op->getResult(0);
      if (mlir::isa<mlir::IntegerType>(src.getType()) && mlir::isa<mlir::IndexType>(res.getType())) {
         auto [_, src_vpr] = this->val_ref_single(src);
         auto res_ref = this->result_ref(res);
         if (src.getType().getIntOrFloatBitWidth() == 128) {
            res_ref.part(0).set_value(std::move(src_vpr));
            return true;
         }
         assert(src.getType().getIntOrFloatBitWidth() <= 64);
         if (src.getType().getIntOrFloatBitWidth() != 64)
            src_vpr = std::move(src_vpr).into_extended(false, src.getType().getIntOrFloatBitWidth(), 64);
         res_ref.part(0).set_value(std::move(src_vpr));
         return true;
      }
      if (mlir::isa<mlir::IndexType>(src.getType()) && mlir::isa<mlir::IntegerType>(res.getType())) {
         assert(res.getType().getIntOrFloatBitWidth() <= 64);
         auto [_, src_vpr] = this->val_ref_single(src);
         auto res_ref = this->result_ref(res);
         res_ref.part(0).set_value(std::move(src_vpr));
         return true;
      }
      assert(0 && "Index cast operation must be between integer and index types");
      return false;
   }

   bool compile_util_buffer_get_len_op(dialect::util::BufferGetLen op) {
      auto elem_size = get_size(op.getBuffer().getType().getT());
      if (!elem_size) {
         assert(0 && "Unsupported type for buffer length operation");
         error.emit() << "Unsupported type for buffer length operation.";
         return false;
      }
      // we store the length in the first 8 bytes of the buffer
      auto buf_vr = this->val_ref(op.getBuffer());
      auto res_vr = this->result_ref(op.getResult());
      ValuePartRef elem_size_vp{this, *elem_size, 8, Config::GP_BANK};
      derived()->encode_arith_udiv_i64(buf_vr.part(0), GenericValuePart{std::move(elem_size_vp)}, res_vr.part(0));
      return true;
   }

   bool compile_util_varlen_get_len_op(dialect::util::VarLenGetLen op) {
      auto varlen_vr = this->val_ref(op.getVarlen());
      auto res_vr = this->result_ref(op.getResult());
      res_vr.part(0).set_value(varlen_vr.part(0).into_extended(false, 32, 64));
      return true;
   }

   bool compile_util_const_varlen_op(const dialect::util::CreateConstVarLen op) {
      const mlir::StringRef content = mlir::cast<mlir::StringAttr>(op->getAttrs().front().getValue()).getValue();
      const size_t len = content.size();
      if (len <= 12) {
         // short strings are stored as constants in 128-bit and are therefore handled by val_ref_special
         return true;
      }
      mlir::Value res = op->getResult(0);
      ValueRef res_ref = this->result_ref(res);

      // part0 is always constant
      uint64_t first4 = 0;
      memcpy(&first4, content.data(), std::min(4ul, len));
      uint64_t c1 = (first4 << 32) | len;
      res_ref.part(0).set_value(ValuePartRef(this, c1, 8, Config::GP_BANK));

      // part1 is the pointer to the content
      // -> content is a StringRef pointing inside the mlir module
      // -> as long as we keep that alive past query execution, we can just hand out the pointer!
      res_ref.part(1).set_value(ValuePart{std::bit_cast<uint64_t>(content.data()), 8, Config::GP_BANK});
      return true;
   }

   bool compile_arith_trunci_op(mlir::arith::TruncIOp op) {
      auto src = op.getIn();
      auto dst = op.getOut();
      assert(
         mlir::isa<mlir::IntegerType>(src.getType()) &&
         "Source value must be an integer type for truncation");
      assert(
         mlir::isa<mlir::IntegerType>(dst.getType()) &&
         "Destination value must be an integer type for truncation");
      unsigned dest_width = dst.getType().getIntOrFloatBitWidth();
      if (dest_width <= 64) {
         auto src_vr = this->val_ref(src);
         auto [_, dst_vpr] = this->result_ref_single(dst);
         dst_vpr.set_value(src_vr.part(0));
         return true;
      }
      assert(false && "invalid truncation");
      return false;
   }

   bool compile_func_constant_op(mlir::func::ConstantOp op) {
      const auto funcName = op.getValue();
      auto res_vr = this->result_ref(op.getResult());
      if (localFuncMap.contains(funcName)) {
         uint32_t func_idx = localFuncMap[funcName];
         assert(func_idx < this->func_syms.size() && "Function index out of bounds");
         SymRef func_ref = this->func_syms[func_idx];
         derived()->load_address_of_got_sym(func_ref, res_vr.part(0).alloc_reg());
         return true;
      }
      if (externFuncMap.contains(funcName)) {
         res_vr.part(0).set_value(ValuePart{std::bit_cast<uint64_t>(externFuncMap[funcName]), 8, Config::GP_BANK});
         return true;
      }
      error.emit() << "Function constant refers to neither a local nor an external function: " << funcName.str() << "\n";
      return false;
   }

   bool compile_util_array_element_ptr_op(dialect::util::ArrayElementPtrOp op) {
      const mlir::TypedValue<dialect::util::RefType> array_ptr = op.getRef();
      const mlir::TypedValue<mlir::IndexType> idx = op.getIdx();
      const mlir::Type elem_type = array_ptr.getType().getElementType();

      assert(val_parts(array_ptr).count() == 1);
      auto [array_ptr_vr, array_ptr_pr] = this->val_ref_single(array_ptr);

      auto elem_size = get_size(elem_type);
      if (!elem_size) {
         assert(0 && "Unsupported type for store operation");
         error.emit() << "Unsupported type for store operation.";
         return false;
      }
      auto offset_expr = create_idx_offset_expr(std::move(array_ptr_pr), idx, elem_type);
      if (!offset_expr) {
         return false;
      }

      // load value to register (e.g. mov + add / lea for x86_64)
      AsmReg res_reg = derived()->gval_expr_as_reg(*offset_expr);
      ScratchReg res_scratch{derived()};
      derived()->mov(res_scratch.alloc_gp(), res_reg, 8);

      const auto dst = op.getRes();
      assert(val_parts(dst).count() == 1);
      auto [_, res_vr] = this->result_ref_single(dst);
      res_vr.set_value(std::move(res_scratch));
      return true;
   }

   bool compile_util_create_varlen_op(dialect::util::CreateVarLen op) {
      auto builder = derived()->create_call_builder();
      builder.add_arg(typename Base::CallArg{op.getRef()});
      builder.add_arg(typename Base::CallArg{op.getLen()});
      assert(externFuncMap.contains("createVarLen32"));
      ValuePart funcPtrRef{
         std::bit_cast<uint64_t>(externFuncMap["createVarLen32"]), 8, Config::GP_BANK};
      builder.call(std::move(funcPtrRef));
      ValueRef res = this->result_ref(op.getVarlen());
      builder.add_ret(res);
      return true;
   }

   bool compile_util_varlen_cmp_op(dialect::util::VarLenCmp op) {
      auto lhs_vr = this->val_ref(op.getLeft());
      auto rhs_vr = this->val_ref(op.getRight());
      auto eq_res_vr = this->result_ref(op.getEq());
      auto more_cmp_res_vr = this->result_ref(op.getNeedsDetailedEval());
      return derived()->encode_util_varlen_cmp(lhs_vr.part(0), lhs_vr.part(1), rhs_vr.part(0), rhs_vr.part(1), eq_res_vr.part(0), more_cmp_res_vr.part(0));
   }

   bool compile_arith_cmp_float_op(mlir::arith::CmpFOp op) {
      const mlir::Value lhs = op.getLhs();
      const mlir::Value rhs = op.getRhs();
      const mlir::Value res = op.getResult();
      const mlir::arith::CmpFPredicate pred = op.getPredicate();
      assert(lhs.getType() == rhs.getType() && "LHS and RHS must have the same type for float comparison");
      const mlir::Type cmp_type = lhs.getType();
      assert((cmp_type.isF32() || cmp_type.isF64()) && "Unsupported float type for comparison");

      using mlir::arith::CmpFPredicate;
      if (pred == CmpFPredicate::AlwaysFalse || pred == CmpFPredicate::AlwaysTrue) {
         uint64_t val = pred == CmpFPredicate::AlwaysFalse ? 0u : 1u;
         (void) this->val_ref(lhs); // ref-count
         (void) this->val_ref(rhs); // ref-count
         auto const_ref = ValuePartRef{this, val, 1, Config::GP_BANK};
         this->result_ref(res).part(0).set_value(std::move(const_ref));
         return true;
      }

      using EncodeFnTy = bool (Derived::*)(GenericValuePart&&, GenericValuePart&&, ValuePart&);
      EncodeFnTy fn = nullptr;
      if (cmp_type.isF32()) {
         switch (pred) {
            case CmpFPredicate::OEQ:
               fn = &Derived::encode_arith_cmp_f32_oeq;
               break;
            case CmpFPredicate::OLT:
               fn = &Derived::encode_arith_cmp_f32_olt;
               break;
            case CmpFPredicate::OLE:
               fn = &Derived::encode_arith_cmp_f32_ole;
               break;
            case CmpFPredicate::OGT:
               fn = &Derived::encode_arith_cmp_f32_ogt;
               break;
            case CmpFPredicate::OGE:
               fn = &Derived::encode_arith_cmp_f32_oge;
               break;
            case CmpFPredicate::ONE:
               fn = &Derived::encode_arith_cmp_f32_one;
               break;
            case CmpFPredicate::ORD:
               fn = &Derived::encode_arith_cmp_f32_ord;
               break;
            case CmpFPredicate::UNO:
               fn = &Derived::encode_arith_cmp_f32_uno;
               break;
            case CmpFPredicate::UEQ:
               fn = &Derived::encode_arith_cmp_f32_ueq;
               break;
            case CmpFPredicate::ULT:
               fn = &Derived::encode_arith_cmp_f32_ult;
               break;
            case CmpFPredicate::ULE:
               fn = &Derived::encode_arith_cmp_f32_ule;
               break;
            case CmpFPredicate::UGT:
               fn = &Derived::encode_arith_cmp_f32_ugt;
               break;
            case CmpFPredicate::UGE:
               fn = &Derived::encode_arith_cmp_f32_uge;
               break;
            case CmpFPredicate::UNE:
               fn = &Derived::encode_arith_cmp_f32_une;
               break;
            default: assert(0);
         }
      } else {
         switch (pred) {
            case CmpFPredicate::OEQ:
               fn = &Derived::encode_arith_cmp_f64_oeq;
               break;
            case CmpFPredicate::OLT:
               fn = &Derived::encode_arith_cmp_f64_olt;
               break;
            case CmpFPredicate::OLE:
               fn = &Derived::encode_arith_cmp_f64_ole;
               break;
            case CmpFPredicate::OGT:
               fn = &Derived::encode_arith_cmp_f64_ogt;
               break;
            case CmpFPredicate::OGE:
               fn = &Derived::encode_arith_cmp_f64_oge;
               break;
            case CmpFPredicate::ONE:
               fn = &Derived::encode_arith_cmp_f64_one;
               break;
            case CmpFPredicate::ORD:
               fn = &Derived::encode_arith_cmp_f64_ord;
               break;
            case CmpFPredicate::UNO:
               fn = &Derived::encode_arith_cmp_f64_uno;
               break;
            case CmpFPredicate::UEQ:
               fn = &Derived::encode_arith_cmp_f64_ueq;
               break;
            case CmpFPredicate::ULT:
               fn = &Derived::encode_arith_cmp_f64_ult;
               break;
            case CmpFPredicate::ULE:
               fn = &Derived::encode_arith_cmp_f64_ule;
               break;
            case CmpFPredicate::UGT:
               fn = &Derived::encode_arith_cmp_f64_ugt;
               break;
            case CmpFPredicate::UGE:
               fn = &Derived::encode_arith_cmp_f64_uge;
               break;
            case CmpFPredicate::UNE:
               fn = &Derived::encode_arith_cmp_f64_une;
               break;
            default: assert(0);
         }
      }
      ValueRef lhs_vr = this->val_ref(lhs);
      ValueRef rhs_vr = this->val_ref(rhs);
      auto [res_vr, res_ref] = this->result_ref_single(res);
      if (!(derived()->*fn)(lhs_vr.part(0), rhs_vr.part(0), res_ref)) { return false; }
      return true;
   }

   bool compile_util_hash_64_op(dialect::util::Hash64 op) {
      auto [_, val_pr] = this->val_ref_single(op.getVal());
      auto res_pr = this->result_ref(op.getResult());
      return derived()->encode_util_hash_64(std::move(val_pr), res_pr.part(0));
   }

   bool compile_util_ptr_tag_matches_op(dialect::util::PtrTagMatches op) {
      auto hash_vr = this->val_ref(op.getHash());
      auto ref_vr = this->val_ref(op.getRef());

      assert(externFuncMap.contains("bloomMasks"));
      ValuePartRef bloom_masks_array_vpr{
         this, std::bit_cast<uint64_t>(externFuncMap["bloomMasks"]), 8, Config::GP_BANK};
      auto res_ref = this->result_ref(op.getMatches());
      return derived()->encode_util_ptr_tag_matches(ref_vr.part(0), hash_vr.part(0), std::move(bloom_masks_array_vpr), res_ref.part(0));
   }

   bool compile_util_untag_ptr_op(dialect::util::UnTagPtr op) {
      auto ref_vr = this->val_ref(op.getRef());
      auto res_ref = this->result_ref(op.getRes());
      return derived()->encode_util_untag_ptr(ref_vr.part(0), res_ref.part(0));
   }

   bool compile_util_is_ref_valid_op(dialect::util::IsRefValidOp op) {
      auto ref_vr = this->val_ref(op.getRef());
      auto res_vr = this->result_ref(op.getValid());
      return derived()->encode_util_is_ref_valid(ref_vr.part(0), res_vr.part(0));
   }

   bool compile_util_varlen_try_cheap_hash_op(dialect::util::VarLenTryCheapHash op) {
      auto varlen_vr = this->val_ref(op.getVarlen());
      auto res_complete_vr = this->result_ref(op.getComplete());
      auto res_hash_vr = this->result_ref(op.getHash());
      return derived()->encode_util_varlen_try_cheap_hash(varlen_vr.part(0), varlen_vr.part(1), res_complete_vr.part(0), res_hash_vr.part(0));
   }

   bool compile_util_hash_varlen_op(dialect::util::HashVarLen op) {
      auto builder = derived()->create_call_builder();
      builder.add_arg(typename Base::CallArg{op.getVal()});
      assert(externFuncMap.contains("hashVarLenData"));
      ValuePart funcPtrRef{
         std::bit_cast<uint64_t>(externFuncMap["hashVarLenData"]), 8, Config::GP_BANK};
      builder.call(std::move(funcPtrRef));
      ValueRef res = this->result_ref(op.getHash());
      builder.add_ret(res);
      return true;
   }

   bool compile_util_invalid_ref_op(dialect::util::InvalidRefOp op) {
      auto res_ref = this->result_ref(op.getResult());
      ValuePartRef null_ref{this, 0, 8, Config::GP_BANK};
      res_ref.part(0).set_value(std::move(null_ref));
      return true;
   }

   bool compile_util_hash_combine_op(dialect::util::HashCombine op) {
      auto h1_vr = this->val_ref(op.getH1());
      auto h2_vr = this->val_ref(op.getH2());
      auto res_ref = this->result_ref(op.getResult());
      return derived()->encode_util_hash_combine(h1_vr.part(0), h2_vr.part(0), res_ref.part(0));
   }

   bool compile_util_buffer_get_ref_op(dialect::util::BufferGetRef op) {
      const mlir::TypedValue<dialect::util::BufferType> buf = op.getBuffer();

      const auto dst = op->getResult(0);
      assert(val_parts(buf).count() == 2);
      assert(val_parts(dst).count() == 1);
      auto buf_vr = this->val_ref(buf);

      auto [_, res_vr] = this->result_ref_single(dst);
      res_vr.set_value(buf_vr.part(1));
      return true;
   }

   bool compile_arith_sitofp_op(const auto op, const bool sign) {
      mlir::Value src = op->getOperand(0);
      mlir::Value res = op->getResult(0);
      assert(mlir::isa<mlir::IntegerType>(src.getType()) && "Source value must be an integer type for sitofp");
      assert(mlir::isa<mlir::FloatType>(res.getType()) && "Result value must be a float type for sitofp");
      unsigned src_width = src.getType().getIntOrFloatBitWidth();
      unsigned dst_width = res.getType().getIntOrFloatBitWidth();
      assert((src_width <= 64 || src_width == 128) && (dst_width == 32 || dst_width == 64) &&
             "Source width must be less than or equal to 64 bits and destination width must be 32 or 64 bits");

      if (src_width == 128) {
         auto builder = derived()->create_call_builder();
         builder.add_arg(typename Base::CallArg{src});
         std::string_view func_name;
         if (dst_width == 32) {
            func_name = "sitofpI128F32";
         } else {
            func_name = "sitofpI128F64";
         }
         assert(externFuncMap.contains(func_name));
         ValuePart funcPtrRef{
            std::bit_cast<uint64_t>(externFuncMap[func_name]), 8, Config::GP_BANK};

         builder.call(std::move(funcPtrRef));
         ValueRef res_vr = this->result_ref(res);
         builder.add_ret(res_vr);
         return true;
      }

      auto [_, src_vpr] = this->val_ref_single(src);
      auto res_ref = this->result_ref(res);
      bool ret = false;

      if (src_width < 64) {
         src_vpr = std::move(src_vpr).into_extended(true, src_width, 64);
      }
      if (sign) {
         if (dst_width == 32) {
            ret = derived()->encode_arith_sitofp_i64_f32(std::move(src_vpr), res_ref.part(0));
         } else {
            ret = derived()->encode_arith_sitofp_i64_f64(std::move(src_vpr), res_ref.part(0));
         }
      } else {
         if (dst_width == 32) {
            ret = derived()->encode_arith_uitofp_i64_f32(std::move(src_vpr), res_ref.part(0));
         } else {
            ret = derived()->encode_arith_uitofp_i64_f64(std::move(src_vpr), res_ref.part(0));
         }
      }
      return ret;
   }

   bool compile_util_pack_op(dialect::util::PackOp op) {
      mlir::TypedValue<mlir::Type> res = op.getTuple();
      auto vals = op.getVals();

      const mlir::TupleType tuple_type = mlir::cast<mlir::TupleType>(res.getType());
      assert(tuple_type.getTypes().size() == vals.size() && "Number of values must match number of tuple elements");

      auto res_vr = this->result_ref(res);
      unsigned offset = 0;
      for (size_t i = 0; i < vals.size(); ++i) {
         auto val_vr = this->val_ref(vals[i]);
         auto val_type = vals[i].getType();
         mlir::TypeSwitch<mlir::Type>(val_type)
            .Case<mlir::TupleType>([](const mlir::TupleType) {
               ;
               assert(0 && "Nested tuples are not supported");
               return;
            })
            .template Case<mlir::IntegerType>([&](const mlir::IntegerType int_type) {
               res_vr.part(offset++).set_value(val_vr.part(0));
               if (int_type.getWidth() == 128) {
                  res_vr.part(offset++).set_value(val_vr.part(1));
               }
            })
            .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
               res_vr.part(offset++).set_value(val_vr.part(0));
               res_vr.part(offset++).set_value(val_vr.part(1));
            })
            .Default([&](const mlir::Type) {
               res_vr.part(offset++).set_value(val_vr.part(0));
            });
      }
      return true;
   }

   bool compile_util_get_tuple_op(dialect::util::GetTupleOp op) {
      mlir::TypedValue<mlir::Type> val = op.getTuple();
      mlir::TypedValue<mlir::Type> res = op.getVal();

      auto val_vr = this->val_ref(val);
      auto res_vr = this->result_ref(res);

      auto tuple_type = mlir::cast<mlir::TupleType>(val.getType());

      unsigned offset = 0;
      for (uint32_t i = 0; i < op.getOffset(); ++i) {
         offset += mlir::TypeSwitch<mlir::Type, unsigned>(tuple_type.getType(i))
                      .Case<mlir::TupleType>([](const mlir::TupleType t) {
                         return TupleHelper::numSlots(t);
                      })
                      .template Case<mlir::IntegerType>([](const mlir::IntegerType int_type) {
                         return int_type.getWidth() == 128 ? 2u : 1u;
                      })
                      .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) {
                         return 2u;
                      })
                      .Default([](const mlir::Type) {
                         return 1u;
                      });
      }

      mlir::TypeSwitch<mlir::Type>(TupleHelper::typeAtSlot(tuple_type, offset))
         .template Case<mlir::TupleType>([&](const mlir::TupleType t) {
            for (size_t i = 0; i < TupleHelper::numSlots(t); ++i) {
               res_vr.part(i).set_value(val_vr.part(offset + i));
            }
         })
         .template Case<mlir::IntegerType>([&](const mlir::IntegerType int_type) {
            res_vr.part(0).set_value(val_vr.part(offset));
            if (int_type.getWidth() == 128) {
               res_vr.part(1).set_value(val_vr.part(offset + 1));
            }
         })
         .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
            res_vr.part(0).set_value(val_vr.part(offset));
            res_vr.part(1).set_value(val_vr.part(offset + 1));
         })
         .Default([&](const mlir::Type) {
            res_vr.part(0).set_value(val_vr.part(offset));
         });
      return true;
   }

   bool compile_inst(const IRInstRef inst, InstRange) noexcept {
      return mlir::TypeSwitch<IRInstRef, bool>(inst)
         .Case<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp, mlir::arith::DivSIOp,
               mlir::arith::DivUIOp, mlir::arith::RemSIOp, mlir::arith::RemUIOp,
               mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp, mlir::arith::ShLIOp,
               mlir::arith::ShRUIOp, mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
               mlir::arith::DivFOp, mlir::arith::MaxSIOp, mlir::arith::MinSIOp, mlir::arith::MaxUIOp,
               mlir::arith::MinUIOp>([&](auto op) {
            return compile_arith_binary_op(op);
         })
         .template Case<mlir::arith::CmpIOp>(
            [&](auto op) { return derived()->compile_arith_cmp_int_op(op); })
         .template Case<mlir::arith::CmpFOp>([&](auto op) { return compile_arith_cmp_float_op(op); })
         .template Case<mlir::cf::BranchOp>([&](auto op) { return compile_cf_br_op(op); })
         .template Case<mlir::cf::CondBranchOp>(
            [&](auto op) { return derived()->compile_cf_cond_br_op(op); })
         .template Case<mlir::arith::ConstantOp, dialect::util::SizeOfOp, dialect::util::AllocaOp,
                        dialect::util::UndefOp>(
            [&](auto) {
               // these are all constant operations whose value is handled in val_ref_special / val_part_ref_special
               return true;
            })
         .template Case<mlir::func::ConstantOp>([&](auto op) { return compile_func_constant_op(op); })
         .template Case<dialect::util::GenericMemrefCastOp>([&](auto op) {
            return compile_util_generic_memref_cast_op(op);
         })
         .template Case<dialect::util::TupleElementPtrOp>([&](auto op) {
            return compile_util_tuple_element_ptr_op(op);
         })
         .template Case<dialect::util::LoadOp>([&](auto op) { return compile_util_load_op(op); })
         .template Case<dialect::util::StoreOp>([&](auto op) { return compile_util_store_op(op); })
         .template Case<mlir::func::CallOp>([&](auto op) { return compile_func_call_op(op); })
         .template Case<mlir::func::ReturnOp>([&](auto op) { return compile_func_return_op(op); })
         .template Case<mlir::arith::ExtUIOp>([&](auto op) { return compile_arith_exti_op(op, false); })
         .template Case<mlir::arith::ExtSIOp>([&](auto op) { return compile_arith_exti_op(op, true); })
         .template Case<mlir::arith::ExtFOp>([&](auto op) { return compile_arith_extf_op(op); })
         .template Case<mlir::arith::SelectOp>([&](auto op) { return compile_arith_select_op(op); })
         .template Case<mlir::arith::IndexCastOp>([&](auto op) { return compile_arith_index_cast_op(op); })
         .template Case<dialect::util::CreateConstVarLen>([&](auto op) {
            return compile_util_const_varlen_op(op);
         })
         .template Case<dialect::util::BufferCastOp>(
            [&](auto op) { return compile_util_buffer_cast_op(op); })
         .template Case<dialect::util::BufferGetLen>([&](auto op) {
            return compile_util_buffer_get_len_op(op);
         })
         .template Case<dialect::util::BufferGetElementRef>([&](auto op) {
            return compile_util_buffer_get_element_ref_op(op);
         })
         .template Case<mlir::arith::TruncIOp>([&](auto op) { return compile_arith_trunci_op(op); })
         .template Case<dialect::util::ArrayElementPtrOp>([&](auto op) {
            return compile_util_array_element_ptr_op(op);
         })
         .template Case<dialect::util::CreateVarLen>([&](auto op) {
            return compile_util_create_varlen_op(op);
         })
         .template Case<dialect::util::Hash64>([&](auto op) {
            return compile_util_hash_64_op(op);
         })
         .template Case<dialect::util::PtrTagMatches>([&](auto op) {
            return compile_util_ptr_tag_matches_op(op);
         })
         .template Case<dialect::util::UnTagPtr>([&](auto op) {
            return compile_util_untag_ptr_op(op);
         })
         .template Case<dialect::util::IsRefValidOp>([&](auto op) {
            return compile_util_is_ref_valid_op(op);
         })
         .template Case<dialect::util::InvalidRefOp>([&](auto op) {
            return compile_util_invalid_ref_op(op);
         })
         .template Case<dialect::util::HashCombine>([&](auto op) {
            return compile_util_hash_combine_op(op);
         })
         .template Case<dialect::util::VarLenGetLen>([&](auto op) {
            return compile_util_varlen_get_len_op(op);
         })
         .template Case<dialect::util::VarLenTryCheapHash>([&](auto op) {
            return compile_util_varlen_try_cheap_hash_op(op);
         })
         .template Case<dialect::util::HashVarLen>([&](auto op) {
            return compile_util_hash_varlen_op(op);
         })
         .template Case<dialect::util::BufferGetRef>([&](auto op) {
            return compile_util_buffer_get_ref_op(op);
         })
         .template Case<mlir::arith::SIToFPOp>([&](auto op) {
            return compile_arith_sitofp_op(op, true);
         })
         .template Case<mlir::arith::UIToFPOp>([&](auto op) {
            return compile_arith_sitofp_op(op, false);
         })
         .template Case<dialect::util::PackOp>([&](auto op) {
            return compile_util_pack_op(op);
         })
         .template Case<dialect::util::GetTupleOp>([&](auto op) {
            return compile_util_get_tuple_op(op);
         })
         .template Case<dialect::util::VarLenCmp>([&](auto op) { return compile_util_varlen_cmp_op(op); })
         .Default([&](IRInstRef op) {
            error.emit() << "Encountered unimplemented instruction: " << op->getName().getStringRef().str()
                         << "\n";
            return false;
         });
   }
};

// NOLINTEND(readability-identifier-naming)
} // namespace lingodb::execution::baseline
