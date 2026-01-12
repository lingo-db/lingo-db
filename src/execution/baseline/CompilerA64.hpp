#pragma once

#include "Adaptor.hpp"
#include "CompilerBase.hpp"
#include "CompilerConfig.hpp"
#include "snippet_encoders_arm.hpp"

#include <tpde/ValueRef.hpp>
#include <tpde/arm64/CompilerA64.hpp>

namespace lingodb::execution::baseline {
// NOLINTBEGIN(readability-identifier-naming)

// aarch64 target specific compiler
// based on https://github.com/tpde2/tpde/blob/5df6e8af796351971b4d44add6ad3c4a7b5532d2/tpde-llvm/src/arm64/LLVMCompilerArm64.cpp
struct IRCompilerA64
   : tpde::a64::CompilerA64<IRAdaptor, IRCompilerA64, IRCompilerBase, CompilerConfig>,
     tpde_encodegen::EncodeCompiler<IRAdaptor, IRCompilerA64, IRCompilerBase, CompilerConfig> {
   using Base = tpde::a64::CompilerA64<IRAdaptor, IRCompilerA64, IRCompilerBase, CompilerConfig>;
   using ScratchReg = typename Base::ScratchReg;
   using ValuePartRef = typename Base::ValuePartRef;
   using ValuePart = typename Base::ValuePart;
   using ValueRef = typename Base::ValueRef;
   using SymRef = tpde::SymRef;
   using AsmReg = typename Base::AsmReg;
   using GenericValuePart = typename Base::GenericValuePart;

   std::unique_ptr<IRAdaptor> adaptor;

   // since this needs to life past the create_call_builder function call, we store it here
   std::variant<std::monostate, tpde::a64::CCAssignerAAPCS> cc_assigners;

   explicit IRCompilerA64(std::unique_ptr<IRAdaptor>&& adaptor)
      : Base{adaptor.get()},
        adaptor(std::move(adaptor)) { static_assert(tpde::Compiler<IRCompilerA64, tpde::a64::PlatformConfig>); }

   void load_address_of_global(const SymRef global_sym, const AsmReg dst) {
      // emit lea with relocation
      reloc_text(
         global_sym, tpde::elf::R_AARCH64_ADR_PREL_PG_HI21, this->text_writer.offset());
      ASMNC(ADRP, dst, 0, 0);
      reloc_text(
         global_sym, tpde::elf::R_AARCH64_ADD_ABS_LO12_NC, this->text_writer.offset());
      ASMNC(ADDxi, dst, dst, 0);
   }

   void create_helper_call(std::span<IRValueRef> args, ValueRef* result, SymRef sym) noexcept {
      tpde::util::SmallVector<CallArg, 8> arg_vec{};
      for (const auto arg : args) { arg_vec.push_back(CallArg{arg}); }
      generate_call(sym, arg_vec, result);
   }

   bool compile_arith_cmp_int_op(mlir::arith::CmpIOp op) {
      mlir::Type ty = op.getLhs().getType();
      unsigned int_width;
      if (mlir::isa<mlir::IntegerType>(ty)) {
         int_width = ty.getIntOrFloatBitWidth();
      } else if (mlir::isa<mlir::IndexType>(ty)) {
         int_width = 64;
      } else {
         assert(0 && "Unsupported type for comparison operation");
      }
      Jump::Kind jump;

      bool is_signed = false;
      switch (op.getPredicate()) {
         case mlir::arith::CmpIPredicate::eq:
            jump = Jump::Jeq;
            break;
         case mlir::arith::CmpIPredicate::ne:
            jump = Jump::Jne;
            break;
         case mlir::arith::CmpIPredicate::ugt:
            jump = Jump::Jhi;
            break;
         case mlir::arith::CmpIPredicate::uge:
            jump = Jump::Jhs;
            break;
         case mlir::arith::CmpIPredicate::ult:
            jump = Jump::Jlo;
            break;
         case mlir::arith::CmpIPredicate::ule:
            jump = Jump::Jls;
            break;
         case mlir::arith::CmpIPredicate::sgt:
            jump = Jump::Jgt;
            break;
         case mlir::arith::CmpIPredicate::sge:
            jump = Jump::Jge;
            break;
         case mlir::arith::CmpIPredicate::slt:
            jump = Jump::Jlt;
            break;
         case mlir::arith::CmpIPredicate::sle:
            jump = Jump::Jle;
            break;
         default:
            assert(0);
            return false;
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

      auto lhs = this->val_ref(op.getLhs());
      auto rhs = this->val_ref(op.getRhs());
      ScratchReg res_scratch{this};

      if (int_width == 128) {
         auto lhs_lo = lhs.part(0);
         auto lhs_hi = lhs.part(1);
         auto lhs_reg_lo = lhs_lo.load_to_reg();
         auto lhs_reg_hi = lhs_hi.load_to_reg();

         auto rhs_lo = rhs.part(0);
         auto rhs_hi = rhs.part(1);
         auto rhs_reg_lo = rhs_lo.load_to_reg();
         auto rhs_reg_hi = rhs_hi.load_to_reg();

         if (jump == Jump::Jeq || jump == Jump::Jne) {
            // Use CCMP for equality
            ASM(CMPx, lhs_reg_lo, rhs_reg_lo);
            ASM(CCMPx, lhs_reg_hi, rhs_reg_hi, 0, DA_EQ);
         } else if (jump == Jump::Jhi || jump == Jump::Jls || jump == Jump::Jle ||
                    jump == Jump::Jgt) {
            // gt and le need inverse operand order for comparison.
            jump = swap_jump(jump).kind;
            ASM(CMPx, rhs_reg_lo, lhs_reg_lo);
            ASM(SBCSx, DA_ZR, rhs_reg_hi, lhs_reg_hi);
         } else {
            ASM(CMPx, lhs_reg_lo, rhs_reg_lo);
            ASM(SBCSx, DA_ZR, lhs_reg_hi, rhs_reg_hi);
         }
      } else {
         ValuePartRef lhs_pr = lhs.part(0);
         ValuePartRef rhs_pr = rhs.part(0);

         if (lhs_pr.is_const() && !rhs_pr.is_const()) {
            std::swap(lhs_pr, rhs_pr);
            jump = swap_jump(jump).kind;
         }

         if (int_width != 32 && int_width != 64) {
            unsigned ext_bits = tpde::util::align_up(int_width, 32);
            lhs_pr = std::move(lhs_pr).into_extended(is_signed, int_width, ext_bits);
            rhs_pr = std::move(rhs_pr).into_extended(is_signed, int_width, ext_bits);
         }

         AsmReg lhs_reg = lhs_pr.has_reg() ? lhs_pr.cur_reg() : lhs_pr.load_to_reg();
         if (rhs_pr.is_const()) {
            uint64_t imm = rhs_pr.const_data()[0];
            switch (int_width) {
               case 1:
               case 8:
               case 16:
               case 32:
                  if (!ASMIF(CMPwi, lhs_reg, imm)) {
                     AsmReg rhs_reg = rhs_pr.has_reg() ? rhs_pr.cur_reg() : rhs_pr.load_to_reg();
                     ASM(CMPw, lhs_reg, rhs_reg);
                  }
                  break;
               case 64: {
                  AsmReg rhs_reg = rhs_pr.has_reg() ? rhs_pr.cur_reg() : rhs_pr.load_to_reg();
                  ASM(CMPx, lhs_reg, rhs_reg);
               } break;
               default:
                  assert(0);
                  return false;
            }
         } else {
            auto rhs_reg = rhs_pr.load_to_reg();
            switch (int_width) {
               case 1:
               case 8:
               case 16:
               case 32:
                  ASM(CMPw, lhs_reg, rhs_reg);
                  break;
               case 64:
                  ASM(CMPx, lhs_reg, rhs_reg);
                  break;
               default:
                  assert(0);
                  return false;
            }
         }
      }
      lhs.reset();
      rhs.reset();

      const mlir::Operation* next = op->getNextNode();
      if (auto br = mlir::dyn_cast_or_null<mlir::cf::CondBranchOp>(next);
          br && br.getCondition() == op.getResult()) {
         // br follows cmp immediately -> only materialize flags into register if cmp has multiple users
         if (!op->hasOneUse()) {
            (void) result_ref(op.getResult());
            generate_raw_set(jump, result_ref(op.getResult()).part(0).alloc_reg());
         }
         auto* true_block = br.getTrueDest();
         auto* false_block = br.getFalseDest();
         generate_cond_branch(jump, true_block, false_block);
         this->adaptor->inst_set_fused(br, true);
         return true;
      }

      if (op->hasOneUse() && *op->user_begin() == next) {
         if (auto zext_op = mlir::dyn_cast_or_null<mlir::arith::ExtUIOp>(next); zext_op && zext_op.getResult().getType().getIntOrFloatBitWidth() < 64) {
            // chain: cmp -> zext => immediately generate the zext set into target register
            auto [_, res_ref] = result_ref_single(zext_op.getResult());
            generate_raw_set(jump, res_ref.alloc_reg());
            this->adaptor->inst_set_fused(zext_op, true);
            return true;
         }
         if (auto sext_op = mlir::dyn_cast_or_null<mlir::arith::ExtSIOp>(next); sext_op && sext_op.getResult().getType().getIntOrFloatBitWidth() < 64) {
            // chain: cmp -> sext => immediately generate the sext set into target register
            auto [_, res_ref] = result_ref_single(sext_op.getResult());
            generate_raw_mask(jump, res_ref.alloc_reg());
            this->adaptor->inst_set_fused(sext_op, true);
            return true;
         }
      }

      // no fusion possible, just generate the set instruction
      auto [_, res_ref] = result_ref_single(op);
      generate_raw_set(jump, res_scratch.alloc_gp());
      res_ref.set_value(std::move(res_scratch));
      return true;
   }
   bool compile_cf_cond_br_op(mlir::cf::CondBranchOp op) {
      auto* const true_block = op.getTrueDest();
      auto* const false_block = op.getFalseDest();

      auto [_, cond_ref] = this->val_ref_single(op.getCondition());
      const auto cond_reg = cond_ref.load_to_reg();
      ASM(TSTwi, cond_reg, 1);

      generate_cond_branch(Jump::Jne, true_block, false_block);
      return true;
   }

   void load_address_of_got_sym(const SymRef sym, const AsmReg dst) noexcept {
      assert(sym.valid());
      // mov the ptr from the GOT
      reloc_text(
         sym, tpde::elf::R_AARCH64_ADR_GOT_PAGE, this->text_writer.offset());
      ASMNC(ADRP, dst, 0, 0);
      reloc_text(
         sym, tpde::elf::R_AARCH64_LD64_GOT_LO12_NC, this->text_writer.offset());
      ASMNC(LDRxu, dst, dst, 0);
   }

   void reset() noexcept {
      Base::reset();
      EncodeCompiler::reset();
   }

   Error& getError() { return Base::getError(); }

   CallBuilder create_call_builder() {
      cc_assigners = tpde::a64::CCAssignerAAPCS();
      return CallBuilder{*this, std::get<tpde::a64::CCAssignerAAPCS>(cc_assigners)};
   }
};

// NOLINTEND(readability-identifier-naming)
}