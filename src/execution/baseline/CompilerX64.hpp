#pragma once

#include "Adaptor.hpp"
#include "CompilerBase.hpp"
#include "CompilerConfig.hpp"
#include "snippet_encoders_x64.hpp"

#include <tpde/ValueRef.hpp>
#include <tpde/x64/CompilerX64.hpp>

namespace lingodb::execution::baseline {
// NOLINTBEGIN(readability-identifier-naming)

// x86_64 target specific compiler
struct IRCompilerX64
   : tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>,
     tpde_encodegen::EncodeCompiler<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig> {
   using Base = tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;
   using ScratchReg = typename Base::ScratchReg;
   using ValuePartRef = typename Base::ValuePartRef;
   using ValuePart = typename Base::ValuePart;
   using ValueRef = typename Base::ValueRef;
   using Assembler = typename Base::Assembler;
   using SymRef = tpde::SymRef;
   using AsmReg = typename Base::AsmReg;
   using GenericValuePart = typename Base::GenericValuePart;

   std::unique_ptr<IRAdaptor> adaptor;

   // since this needs to life past the create_call_builder function call, we store it here
   std::variant<std::monostate, tpde::x64::CCAssignerSysV> cc_assigners;

   explicit IRCompilerX64(std::unique_ptr<IRAdaptor>&& adaptor)
      : Base{adaptor.get()},
        adaptor(std::move(adaptor)) { static_assert(tpde::Compiler<IRCompilerX64, tpde::x64::PlatformConfig>); }

   void load_address_of_global(const SymRef global_sym, const AsmReg dst) {
      ASM(MOV64rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
      reloc_text(global_sym, R_X86_64_PC32, text_writer.offset() - 4, -4);
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
      } else if (mlir::isa<
                    mlir::IndexType>(ty)) {
         // index type is always 64-bit on x86_64
         int_width = 64;
      } else {
         assert(0 && "Unsupported type for comparison operation");
         return false;
      }
      Jump jump;

      bool is_signed = false;
      switch (op.getPredicate()) {
         case mlir::arith::CmpIPredicate::eq:
            jump = Jump::je;
            break;
         case mlir::arith::CmpIPredicate::ne:
            jump = Jump::jne;
            break;
         case mlir::arith::CmpIPredicate::ugt:
            jump = Jump::ja;
            break;
         case mlir::arith::CmpIPredicate::uge:
            jump = Jump::jae;
            break;
         case mlir::arith::CmpIPredicate::ult:
            jump = Jump::jb;
            break;
         case mlir::arith::CmpIPredicate::ule:
            jump = Jump::jbe;
            break;
         case mlir::arith::CmpIPredicate::sgt:
            jump = Jump::jg;
            break;
         case mlir::arith::CmpIPredicate::sge:
            jump = Jump::jge;
            break;
         case mlir::arith::CmpIPredicate::slt:
            jump = Jump::jl;
            break;
         case mlir::arith::CmpIPredicate::sle:
            jump = Jump::jle;
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
         if ((jump == Jump::ja) || (jump == Jump::jbe) || (jump == Jump::jle) || (jump == Jump::jg)) {
            std::swap(lhs, rhs);
            jump = swap_jump(jump);
         }

         auto rhs_lo = rhs.part(0);
         auto rhs_hi = rhs.part(1);
         auto rhs_reg_lo = rhs_lo.load_to_reg();
         auto rhs_reg_hi = rhs_hi.load_to_reg();

         // Compare the ints using carried subtraction
         if ((jump == Jump::je) || (jump == Jump::jne)) {
            // for eq,neq do something a bit quicker
            ScratchReg scratch{this};
            lhs.part(0).reload_into_specific_fixed(this, res_scratch.alloc_gp());
            lhs.part(1).reload_into_specific_fixed(this, scratch.alloc_gp());

            ASM(XOR64rr, res_scratch.cur_reg(), rhs_reg_lo);
            ASM(XOR64rr, scratch.cur_reg(), rhs_reg_hi);
            ASM(OR64rr, res_scratch.cur_reg(), scratch.cur_reg());
         } else {
            auto lhs_lo = lhs.part(0);
            auto lhs_reg_lo = lhs_lo.load_to_reg();
            auto lhs_high_tmp =
               lhs.part(1).reload_into_specific_fixed(this, res_scratch.alloc_gp());

            ASM(CMP64rr, lhs_reg_lo, rhs_reg_lo);
            ASM(SBB64rr, lhs_high_tmp, rhs_reg_hi);
         }
      } else {
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
               case 1:
               case 8:
               case 16:
               case 32:
                  ASM(CMP32ri, lhs_reg, static_cast<uint32_t>(rhs_const));
                  break;
               case 64:
                  // test if the constant fits into a signed 32-bit integer
                  if (static_cast<int32_t>(rhs_const) == static_cast<int64_t>(rhs_const)) {
                     ASM(CMP64ri, lhs_reg, static_cast<int32_t>(rhs_const));
                  } else {
                     // ScratchReg scratch3{this};
                     // auto tmp = scratch3.alloc_gp();
                     // ASM(MOV64ri, tmp, rhs_const);
                     // ASM(CMP64rr, lhs_reg, tmp);
                     AsmReg rhs_reg = rhs_pr.has_reg() ? rhs_pr.cur_reg() : rhs_pr.load_to_reg();
                     ASM(CMP64rr, lhs_reg, rhs_reg);
                  }
                  break;
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
                  ASM(CMP32rr, lhs_reg, rhs_reg);
                  break;
               case 64:
                  ASM(CMP64rr, lhs_reg, rhs_reg);
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
            generate_raw_set(jump, result_ref(op.getResult()).part(0).alloc_reg(), /*zext=*/false);
         }
         auto* true_block = br.getTrueDest();
         auto* false_block = br.getFalseDest();
         generate_conditional_branch(jump, true_block, false_block);
         this->adaptor->inst_set_fused(br, true);
         return true;
      }

      if (op->hasOneUse() && *op->user_begin() == next) {
         if (auto zext_op = mlir::dyn_cast_or_null<mlir::arith::ExtUIOp>(next); zext_op && zext_op.getResult().getType().getIntOrFloatBitWidth() < 64) {
            // chain: cmp -> zext => immediately generate the zext set into target register
            auto [_, res_ref] = result_ref_single(zext_op.getResult());
            generate_raw_set(jump, res_ref.alloc_reg(), /*zext=*/true);
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
      ASM(TEST32ri, cond_reg, 1);

      return generate_conditional_branch(Jump::jne, true_block, false_block);
   }

   bool generate_conditional_branch(const Jump jmp, const IRBlockRef true_block, const IRBlockRef false_block) noexcept {
      auto* const next_block = this->analyzer.block_ref(this->next_block());

      const auto true_needs_split = this->branch_needs_split(true_block);
      const auto false_needs_split = this->branch_needs_split(false_block);

      const auto spilled = this->spill_before_branch();
      this->begin_branch_region();

      if (next_block == true_block || (next_block != false_block && true_needs_split)) {
         generate_branch_to_block(invert_jump(jmp), false_block, false_needs_split, false);
         generate_branch_to_block(Jump::jmp, true_block, false, true);
      } else if (next_block == false_block) {
         generate_branch_to_block(jmp, true_block, true_needs_split, false);
         generate_branch_to_block(Jump::jmp, false_block, false, true);
      } else {
         assert(!true_needs_split);
         this->generate_branch_to_block(jmp, true_block, false, false);
         this->generate_branch_to_block(Jump::jmp, false_block, false, true);
      }
      this->end_branch_region();
      this->release_spilled_regs(spilled);
      return true;
   }

   void load_address_of_got_sym(const SymRef sym, const AsmReg dst) noexcept {
      assert(sym.valid());
      ASM(MOV64rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
      reloc_text(sym, R_X86_64_GOTPCREL, text_writer.offset() - 4, -4);
   }

   void reset() noexcept {
      Base::reset();
      EncodeCompiler::reset();
   }

   Error& getError() { return Base::getError(); }

   CallBuilder create_call_builder() {
      cc_assigners = tpde::x64::CCAssignerSysV(false);
      return CallBuilder{*this, std::get<tpde::x64::CCAssignerSysV>(cc_assigners)};
   }
};

// NOLINTEND(readability-identifier-naming)
}
