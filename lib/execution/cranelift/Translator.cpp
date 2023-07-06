#include "ccranelift.h"

#include "mlir/Dialect/cranelift/CraneliftDialect.h"
#include "mlir/Dialect/cranelift/CraneliftExecutionEngine.h"
#include "mlir/Dialect/cranelift/CraneliftOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include <chrono>
#include <iostream>

mlir::cranelift::CraneliftExecutionEngine::CraneliftExecutionEngine(mlir::ModuleOp module) : moduleOp(module) {
   mod = cranelift_module_new(
      "x86_64-pc-linux", "enable_simd,enable_atomics,enable_llvm_abi_extensions", "mymodule", 0,
      [](uintptr_t userdata, const char* err, const char* fn) {
         printf("Error %s: %s\n", fn, err);
      },
      [](uintptr_t userdata, const char* err, const char* fn) { printf("Message %s: %s\n", fn, err); });
   auto start = std::chrono::high_resolution_clock::now();

   translate(module);
   auto end = std::chrono::high_resolution_clock::now();
   cranelift_compile(mod);
   jitTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
uint8_t mlir::cranelift::CraneliftExecutionEngine::translateType(mlir::Type t) {
   return ::llvm::TypeSwitch<mlir::Type, uint8_t>(t)
      .Case<mlir::IntegerType>([&](mlir::IntegerType integerType) {
         switch (integerType.getWidth()) {
            case 1: return TypeI8;
            case 8: return TypeI8;
            case 16: return TypeI16;
            case 32: return TypeI32;
            case 64: return TypeI64;
            case 128: return TypeI128;
         }
         assert(false && "unknown type");
      })
      .Case<mlir::Float32Type>([&](mlir::Float32Type) { return TypeF32; })
      .Case<mlir::Float64Type>([&](mlir::Float64Type) { return TypeF64; })
      .Default([](mlir::Type) { return 0; });
}
uint8_t mlir::cranelift::CraneliftExecutionEngine::translateFuncType(mlir::Type t) {
   if (t.isInteger(1)) {
      return TypeI8;
   } else {
      return translateType(t);
   }
}

void mlir::cranelift::CraneliftExecutionEngine::translate(mlir::cranelift::FuncOp fn) {
   if (fn.getBody().empty()) {
      return;
   }
   cranelift_signature_builder_reset(mod, CraneliftCallConv::CraneliftCallConvSystemV);

   for (auto t : fn.getArgumentTypes()) {
      cranelift_signature_builder_add_param(mod, translateType(t));
   }
   for (auto t : fn.getResultTypes()) {
      cranelift_signature_builder_add_result(mod, translateType(t));
   }
   struct Context {
      CraneliftExecutionEngine* thisPtr;
      mlir::Operation* currFunc;
   };
   Context ctxt{this, fn.getOperation()};

   cranelift_build_function(mod, (uintptr_t) &ctxt, [](uintptr_t userdata, FunctionData* fd) {
      auto* ctxt = (Context*) userdata;
      mlir::cranelift::FuncOp fn = mlir::cast<mlir::cranelift::FuncOp>(ctxt->currFunc);
      std::unordered_map<mlir::Block*, BlockCode> translatedBlocks;
      for (auto& x : fn.getBody().getBlocks()) {
         translatedBlocks.insert({&x, cranelift_create_block(fd)});
      }
      llvm::DenseMap<mlir::Value, ValueCode> varMapping;
      for (auto& x : fn.getBody().getBlocks()) {
         auto blockCode = translatedBlocks.at(&x);
         cranelift_switch_to_block(fd, blockCode);
         for (auto blockParamType : x.getArgumentTypes()) {
            cranelift_append_block_param(fd, blockCode, translateType(blockParamType));
         }
         std::vector<ValueCode> blockParams(x.getNumArguments());
         cranelift_block_params(fd, blockCode, blockParams.data());
         assert(x.getNumArguments() == static_cast<size_t>(cranelift_block_params_count(fd, blockCode)));
         for (size_t i = 0; i < x.getNumArguments(); i++) {
            varMapping.insert({x.getArgument(i), blockParams[i]});
         }
         auto store = [&](mlir::Value v, ValueCode vc) {
            varMapping.insert({v, vc});
         };
         auto load = [&](mlir::Value v) {
            assert(varMapping.count(v));
            return varMapping[v];
         };
         for (auto& currentOp : x) {
            llvm::TypeSwitch<mlir::Operation*>(&currentOp)
               .Case([&](mlir::cranelift::IConstOp op) { store(op, cranelift_iconst(fd, translateType(op.getType()), op.getValue())); })
               .Case([&](mlir::cranelift::F32ConstOp op) { store(op, cranelift_f32const(fd, op.getValue().convertToFloat())); })
               .Case([&](mlir::cranelift::F64ConstOp op) { store(op, cranelift_f64const(fd, op.getValue().convertToDouble())); })
               .Case([&](mlir::cranelift::IAddOp op) { store(op, cranelift_iadd(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::ISubOp op) { store(op, cranelift_isub(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::IMulOp op) { store(op, cranelift_imul(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::UMulHiOp op) { store(op, cranelift_umulhi(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::FAddOp op) { store(op, cranelift_fadd(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::FSubOp op) { store(op, cranelift_fsub(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::FMulOp op) { store(op, cranelift_fmul(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::FDivOp op) { store(op, cranelift_fdiv(fd, load(op.getLhs()), load(op.getRhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::SShrOp op) { store(op, cranelift_sshr(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::UShrOp op) { store(op, cranelift_ushr(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::IShlOp op) { store(op, cranelift_ishl(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::BOrOp op) { store(op, cranelift_bor(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::BXOrOp op) {
                  assert(!op.getType().isInteger(128));
                  if(op.getType().isInteger(1)){
                     store(op,cranelift_band(fd, cranelift_bxor(fd, load(op.getLhs()), load(op.getRhs())),cranelift_iconst(fd,TypeI8,1)));
                  }
                  store(op, cranelift_bxor(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::BAndOp op) { store(op, cranelift_band(fd, load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::BSwap op) { store(op, cranelift_bswap(fd, load(op.getX()))); })
               .Case([&](mlir::cranelift::SelectOp op) {
                  store(op, cranelift_select(fd, load(op.getCondition()), load(op.getTrueVal()), load(op.getFalseVal()))); })
               .Case([&](mlir::cranelift::ICmpOp op) { store(op, cranelift_icmp(fd, static_cast<CraneliftIntCC>(op.getPredicate()), load(op.getLhs()), load(op.getRhs()))); })
               .Case([&](mlir::cranelift::FCmpOp op) { store(op, cranelift_fcmp(fd, static_cast<CraneliftFloatCC>(op.getPredicate()), load(op.getLhs()), load(op.getRhs()))); })

               .Case([&](mlir::cranelift::SExtendOp op) {
                  store(op, cranelift_sextend(fd, translateType(op.getType()), load(op.getValue()))); })
               .Case([&](mlir::cranelift::UExtendOp op) {
                  auto val = load(op.getValue());
                  if (op.getValue().getType().isInteger(1)) {
                     val = cranelift_band(fd, val, cranelift_iconst(fd, TypeI8, 1));
                  }
                  store(op, cranelift_uextend(fd, translateType(op.getType()), val));
               })
               .Case([&](mlir::cranelift::SIToFP op) { store(op, cranelift_fcvt_from_sint(fd, translateType(op.getType()), load(op.getValue()))); })
               .Case([&](mlir::cranelift::UIToFP op) { store(op, cranelift_fcvt_from_uint(fd, translateType(op.getType()), load(op.getValue()))); })
               .Case([&](mlir::cranelift::FPromoteOp op) { store(op, cranelift_fpromote(fd, translateType(op.getType()), load(op.getValue()))); })
               .Case([&](mlir::cranelift::FDemoteOp op) { store(op, cranelift_fdemote(fd, translateType(op.getType()), load(op.getValue()))); })
               .Case([&](mlir::cranelift::IConcatOp op) { store(op, cranelift_iconcat(fd, load(op.getLower()), load(op.getHigher()))); })
               .Case([&](mlir::cranelift::ISplitOp op) {
                  ValueCode lower, higher;
                  cranelift_isplit(fd, load(op.getVal()), &lower, &higher);

                  store(op.getResult(0), lower);
                  store(op.getResult(1), higher);
               })
               .Case([&](mlir::cranelift::IReduceOp op) { store(op, cranelift_ireduce(fd, translateType(op.getType()), load(op.getValue()))); })
               .Case([&](mlir::cranelift::UDivOp op) { store(op, cranelift_udiv(fd, load(op.getLhs()), load(op.getRhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::SDivOp op) { store(op, cranelift_sdiv(fd, load(op.getLhs()), load(op.getRhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::URemOp op) { store(op, cranelift_urem(fd, load(op.getLhs()), load(op.getRhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::SRemOp op) { store(op, cranelift_srem(fd, load(op.getLhs()), load(op.getRhs()))); }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::AtomicRmwOp op) {
                  assert(op.getRmwOp() == AtomicRmwOpType::Xchg);
                  store(op, cranelift_atomic_rmw(fd, translateType(op.getType()), "xchg", load(op.getP()), load(op.getX())));
               })

               .Case([&](mlir::cranelift::StoreOp op) {
                  cranelift_store(fd, 0, load(op.getX()), load(op.getP()), 0);
               }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::LoadOp op) {
                  if (op.getType().isInteger(1)) {
                     auto loaded = cranelift_load(fd, TypeI8, 0, load(op.getP()), 0);
                     store(op, cranelift_icmp_imm(fd, CraneliftIntCC::CraneliftIntCCNotEqual, loaded, 0));
                  } else {
                     store(op, cranelift_load(fd, translateType(op.getType()), 0, load(op.getP()), 0));
                  }
               }) //todo: use immediate if possible?
               .Case([&](mlir::cranelift::AllocaOp op) {
                  auto stackSlot = cranelift_create_stack_slot(fd, op.getSize());
                  store(op.getRef(), cranelift_stack_addr(fd, TypeI64, stackSlot, 0));
               })
               .Case([&](mlir::cranelift::AddressOfOp op) {
                  auto dataId = cranelift_declare_data_in_current_func(fd, ctxt->thisPtr->dataIds.at(op.getSymbolName().str()));
                  store(op, cranelift_symbol_value(fd, translateType(op.getType()), dataId));
               })
               .Case([&](mlir::cranelift::ReturnOp op) {
                  std::vector<ValueCode> returnVals;
                  for (auto x : op.getOperands()) {
                     if (x.getType().isInteger(1)) {
                        returnVals.push_back(cranelift_band(fd, varMapping[x], cranelift_iconst(fd, TypeI8, 1)));
                     } else {
                        returnVals.push_back(varMapping[x]);
                     }
                  }
                  cranelift_return(fd, returnVals.size(), returnVals.data());
               })
               .Case([&](mlir::cranelift::CallOp op) {
                  uint32_t funcId;
                  if (ctxt->thisPtr->functionIds.contains(op.getCallee().str())) {
                     funcId = ctxt->thisPtr->functionIds[op.getCallee().str()];
                  } else {
                     std::vector<uint8_t> argTypes;
                     std::vector<uint8_t> resTypes;

                     for (auto t : op.getOperandTypes()) {
                        argTypes.push_back(translateType(t));
                     }
                     for (auto t : op.getResultTypes()) {
                        resTypes.push_back(translateType(t));
                     }

                     funcId = cranelift_import_func(fd, op.getCallee().data(), CraneliftCallConv::CraneliftCallConvSystemV, argTypes.size(), argTypes.data(), resTypes.size(), resTypes.data());
                     ctxt->thisPtr->functionIds[op.getCallee().str()] = funcId;
                  }
                  std::vector<ValueCode> args;
                  for (auto v : op.getOperands()) {
                     if (v.getType().isInteger(1)) {
                        args.push_back(cranelift_band(fd, load(v), cranelift_iconst(fd, TypeI8, 1)));
                     } else {
                        args.push_back(load(v));
                     }
                  }
                  auto declaredFn = cranelift_declare_func_in_current_func(fd, funcId);
                  auto inst = cranelift_call(fd, declaredFn, op.getNumOperands(), args.data());
                  for (size_t i = 0; i < op.getNumResults(); i++) {
                     store(op.getResult(i), cranelift_inst_result(fd, inst, i));
                  }
               })
               .Case([&](mlir::cranelift::FuncAddrOp op) {
                  auto declaredFn = cranelift_declare_func_in_current_func(fd, ctxt->thisPtr->functionIds.at(op.getCallee().str()));
                  store(op, cranelift_func_addr(fd, TypeI64, declaredFn));
               })
               .Case([&](mlir::cranelift::BranchOp op) {
                  std::vector<ValueCode> returnVals;
                  for (auto x : op.getDestOperands()) {
                     returnVals.push_back(varMapping[x]);
                  }
                  cranelift_ins_jump(fd, translatedBlocks.at(op.getDest()), returnVals.size(), returnVals.data());
               })
               .Case([&](mlir::cranelift::CondBranchOp op) {
                  std::vector<ValueCode> falseDestVals;
                  for (auto x : op.getFalseDestOperands()) {
                     falseDestVals.push_back(varMapping[x]);
                  }
                  std::vector<ValueCode> trueDestVals;
                  for (auto x : op.getTrueDestOperands()) {
                     trueDestVals.push_back(varMapping[x]);
                  }
                  cranelift_ins_brnz(fd, load(op.getCondition()), translatedBlocks.at(op.getTrueDest()), trueDestVals.size(), trueDestVals.data());

                  cranelift_ins_jump(fd, translatedBlocks.at(op.getFalseDest()), falseDestVals.size(), falseDestVals.data());
               })
               .Default([](mlir::Operation* op) {op->dump();assert(false&&"unknown operation"); });
         }
      }
      cranelift_seal_all_blocks(fd);
   });

   /*cranelift_function_to_string(mod, 0, [](uintptr_t ud, const char* data) {
      printf("\n%s\n", data);
   });*/

   uint32_t id;
   cranelift_declare_function(mod, fn.getSymName().data(), CraneliftLinkage::Export, &id);
   cranelift_define_function(mod, id);
   functionIds[fn.getSymName().str()] = id;
   cranelift_clear_context(mod);
}

void mlir::cranelift::CraneliftExecutionEngine::translate(mlir::ModuleOp module) {
   for (mlir::Operation& op : module.getBody()->getOperations()) {
      if (auto globalOp = mlir::dyn_cast_or_null<mlir::cranelift::GlobalOp>(&op)) {
         uint32_t id2;
         cranelift_set_data_value(mod, (const uint8_t*) globalOp.getValue().data(), globalOp.getValue().size());
         cranelift_define_data(mod, globalOp.getSymbolName().data(), CraneliftLinkage::Export, CraneliftDataFlags::Writable, 0, &id2);
         cranelift_assign_data_to_global(mod, id2);
         cranelift_clear_data(mod);
         dataIds[globalOp.getSymbolName().str()] = id2;
      } else if (auto funcOp = mlir::dyn_cast_or_null<mlir::cranelift::FuncOp>(&op)) {
         translate(funcOp);
      }
   }
   success = true;
}

void* mlir::cranelift::CraneliftExecutionEngine::getFunction(std::string name) {
   auto funcId = functionIds.at(name);
   return const_cast<void*>(reinterpret_cast<const void*>(cranelift_get_compiled_fun(mod, funcId)));
}
size_t mlir::cranelift::CraneliftExecutionEngine::getJitTime() const {
   return jitTime;
}

mlir::cranelift::CraneliftExecutionEngine::~CraneliftExecutionEngine() {
   cranelift_module_delete(mod);
}