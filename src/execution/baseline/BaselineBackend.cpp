#if BASELINE_ENABLED == 1
#if !defined(__linux__)
#error "Baseline backend is only supported on Linux systems."
#endif

#include "lingodb/execution/BackendPasses.h"
#include "lingodb/execution/BaselineBackend.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/execution/baseline/utils.hpp"
#include "lingodb/utility/Setting.h"
#include "lingodb/utility/Tracer.h"

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/ExecutionEngine/JITLink/EHFrameSupport.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>

#include <tpde/CompilerBase.hpp>
#include <tpde/x64/CompilerX64.hpp>
#include <tpde/ElfMapper.hpp>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <sstream>

#include <dlfcn.h>

#include "snippet_encoders_x64.hpp"

namespace lingodb::execution::baseline {
    namespace utility = lingodb::utility;
    utility::Tracer::Event execution("Execution", "run");
    utility::Tracer::Event llvmCodeGen("Compilation", "BaselineCodeGen");

    utility::GlobalSetting<std::string> baselineDebugFileOut("system.compilation.baseline_object_out", "");

    using namespace compiler;


    class SpdLogSpoof {
        // storage for the log messages
        std::ostringstream oss;
        std::shared_ptr<spdlog::sinks::ostream_sink_mt> ostream_sink;
        std::shared_ptr<spdlog::logger> logger;
        std::shared_ptr<spdlog::logger> old_logger;

    public:
        SpdLogSpoof() : ostream_sink(std::make_shared<spdlog::sinks::ostream_sink_mt>(oss)),
                        logger(std::make_shared<spdlog::logger>("string_logger", ostream_sink)) {
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

    static std::optional<size_t> get_size(mlir::Type type) noexcept {
        return mlir::TypeSwitch<mlir::Type, size_t>(type)
                .Case<mlir::IntegerType>([](auto intType) { return intType.getIntOrFloatBitWidth() / 8; })
                .Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) { return 16; })
                .Case<mlir::TupleType>([](auto t) { return TupleHelper{t}.sizeAndPadding().first; })
                .Case<mlir::IndexType, dialect::util::RefType>([](auto) { return 8; })
                .Default([](mlir::Type t) {
                    t.dump();
                    assert(0 && "Unsupported type for size calculation");
                    return 0;
                });
    }

    // adaptor mlir -> tpde
    // NOLINTBEGIN(readability-identifier-naming)
    struct IRAdaptor {
        using IRFuncRef = mlir::func::FuncOp;
        using IRBlockRef = mlir::Block *;
        using IRInstRef = mlir::Operation *;
        using IRValueRef = mlir::Value;

        [[maybe_unused]] static IRFuncRef INVALID_FUNC_REF;
        [[maybe_unused]] static IRBlockRef INVALID_BLOCK_REF;
        [[maybe_unused]] static IRValueRef INVALID_VALUE_REF;

        [[maybe_unused]] static constexpr bool TPDE_PROVIDES_HIGHEST_VAL_IDX = true;
        [[maybe_unused]] static constexpr bool TPDE_LIVENESS_VISIT_ARGS = true;

        using IR = mlir::ModuleOp;

        IR *module;
        IRFuncRef cur_func = INVALID_FUNC_REF;
        Error &error;

        struct ValInfo {
            tpde::ValLocalIdx local_idx;
        };

        llvm::DenseMap<IRBlockRef, std::pair<uint32_t, uint32_t> > blockInfoMap;
        llvm::DenseMap<IRValueRef, ValInfo> values;

        IRAdaptor(mlir::ModuleOp *module, Error &error) : module(module), error(error) {
        }

        Error &getError() { return error; }

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
            return cur_func.getFunctionBody().getOps<dialect::util::AllocaOp>();
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
                   std::views::transform([](mlir::Block &block) {
                       return &block;
                   });
        }

        [[maybe_unused]] auto block_succs(IRBlockRef block) const noexcept {
            return block->getSuccessors();
        }

        [[maybe_unused]] auto block_insts(IRBlockRef block) const noexcept {
            return block->getOperations() | std::views::transform([](mlir::Operation &op) {
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
                mlir::Operation *terminator = predecessor->getTerminator();
                const mlir::Value incomingVal = mlir::TypeSwitch<mlir::Operation *, mlir::Value>(terminator)
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
                        .Default([](mlir::Operation *op) {
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
                mlir::Block *predecessor = incoming_block_for_slot(slot);
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

        [[maybe_unused]] uint32_t val_alloca_size(IRValueRef val) const noexcept {
            assert(mlir::isa<dialect::util::AllocaOp>(val.getDefiningOp()) && "Value is not an alloca operation");
            auto allocaOp = cast<dialect::util::AllocaOp>(val.getDefiningOp());
            unsigned count = 1; // default size for an alloca without a size is 1
            if (const auto size = allocaOp.getSize()) {
                if (auto const_int = mlir::dyn_cast<mlir::arith::ConstantIntOp>(size.getDefiningOp())) {
                    count = const_int.value();
                }
                if (auto const_index = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(size.getDefiningOp())) {
                    count = const_index.value();
                }
                error.emit() << "Value is not an arith int constant";
                abort();
            }

            const mlir::TypedValue<dialect::util::RefType> res_ptr = allocaOp.getRef();
            const mlir::Type element_type = res_ptr.getType().getElementType();
            const auto size = get_size(element_type);
            if (!size) {
                error.emit() << "Value is not a supported type for alloca size calculation: ";
                element_type.dump();
                abort();
            }
            return *size * count;
        }

        [[maybe_unused]] uint32_t val_alloca_align(IRValueRef val) const noexcept {
            return 1;
        }

        [[maybe_unused]] std::string value_fmt_ref(IRValueRef val) const noexcept {
            if (const auto op = val.getDefiningOp()) {
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
            values.clear();
            for (auto &block: func.getFunctionBody()) {
                for (auto arg: block.getArguments())
                    values[arg] = ValInfo{
                        .local_idx = tpde::ValLocalIdx(values.size())
                    };
                for (auto &op: block) {
                    for (auto result: op.getResults())
                        values[result] = ValInfo{
                            .local_idx = tpde::ValLocalIdx(values.size())
                        };
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
    template<typename Adapter, typename Derived, typename Config>
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
        using SymRef = typename Assembler::SymRef;

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

            SymRef get_symbol(IRCompilerBase *compiler, Type type) {
                if (SymRef ref = funcs[static_cast<unsigned>(type)]; ref != SymRef{}) { return ref; } else {
                    std::string name;
                    switch (type) {
                        case Type::divti3: name = "__divti3";
                            break;
                        case Type::udivti3: name = "__udivti3";
                            break;
                        case Type::modti3: name = "__modti3";
                            break;
                        case Type::umodti3: name = "__umodti3";
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
        llvm::StringMap<void *> externFuncMap;

        IRCompilerBase(IRAdaptor *adaptor)
            : Base{adaptor} {
            static_assert(tpde::Compiler<Derived, Config>);
            static_assert(std::is_same_v<Adapter, IRAdaptor>, "Adapter must be IRAdaptor");

            dialect::util::FunctionHelper::visitAllFunctions([&](std::string s, void *ptr) { externFuncMap[s] = ptr; });
            execution::visitBareFunctions([&](std::string s, void *ptr) { externFuncMap[s] = ptr; });
        }

        Error &getError() { return error; }

        // shortcuts to access the derived class later
        Derived *derived() noexcept { return static_cast<Derived *>(this); }

        const Derived *derived() const noexcept { return static_cast<Derived *>(this); }

        const IR *ir() const noexcept { return this->adaptor->module; }

        bool cur_func_may_emit_calls() { return true; }

        static CompilerConfig::Assembler::SymRef cur_personality_func() {
            // we do not support exceptions, so we do not need a personality function
            return {};
        }

        bool try_force_fixed_assignment(IRAdaptor::IRValueRef) const noexcept { return false; }

        struct ValueParts {
            mlir::Type valType;

            uint32_t count() const noexcept {
                return mlir::TypeSwitch<mlir::Type, uint32_t>(valType)
                        .Case<mlir::IntegerType>([](auto intType) {
                            return (intType.getIntOrFloatBitWidth() + 64 - 1) / 64;
                        })
                        .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) { return 2; })
                        .Default([](mlir::Type t) { return 1; });
            }

            uint32_t size_bytes(uint32_t part_idx) const noexcept {
                return mlir::TypeSwitch<mlir::Type, uint32_t>(valType)
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
                        .Default([](mlir::Type t) {
                            t.dump();
                            assert(0 && "invalid type");
                            return 0;
                        }); // all other types are not supported yet
            }

            tpde::RegBank reg_bank(uint32_t part_idx) const noexcept {
                return mlir::TypeSwitch<mlir::Type, tpde::RegBank>(valType)
                        .Case<mlir::IntegerType, mlir::IndexType, dialect::util::RefType, mlir::FunctionType,
                            dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
                            return Config::GP_BANK;
                        })
                        .template Case<mlir::FloatType>([](auto) { return Config::FP_BANK; })

                        .Default([](mlir::Type t) {
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
            if (const auto *op = val.getDefiningOp()) {
                return mlir::TypeSwitch<const mlir::Operation *, std::optional<ValRefSpecial> >(op)
                        .template Case<mlir::arith::ConstantOp, dialect::util::SizeOfOp, dialect::util::UndefOp>(
                            [&](auto) { return ValRefSpecial{.mode = 4, .value = val}; })
                        .template Case<dialect::util::CreateConstVarLen>(
                            [&](auto constVarLenOp) -> std::optional<ValRefSpecial> {
                                const mlir::StringRef content = mlir::cast<mlir::StringAttr>(
                                    constVarLenOp->getAttr(constVarLenOp.getStrAttrName())).getValue();
                                if (content.size() <= 12) { return ValRefSpecial{.mode = 4, .value = val}; } else {
                                    // long string require a pointer to the content, making them non-constant
                                    return std::nullopt;
                                }
                            })
                        .Default([&](auto) { return std::nullopt; });
            }
            return std::nullopt;
        }

        ValuePartRef val_part_ref_special(ValRefSpecial &ref, uint32_t part) noexcept {
            if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(ref.value.getDefiningOp())) {
                if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constOp.getValue())) {
                    const mlir::APInt containedInt = intAttr.getValue();
                    unsigned bitWidth;
                    if (mlir::isa<mlir::IndexType>(intAttr.getType())) { bitWidth = 64; } else {
                        bitWidth = intAttr.getType().getIntOrFloatBitWidth();
                    }
                    if (bitWidth <= 64) {
                        assert(part == 0);
                        return ValuePartRef(this, containedInt.getRawData()[0], bitWidth / 8, Config::GP_BANK);
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
                const uint64_t tuple_size = TupleHelper{mlir::cast<mlir::TupleType>(type_attr.getValue())}.
                        sizeAndPadding().first;
                return ValuePartRef(this, tuple_size, 8, Config::GP_BANK);
            }
            if (const auto constVarLenOp = mlir::dyn_cast_or_null<dialect::util::CreateConstVarLen>(
                ref.value.getDefiningOp())) {
                assert(constVarLenOp->getAttrs().size() == 1);
                const mlir::StringRef content = mlir::cast<mlir::StringAttr>(
                            constVarLenOp->getAttrs().front().getValue()).
                        getValue();
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
                    const unsigned register_width = width == 128 ? 8 : width / 8;
                    return ValuePartRef(this, 0, register_width, Config::GP_BANK);
                } else if (mlir::isa<mlir::FloatType>(ret_type)) {
                    return ValuePartRef(this, 0, ret_type.getIntOrFloatBitWidth() / 8, Config::FP_BANK);
                } else if (mlir::isa<dialect::util::VarLen32Type, dialect::util::BufferType>(ret_type)) {
                    return ValuePartRef(this, 0, 8, Config::FP_BANK);
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
            // TODO: implement this properly
            return mlir::isa<dialect::util::VarLen32Type>(val.getType()) || (
                       mlir::isa<mlir::IntegerType>(val.getType()) && val.getType().getIntOrFloatBitWidth() == 128) ||
                   mlir::isa<dialect::util::BufferType>(val.getType());
        }

        static bool arg_allow_split_reg_stack_passing(IRAdaptor::IRValueRef val) noexcept {
            return false; // we do not support split register stack passing
        }

        void define_func_idx(IRAdaptor::IRFuncRef func, uint32_t idx) {
            localFuncMap[func.getSymName()] = idx;
        }

        bool compile_arith_binary_op(IRInstRef op) {
            auto res_type = op->getResult(0).getType();
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
                    case 32: op_width = 32;
                        break;
                    case 64: op_width = 64;
                        break;
                    case 128: op_width = 128;
                        break;
                    default: op->dump();
                        assert(0 && "Unsupported integer type width for arithmetic operation");
                        return false;
                }
            } else if (mlir::isa<mlir::FloatType>(res_type)) {
                res_width = res_type.getIntOrFloatBitWidth();
                switch (res_width) {
                    case 32: op_width = 32;
                        break;
                    case 64: op_width = 64;
                        break;
                    default: res_type.dump();
                        assert(0 && "Unsupported float type width for arithmetic operation");
                        return false;
                }
            } else {
                assert(0 && "Unsupported type for arithmetic operation");
                return false;
            }

            if (op_width == 128) {
                auto res = this->result_ref(op->getResult(0));
                auto builtin_func = mlir::TypeSwitch<mlir::Operation *, std::optional<SymRef> >(op)
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

                std::unordered_map<std::string, bool (Derived::*)(GenericValuePart &&, GenericValuePart &&,
                                                                  GenericValuePart &&, GenericValuePart &&,
                                                                  ScratchReg &, ScratchReg &)> encoder_lookup = {
                    {"arith.addi", &Derived::encode_arith_add_i128},
                    {"arith.andi", &Derived::encode_arith_land_i128},
                    {"arith.ori", &Derived::encode_arith_lor_i128},
                    {"arith.xori", &Derived::encode_arith_lxor_i128},
                    {"arith.muli", &Derived::encode_arith_mul_i128},
                    {"arith.subi", &Derived::encode_arith_sub_i128},
                };
#ifndef NDEBUG
                if (!encoder_lookup.contains(op->getName().getStringRef().str().c_str())) {
                    op->dump();
                    error.emit() << op->getName().getStringRef().str().c_str() <<
                            " is not supported by the baseline backend\n";
                    return false;
                }
#endif
                const auto encoder = encoder_lookup[op->getName().getStringRef().str().c_str()];

                auto lhs_vr = this->val_ref(op->getOperand(0));
                auto rhs_vr = this->val_ref(op->getOperand(1));
                ScratchReg res_scratch_high{derived()};
                ScratchReg res_scratch_low{derived()};
                auto res_low = res.part(0);
                auto res_high = res.part(1);
                (derived()->*encoder)(std::move(lhs_vr.part(0)), std::move(lhs_vr.part(1)), std::move(rhs_vr.part(0)),
                                      std::move(rhs_vr.part(1)), res_scratch_low, res_scratch_high);
                this->set_value(res_low, res_scratch_low);
                this->set_value(res_high, res_scratch_high);
                return true;
            } else {
                auto lhs_vr = this->val_ref(op->getOperand(0));
                auto rhs_vr = this->val_ref(op->getOperand(1));
                auto lhs_pr = lhs_vr.part(0);
                auto rhs_pr = rhs_vr.part(0);
                // move constant operands to the right side
                if ((mlir::isa<mlir::arith::AddIOp, mlir::arith::MulIOp, mlir::arith::AndIOp, mlir::arith::OrIOp,
                        mlir::arith::XOrIOp>(op)) && lhs_pr.is_const() && !rhs_pr.is_const()) {
                    std::swap(lhs_vr, rhs_vr);
                    std::swap(lhs_pr, rhs_pr);
                }

                if (op_width > res_width) {
                    lhs_pr = std::move(lhs_pr).into_extended(true, res_width, op_width);
                    rhs_pr = std::move(rhs_pr).into_extended(true, res_width, op_width);
                }

                auto lhs_op = GenericValuePart{std::move(lhs_pr)};
                auto rhs_op = GenericValuePart{std::move(rhs_pr)};

                ScratchReg res_scratch{derived()};

                // encode functions for 32/64 bit operations
                // TODO: replace this map with something more efficient
                std::unordered_map<std::string, std::array<bool (Derived::*)(
                    GenericValuePart &&, GenericValuePart &&, ScratchReg &), 2> > encoder_lookup;

                if (mlir::isa<mlir::FloatType>(res_type)) {
                    encoder_lookup = {
                        {"arith.addf", {&Derived::encode_arith_add_f32, &Derived::encode_arith_add_f64}},
                        {"arith.subf", {&Derived::encode_arith_sub_f32, &Derived::encode_arith_sub_f64}},
                        {"arith.mulf", {&Derived::encode_arith_mul_f32, &Derived::encode_arith_mul_f64}},
                        {"arith.divf", {&Derived::encode_arith_div_f32, &Derived::encode_arith_div_f64}},
                        // {"arith.remf", {&Derived::encode_arith_rem_f32, &Derived::encode_arith_rem_f64}},
                    };
                } else {
                    encoder_lookup = {
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
                        {"arith.shrui", {&Derived::encode_arith_shr_u32, &Derived::encode_arith_shr_u64}}
                    };
                }
#ifndef NDEBUG
                if (!encoder_lookup.contains(op->getName().getStringRef().str().c_str())) {
                    op->dump();
                    std::cerr << op->getName().getStringRef().str().c_str() <<
                            " is not supported by the baseline backend\n";
                    assert(0);
                    return false;
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
            auto base_vr = this->val_ref(base_ref);
            auto res_vr = this->result_ref(dst);

            // create a base + offset expression
            AsmReg base_reg = base_vr.part(0).load_to_reg();
            GenericValuePart addr = typename GenericValuePart::Expr{base_reg, elementOffset};

            // load value to register (e.g. mov + add / lea for x86_64)
            AsmReg res_reg = derived()->gval_expr_as_reg(addr);
            ScratchReg res_scratch{derived()};
            derived()->mov(res_scratch.alloc_gp(), res_reg, 8);
            this->set_value(res_vr.part(0), res_scratch);
            return true;
        }

        bool compile_util_buffer_cast_op(dialect::util::BufferCastOp op) {
            // TODO: replace code below with this code to omit unnecessary copies (currently this doesn't work though)
            // auto [tmp, src_vpr] = this->val_ref_single(op.getVal());
            // auto [tmp1, res_vpr] = this->result_ref_single(op.getRes());
            // res_vpr.set_value(std::move(src_vpr));
            // return true;
            auto src_vr = this->val_ref(op.getVal());
            auto res_vr = this->result_ref(op.getRes());
            ScratchReg res_scratch_low{derived()};
            ScratchReg res_scratch_high{derived()};
            derived()->mov(res_scratch_low.alloc_gp(), src_vr.part(0).load_to_reg(), 8);
            derived()->mov(res_scratch_high.alloc_gp(), src_vr.part(1).load_to_reg(), 8);
            this->set_value(res_vr.part(0), res_scratch_low);
            this->set_value(res_vr.part(1), res_scratch_high);
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

            GenericValuePart ptr;
            // store to [ptr + idx * elem_size]
            auto elem_size = get_size(elem_type);
            if (!elem_size) {
                assert(0 && "Unsupported type for store operation");
                error.emit() << "Unsupported type for store operation.";
                return false;
            }
            AsmReg buf_ptr_reg = buf_vr.part(1).load_to_reg();
            if (auto idx_op = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(idx.getDefiningOp())) {
                ptr = GenericValuePart{
                    Expr{std::move(buf_ptr_reg), static_cast<int64_t>(static_cast<size_t>(idx_op.value()) * *elem_size)}
                };
            } else {
                auto [_, idx_ref] = this->val_ref_single(idx);
                auto generic_ptr_expr = Expr{std::move(buf_ptr_reg)};
                generic_ptr_expr.index = idx_ref.alloc_reg();
                generic_ptr_expr.scale = *elem_size;
                ptr = GenericValuePart{std::move(generic_ptr_expr)};
            }

            // load value to register (e.g. mov + add / lea for x86_64)
            AsmReg res_reg = derived()->gval_expr_as_reg(ptr);
            ScratchReg res_scratch{derived()};
            derived()->mov(res_scratch.alloc_gp(), res_reg, 8);

            auto [_, res_vr] = this->result_ref_single(dst);
            this->set_value(res_vr, res_scratch);
            return true;
        }

        bool compile_util_store_op(dialect::util::StoreOp op) {
            const mlir::Value in = op.getVal();
            const mlir::TypedValue<dialect::util::RefType> ptr = op.getRef();
            const mlir::TypedValue<mlir::IndexType> idx = op.getIdx();
            const mlir::Type stored_type = in.getType();

            auto [_, ptr_ref] = this->val_ref_single(ptr);
            GenericValuePart ptr_part;
            if (idx) {
                // store to [ptr + idx * elem_size]
                auto elem_size = get_size(stored_type);
                if (!elem_size) {
                    assert(0 && "Unsupported type for store operation");
                    error.emit() << "Unsupported type for store operation.";
                    return false;
                }
                if (auto idx_op = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(idx.getDefiningOp())) {
                    AsmReg ptr_reg = ptr_ref.load_to_reg();
                    ptr_part = GenericValuePart{
                        Expr{std::move(ptr_reg), static_cast<int64_t>(static_cast<size_t>(idx_op.value()) * *elem_size)}
                    };
                } else {
                    auto [_, idx_ref] = this->val_ref_single(idx);
                    auto generic_ptr_expr = Expr{ptr_ref.load_to_reg()};
                    generic_ptr_expr.index = idx_ref.load_to_reg();
                    generic_ptr_expr.scale = *elem_size;
                    ptr_part = GenericValuePart{std::move(generic_ptr_expr)};
                }
            } else {
                ptr_part = GenericValuePart{std::move(ptr_ref)};
            }

            auto in_vr = this->val_ref(in);
            return mlir::TypeSwitch<mlir::Type, bool>(stored_type)
                    .Case([&](const mlir::IntegerType t) {
                        switch (t.getIntOrFloatBitWidth()) {
                            case 1: return derived()->encode_util_store_i1(std::move(ptr_part), in_vr.part(0));
                            case 8: return derived()->encode_util_store_i8(std::move(ptr_part), in_vr.part(0));
                            case 16: return derived()->encode_util_store_i16(std::move(ptr_part), in_vr.part(0));
                            case 32: return derived()->encode_util_store_i32(std::move(ptr_part), in_vr.part(0));
                            case 64: return derived()->encode_util_store_i64(std::move(ptr_part), in_vr.part(0));
                            case 128: return derived()->encode_store_i128(
                                    std::move(ptr_part), in_vr.part(0), in_vr.part(1));
                            default:
                                assert(0 && "Unsupported integer type width for store operation");
                                return false;
                        }
                    })
                    .template Case<dialect::util::RefType, mlir::IndexType>([&](auto) {
                        return derived()->encode_util_store_i64(std::move(ptr_part), in_vr.part(0));
                    })
                    .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
                        return derived()->encode_store_i128(std::move(ptr_part), in_vr.part(0), in_vr.part(1));
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
            GenericValuePart ptr_part;
            if (idx) {
                // store to [ptr + idx * elem_size]
                auto elem_size = get_size(loaded_type);
                if (!elem_size) {
                    assert(0 && "Unsupported type for store operation");
                    error.emit() << "Unsupported type for store operation.";
                    return false;
                }
                if (auto idx_op = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(idx.getDefiningOp())) {
                    AsmReg ptr_reg = ptr_ref.load_to_reg();
                    ptr_part = GenericValuePart{
                        Expr{std::move(ptr_reg), static_cast<int64_t>(static_cast<size_t>(idx_op.value()) * *elem_size)}
                    };
                } else {
                    auto [_, idx_ref] = this->val_ref_single(idx);
                    auto generic_ptr_expr = Expr{ptr_ref.load_to_reg()};
                    generic_ptr_expr.index = idx_ref.load_to_reg();
                    generic_ptr_expr.scale = *elem_size;
                    ptr_part = GenericValuePart{std::move(generic_ptr_expr)};
                }
            } else {
                ptr_part = GenericValuePart{std::move(ptr_ref)};
            }

            auto res = this->result_ref(op.getVal());
            ScratchReg res_scratch{this};
            return mlir::TypeSwitch<mlir::Type, bool>(loaded_type)
                    .Case([&](const mlir::IntegerType t) {
                        switch (t.getIntOrFloatBitWidth()) {
                            case 1: derived()->encode_util_load_i1(std::move(ptr_part), res_scratch);
                                break;
                            case 8: derived()->encode_util_load_i8(std::move(ptr_part), res_scratch);
                                break;
                            case 16: derived()->encode_util_load_i16(std::move(ptr_part), res_scratch);
                                break;
                            case 32: derived()->encode_util_load_i32(std::move(ptr_part), res_scratch);
                                break;
                            case 64: derived()->encode_util_load_i64(std::move(ptr_part), res_scratch);
                                break;
                            case 128: {
                                ScratchReg res_scratch_high{derived()};
                                auto res_low = res.part(0);
                                auto res_high = res.part(1);
                                derived()->encode_load_i128(std::move(ptr_part), res_scratch, res_scratch_high);
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
                    .template Case<dialect::util::VarLen32Type, dialect::util::BufferType>([&](auto) {
                        ScratchReg res_scratch_high{derived()};
                        auto res_low = res.part(0);
                        auto res_high = res.part(1);
                        derived()->encode_load_i128(std::move(ptr_part), res_scratch, res_scratch_high);
                        this->set_value(res_low, res_scratch);
                        this->set_value(res_high, res_scratch_high);
                        return true;
                    })
                    .Default([](const mlir::Type t) {
                        t.dump();
                        assert(false && "Unsupported load type");
                        return false;
                    });
        }

        // TODO: this is veeeery brittle -> test!
        bool compile_func_call_op(mlir::func::CallOp op) {
            mlir::func::FuncOp callee_func = mlir::cast<mlir::func::FuncOp>(op.resolveCallable());

            // we only call into the runtime => use C-CallConv (yes, this is a hack since the runtime is actually C++ code. works for now)
            auto call_conv_assigner = tpde::x64::CCAssignerSysV(false /* is_vararg */); // TODO: this only works for x64
            auto builder = typename Derived::CallBuilder(*derived(), call_conv_assigner);

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
                    std::bit_cast<uint64_t>(externFuncMap[callee_func.getSymName()]), 8, Config::GP_BANK
                };
                builder.call(std::move(funcPtrRef));
            } else {
                assert(localFuncMap.contains(callee_func.getSymName()) && "Function not found in local function map");
                uint32_t func_idx = localFuncMap[callee_func.getSymName()];
                assert(func_idx < this->func_syms.size() && "Function index out of bounds");
                typename Base::Assembler::SymRef func_ref = this->func_syms[func_idx];
                builder.call(std::move(func_ref));
            }

            assert(op.getNumResults() <= 1 && "Function call must have exactly one result in the IR");
            if (op.getNumResults() != 0) {
                ValueRef res = this->result_ref(op.getResult(0));
                builder.add_ret(res);
            }
            return true;
        }

        bool compile_func_return_op(const mlir::func::ReturnOp op) {
            assert(!this->adaptor->cur_func.getResAttrs() && "we do not support return attributes yet");
            assert(op->getNumOperands() <= 1 && "Function return must have at most one operand in the IR");
            typename Derived::RetBuilder rb{*derived(), *derived()->cur_cc_assigner()};
            if (op->getNumOperands() != 0) {
                rb.add(op->getOperand(0));
            }
            rb.ret();
            return true;
        }

        // zext and sext operations
        bool compile_arith_exti_op(const auto op, bool sign) {
            mlir::Value src_val = op->getOperand(0);
            mlir::Value res_val = op->getResult(0);
            assert(
                mlir::isa<mlir::IntegerType>(src_val.getType()) &&
                "Source value must be an integer type for zext operation");
            unsigned src_width = src_val.getType().getIntOrFloatBitWidth();
            unsigned dst_width = res_val.getType().getIntOrFloatBitWidth();
            assert(
                src_width <= dst_width &&
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
                ScratchReg res_scratch_low{derived()};
                ScratchReg res_scratch_high{derived()};
                if (src_width != 64) return false;
                if (sign) {
                    derived()->encode_arith_sext_i64_i128(std::move(src_vpr), res_scratch_low, res_scratch_high);
                } else {
                    derived()->encode_arith_zext_i64_i128(std::move(src_vpr), res_scratch_low, res_scratch_high);
                }
                this->set_value(res.part(0), res_scratch_low);
                this->set_value(res.part(1), res_scratch_high);
            }
            return true;
        }

        bool compile_arith_select_op(const mlir::arith::SelectOp op) {
            mlir::Value cond = op->getOperand(0);
            mlir::Value lhs = op->getOperand(1);
            mlir::Value rhs = op->getOperand(2);
            auto [_, cond_vpr] = this->val_ref_single(cond);
            auto lhs_vr = this->val_ref(lhs);
            auto rhs_vr = this->val_ref(rhs);
            ScratchReg res_scratch{derived()};
            auto res = this->result_ref(op->getResult(0));

            switch (lhs.getType().getIntOrFloatBitWidth()) {
                case 1:
                case 8:
                case 16:
                case 32: derived()->encode_arith_select_i32(std::move(cond_vpr), lhs_vr.part(0), rhs_vr.part(0),
                                                            res_scratch);
                    break;
                case 64: derived()->encode_arith_select_i64(std::move(cond_vpr), lhs_vr.part(0), rhs_vr.part(0),
                                                            res_scratch);
                    break;
                case 128: {
                    ScratchReg res_scratch_high{derived()};
                    auto res_ref_high = res.part(1);
                    derived()->encode_arith_select_i128(std::move(cond_vpr),
                                                        lhs_vr.part(0),
                                                        lhs_vr.part(1),
                                                        rhs_vr.part(0),
                                                        rhs_vr.part(1),
                                                        res_scratch,
                                                        res_scratch_high);
                    this->set_value(res_ref_high, res_scratch_high);
                    break;
                }
                default:
                    assert(0 && "Unsupported integer type width for select operation");
                    return false;
            }
            auto res_ref = res.part(0);
            this->set_value(res_ref, res_scratch);
            return true;
        }

        bool compile_arith_index_cast_op(const mlir::arith::IndexCastOp op) {
            mlir::Value src = op->getOperand(0);
            mlir::Value res = op->getResult(0);
            if (mlir::isa<mlir::IntegerType>(src.getType()) && mlir::isa<mlir::IndexType>(res.getType())) {
                assert(src.getType().getIntOrFloatBitWidth() <= 64);
                auto [_, src_vpr] = this->val_ref_single(src);
                if (src.getType().getIntOrFloatBitWidth() != 64)
                    src_vpr = std::move(src_vpr).into_extended(false, src.getType().getIntOrFloatBitWidth(), 64);
                auto res_ref = this->result_ref(res);
                res_ref.part(0).set_value(std::move(src_vpr));
                return true;
            }
            if (mlir::isa<mlir::IndexType>(src.getType()) && mlir::isa<mlir::IntegerType>(res.getType())) {
                assert(res.getType().getIntOrFloatBitWidth() <= 64);
                auto [_, src_vpr] = this->val_ref_single(src);
                if (res.getType().getIntOrFloatBitWidth() != 64)
                    src_vpr = std::move(src_vpr).into_extended(false, 64, res.getType().getIntOrFloatBitWidth());
                auto res_ref = this->result_ref(res);
                res_ref.part(0).set_value(std::move(src_vpr));
                return true;
            }
            assert(0 && "Index cast operation must be between integer and index types");
            return false;
        }

        bool compile_util_buffer_get_len_op(dialect::util::BufferGetLen op) {
            auto elem_size = get_size(op.getBuffer().getType());
            if (!elem_size) {
                assert(0 && "Unsupported type for buffer length operation");
                error.emit() << "Unsupported type for buffer length operation.";
                return false;
            }
            // we store the length in the first 8 bytes of the buffer
            auto buf_vr = this->val_ref(op.getBuffer());
            ScratchReg res_scratch{derived()};
            ValuePart elem_size_vp{*elem_size, 8, Config::GP_BANK};
            derived()->encode_arith_udiv_i64(buf_vr.part(0),
                                             GenericValuePart{Expr{elem_size_vp.load_to_reg(derived())}},
                                             res_scratch);
            elem_size_vp.reset(derived()); // release the temporary register
            auto res_vr = this->result_ref(op.getResult());
            this->set_value(res_vr.part(0), res_scratch);
            return true;
        }

        bool compile_util_varlen_get_len_op(dialect::util::VarLenGetLen op) {
            auto varlen_vr = this->val_ref(op.getVarlen());
            auto res_vr = this->result_ref(op.getResult());
            res_vr.part(0).set_value(varlen_vr.part(0));
            return true;
        }

        bool compile_util_const_varlen_op(const dialect::util::CreateConstVarLen op) {
            const mlir::StringRef content = mlir::cast<mlir::StringAttr>(op->getAttrs().front().getValue()).
                    getValue();
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
                typename Base::Assembler::SymRef func_ref = this->func_syms[func_idx];
                derived()->load_address_of_got_sym(func_ref, res_vr.part(0).alloc_reg());
                return true;
            }
            if (externFuncMap.contains(funcName)) {
                res_vr.part(0).set_value(
                    ValuePart{std::bit_cast<uint64_t>(externFuncMap[funcName]), 8, Config::GP_BANK});
                return true;
            }
            error.emit() << "Function constant refers to neither a local nor an external function: " <<
                    funcName.str() << "\n";
            return false;
        }

        bool compile_util_array_element_ptr_op(dialect::util::ArrayElementPtrOp op) {
            const mlir::TypedValue<dialect::util::RefType> array_ptr = op.getRef();
            const mlir::TypedValue<mlir::IndexType> idx = op.getIdx();
            const mlir::Type elem_type = array_ptr.getType().getElementType();

            assert(val_parts(array_ptr).count() == 1);
            auto [array_ptr_vr, array_ptr_pr] = this->val_ref_single(array_ptr);

            GenericValuePart ptr;
            // store to [ptr + idx * elem_size]
            auto elem_size = get_size(elem_type);
            if (!elem_size) {
                assert(0 && "Unsupported type for store operation");
                error.emit() << "Unsupported type for store operation.";
                return false;
            }
            AsmReg array_ptr_reg = array_ptr_pr.load_to_reg();
            if (auto idx_op = mlir::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(idx.getDefiningOp())) {
                ptr = GenericValuePart{
                    Expr{
                        std::move(array_ptr_reg), static_cast<int64_t>(static_cast<size_t>(idx_op.value()) * *elem_size)
                    }
                };
            } else {
                auto [_, idx_ref] = this->val_ref_single(idx);
                auto generic_ptr_expr = Expr{std::move(array_ptr_reg)};
                generic_ptr_expr.index = idx_ref.alloc_reg();
                generic_ptr_expr.scale = *elem_size;
                ptr = GenericValuePart{std::move(generic_ptr_expr)};
            }

            // load value to register (e.g. mov + add / lea for x86_64)
            AsmReg res_reg = derived()->gval_expr_as_reg(ptr);
            ScratchReg res_scratch{derived()};
            derived()->mov(res_scratch.alloc_gp(), res_reg, 8);

            const auto dst = op.getRes();
            assert(val_parts(dst).count() == 1);
            auto [_, res_vr] = this->result_ref_single(dst);
            this->set_value(res_vr, res_scratch);
            return true;
        }

        bool compile_util_create_varlen_op(dialect::util::CreateVarLen op) {
            auto call_conv_assigner = tpde::x64::CCAssignerSysV(false /* is_vararg */); // TODO: this only works for x64
            auto builder = typename Derived::CallBuilder(*derived(), call_conv_assigner);
            builder.add_arg(typename Base::CallArg{op.getRef()});
            builder.add_arg(typename Base::CallArg{op.getLen()});
            assert(externFuncMap.contains("createVarLen32"));
            ValuePart funcPtrRef{
                std::bit_cast<uint64_t>(externFuncMap["createVarLen32"]), 8, Config::GP_BANK
            };
            builder.call(std::move(funcPtrRef));
            ValueRef res = this->result_ref(op.getVarlen());
            builder.add_ret(res);
            return true;
        }

        bool compile_util_varlen_cmp_op(dialect::util::VarLenCmp op) {
            auto lhs_vr = this->val_ref(op.getLeft());
            auto rhs_vr = this->val_ref(op.getRight());
            ScratchReg res_scratch{derived()};
            ScratchReg res_scratch_more{derived()};
            if (!derived()->encode_util_varlen_cmp(lhs_vr.part(0), lhs_vr.part(1),
                                                   rhs_vr.part(0), rhs_vr.part(1), res_scratch, res_scratch_more)) {
                return false;
            }

            auto eq_res_vr = this->result_ref(op.getEq());
            this->set_value(eq_res_vr.part(0), res_scratch);

            auto more_cmp_res_vr = this->result_ref(op.getNeedsDetailedEval());
            this->set_value(more_cmp_res_vr.part(0), res_scratch_more);
            return true;
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

            using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ScratchReg &);
            EncodeFnTy fn = nullptr;
            if (cmp_type.isF32()) {
                switch (pred) {
                    case CmpFPredicate::OEQ: fn = &Derived::encode_arith_cmp_f32_oeq;
                        break;
                    case CmpFPredicate::OLT: fn = &Derived::encode_arith_cmp_f32_olt;
                        break;
                    case CmpFPredicate::OLE: fn = &Derived::encode_arith_cmp_f32_ole;
                        break;
                    case CmpFPredicate::OGT: fn = &Derived::encode_arith_cmp_f32_ogt;
                        break;
                    case CmpFPredicate::OGE: fn = &Derived::encode_arith_cmp_f32_oge;
                        break;
                    case CmpFPredicate::ONE: fn = &Derived::encode_arith_cmp_f32_one;
                        break;
                    case CmpFPredicate::ORD: fn = &Derived::encode_arith_cmp_f32_ord;
                        break;
                    case CmpFPredicate::UNO: fn = &Derived::encode_arith_cmp_f32_uno;
                        break;
                    case CmpFPredicate::UEQ: fn = &Derived::encode_arith_cmp_f32_ueq;
                        break;
                    case CmpFPredicate::ULT: fn = &Derived::encode_arith_cmp_f32_ult;
                        break;
                    case CmpFPredicate::ULE: fn = &Derived::encode_arith_cmp_f32_ule;
                        break;
                    case CmpFPredicate::UGT: fn = &Derived::encode_arith_cmp_f32_ugt;
                        break;
                    case CmpFPredicate::UGE: fn = &Derived::encode_arith_cmp_f32_uge;
                        break;
                    case CmpFPredicate::UNE: fn = &Derived::encode_arith_cmp_f32_une;
                        break;
                    default: assert(0);
                }
            } else {
                switch (pred) {
                    case CmpFPredicate::OEQ: fn = &Derived::encode_arith_cmp_f64_oeq;
                        break;
                    case CmpFPredicate::OLT: fn = &Derived::encode_arith_cmp_f64_olt;
                        break;
                    case CmpFPredicate::OLE: fn = &Derived::encode_arith_cmp_f64_ole;
                        break;
                    case CmpFPredicate::OGT: fn = &Derived::encode_arith_cmp_f64_ogt;
                        break;
                    case CmpFPredicate::OGE: fn = &Derived::encode_arith_cmp_f64_oge;
                        break;
                    case CmpFPredicate::ONE: fn = &Derived::encode_arith_cmp_f64_one;
                        break;
                    case CmpFPredicate::ORD: fn = &Derived::encode_arith_cmp_f64_ord;
                        break;
                    case CmpFPredicate::UNO: fn = &Derived::encode_arith_cmp_f64_uno;
                        break;
                    case CmpFPredicate::UEQ: fn = &Derived::encode_arith_cmp_f64_ueq;
                        break;
                    case CmpFPredicate::ULT: fn = &Derived::encode_arith_cmp_f64_ult;
                        break;
                    case CmpFPredicate::ULE: fn = &Derived::encode_arith_cmp_f64_ule;
                        break;
                    case CmpFPredicate::UGT: fn = &Derived::encode_arith_cmp_f64_ugt;
                        break;
                    case CmpFPredicate::UGE: fn = &Derived::encode_arith_cmp_f64_uge;
                        break;
                    case CmpFPredicate::UNE: fn = &Derived::encode_arith_cmp_f64_une;
                        break;
                    default: assert(0);
                }
            }
            ValueRef lhs_vr = this->val_ref(lhs);
            ValueRef rhs_vr = this->val_ref(rhs);
            ScratchReg res_scratch{derived()};
            auto [res_vr, res_ref] = this->result_ref_single(res);
            if (!(derived()->*fn)(lhs_vr.part(0), rhs_vr.part(0), res_scratch)) { return false; }
            this->set_value(res_ref, res_scratch);
            return true;
        }

        bool compile_util_hash_64_op(dialect::util::Hash64 op) {
            auto [_, val_pr] = this->val_ref_single(op.getVal());
            ScratchReg res_scratch{this};
            if (!derived()->encode_util_hash_64(std::move(val_pr), res_scratch)) {
                error.emit() << "Failed to compile hash operation\n";
                return false;
            }
            auto res_pr = this->result_ref(op.getResult());
            this->set_value(res_pr.part(0), res_scratch);
            return true;
        }

        bool compile_util_ptr_tag_matches_op(dialect::util::PtrTagMatches op) {
            auto hash_vr = this->val_ref(op.getHash());
            auto ref_vr = this->val_ref(op.getRef());

            assert(externFuncMap.contains("bloomMasks"));
            ValuePartRef bloom_masks_array_vpr{
                this, std::bit_cast<uint64_t>(externFuncMap["bloomMasks"]), 8, Config::GP_BANK
            };

            ScratchReg res_scratch{derived()};
            derived()->encode_util_ptr_tag_matches(ref_vr.part(0), hash_vr.part(0), std::move(bloom_masks_array_vpr),
                                                   res_scratch);
            auto res_ref = this->result_ref(op.getMatches());
            this->set_value(res_ref.part(0), res_scratch);
            return true;
        }

        bool compile_util_untag_ptr_op(dialect::util::UnTagPtr op) {
            auto ref_vr = this->val_ref(op.getRef());

            ScratchReg res_scratch{derived()};
            derived()->encode_util_untag_ptr(ref_vr.part(0), res_scratch);
            auto res_ref = this->result_ref(op.getRes());
            this->set_value(res_ref.part(0), res_scratch);
            return true;
        }

        bool compile_util_is_ref_valid_op(dialect::util::IsRefValidOp op) {
            auto ref_vr = this->val_ref(op.getRef());
            ScratchReg res_scratch{derived()};
            derived()->encode_util_is_ref_valid(ref_vr.part(0), res_scratch);
            auto res_vr = this->result_ref(op.getValid());
            this->set_value(res_vr.part(0), res_scratch);
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
            ScratchReg res_scratch{derived()};
            if (!derived()->encode_util_hash_combine(h1_vr.part(0), h2_vr.part(0), res_scratch)) {
                error.emit() << "Failed to compile hash combine operation\n";
                return false;
            }
            auto res_ref = this->result_ref(op.getResult());
            this->set_value(res_ref.part(0), res_scratch);
            return true;
        }

        bool compile_inst(const IRInstRef inst, InstRange) noexcept {
            dialect::util::CreateConstVarLen
            return mlir::TypeSwitch<IRInstRef, bool>(inst)
                    .Case<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp, mlir::arith::DivSIOp,
                        mlir::arith::DivUIOp, mlir::arith::RemSIOp, mlir::arith::RemUIOp,
                        mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp, mlir::arith::ShLIOp,
                        mlir::arith::ShRUIOp, mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                        mlir::arith::DivFOp>([&](auto op) { return compile_arith_binary_op(op); })
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
                    .template Case<dialect::util::VarLenCmp>([&](auto op) { return compile_util_varlen_cmp_op(op); })
                    .Default([&](IRInstRef op) {
                        error.emit() << "Encountered unimplemented instruction: " << op->getName().getStringRef().
                                str()
                                << "\n";
                        op->dump();
                        return false;
                    });
        }
    };

    // NOLINTEND(readability-identifier-naming)

    // x86_64 target specific compiler
    // NOLINTBEGIN(readability-identifier-naming)
    struct IRCompilerX64
            : tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>,
              tpde_encodegen::EncodeCompiler<IRAdaptor, IRCompilerX64, IRCompilerBase,
                  CompilerConfig> {
        using Base = tpde::x64::CompilerX64<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;
        using EncCompiler = tpde_encodegen::EncodeCompiler<IRAdaptor, IRCompilerX64, IRCompilerBase, CompilerConfig>;

        std::unique_ptr<IRAdaptor> adaptor;

        explicit IRCompilerX64(std::unique_ptr<IRAdaptor> &&adaptor)
            : Base{adaptor.get()},
              adaptor(std::move(adaptor)) { static_assert(tpde::Compiler<IRCompilerX64, tpde::x64::PlatformConfig>); }

        void load_address_of_global(const SymRef global_sym, const AsmReg dst) {
            ASM(MOV64rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
            reloc_text(global_sym, R_X86_64_PC32, text_writer.offset() - 4, -4);
        }

        void create_helper_call(std::span<IRValueRef> args, ValueRef *result, SymRef sym) noexcept {
            tpde::util::SmallVector<CallArg, 8> arg_vec{};
            for (const auto arg: args) { arg_vec.push_back(CallArg{arg}); }
            generate_call(sym, arg_vec, result);
        }

        bool compile_arith_cmp_int_op(mlir::arith::CmpIOp op) {
            mlir::Type ty = op.getLhs().getType();
            unsigned int_width;
            if (mlir::isa<mlir::IntegerType>(ty)) { int_width = ty.getIntOrFloatBitWidth(); } else if (mlir::isa<
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
                case mlir::arith::CmpIPredicate::eq: jump = Jump::je;
                    break;
                case mlir::arith::CmpIPredicate::ne: jump = Jump::jne;
                    break;
                case mlir::arith::CmpIPredicate::ugt: jump = Jump::jg;
                    break;
                case mlir::arith::CmpIPredicate::uge: jump = Jump::jge;
                    break;
                case mlir::arith::CmpIPredicate::ult: jump = Jump::jl;
                    break;
                case mlir::arith::CmpIPredicate::ule: jump = Jump::jle;
                    break;
                case mlir::arith::CmpIPredicate::sgt: jump = Jump::jg;
                    break;
                case mlir::arith::CmpIPredicate::sge: jump = Jump::jge;
                    break;
                case mlir::arith::CmpIPredicate::slt: jump = Jump::jl;
                    break;
                case mlir::arith::CmpIPredicate::sle: jump = Jump::jle;
                    break;
                default: assert(0);
                    return false;
            }
            switch (op.getPredicate()) {
                case mlir::arith::CmpIPredicate::sgt:
                case mlir::arith::CmpIPredicate::sge:
                case mlir::arith::CmpIPredicate::slt:
                case mlir::arith::CmpIPredicate::sle: is_signed = true;
                    break;
                default: break;
            }

            // TODO: check if the result can be fused with a subsequent br instruction
            auto lhs = this->val_ref(op.getLhs());
            auto rhs = this->val_ref(op.getRhs());
            ScratchReg res_scratch{this};

            if (int_width == 128) {
                if ((jump == Jump::ja) || (jump == Jump::jbe) || (jump == Jump::jle) ||
                    (jump == Jump::jg)) {
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
                        case 32: ASM(CMP32ri, lhs_reg, static_cast<uint32_t>(rhs_const));
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
                        default: assert(0);
                            return false;
                    }
                } else {
                    auto rhs_reg = rhs_pr.load_to_reg();
                    switch (int_width) {
                        case 1:
                        case 8:
                        case 16:
                        case 32: ASM(CMP32rr, lhs_reg, rhs_reg);
                            break;
                        case 64: ASM(CMP64rr, lhs_reg, rhs_reg);
                            break;
                        default: assert(0);
                            return false;
                    }
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
            return true;
        }

        void load_address_of_got_sym(const SymRef sym, const AsmReg dst) noexcept {
            assert(sym.valid());
            ASM(MOV64rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
            reloc_text(sym, R_X86_64_GOTPCREL, text_writer.offset() - 4, -4);
        }

        void reset() noexcept {
            Base::reset();
            EncCompiler::reset();
        }

        Error &getError() { return Base::getError(); }
    };

    // NOLINTEND(readability-identifier-naming)

    class DynamicLoader {
    protected:
        Error &error;

    public:
        DynamicLoader(Error &error)
            : error(error) {
        }

        virtual ~DynamicLoader() = default;

        virtual void teardown() {
        }

        virtual mainFnType getMainFunction() { return nullptr; }
        bool has_error = false;
    };

    template<typename Assembler>
    class InMemoryLoader final : public DynamicLoader {
        tpde::ElfMapper mapper;
        llvm::DenseMap<const llvm::GlobalValue *, tpde::AssemblerElfBase::SymRef> globals;
        tpde::AssemblerElfBase::SymRef main_func;

    public:
        InMemoryLoader(Assembler &assembler, Error &error, typename Assembler::SymRef main_func)
            : DynamicLoader(error),
              main_func(main_func) {
            mapper.map(assembler, [](const std::string_view name) {
                return dlsym(RTLD_DEFAULT, std::string(name).c_str());
            });
        }

        mainFnType getMainFunction() override {
            return reinterpret_cast<mainFnType>(mapper.get_sym_addr(main_func));
        }
    };

    template<typename Assembler>
    class DebugLoader final : public DynamicLoader {
        void *handle = nullptr;

    public:
        DebugLoader(Assembler &assembler, Error &error, const std::string_view outFileName)
            : DynamicLoader(error) {
            const auto objFile = assembler.build_object_file();
            const std::string objFileName = std::string{outFileName} + ".o";
            const std::string linkedFileName = std::string{outFileName} + ".so";
            auto *outFile = std::fopen((std::string{outFileName} + ".o").c_str(), "wb");
            if (!outFile) {
                error.emit() << "Could not open output file for baseline object: " << objFileName << " (" <<
                        strerror(errno) << ")\n";
                has_error = true;
                return;
            }
            if (std::fwrite(objFile.data(), 1, objFile.size(), outFile) != objFile.size()) {
                error.emit() << "Could not write object file to output file: " << objFileName << " (" << strerror(
                            errno)
                        << ")\n";
                has_error = true;
                return;
            }
            if (std::fclose(outFile) != 0) {
                error.emit() << "Could not close output file: " << objFileName << " (" << strerror(errno) << ")\n";
                has_error = true;
                return;
            }
            std::string cmd = std::string("cc -shared -o ") + linkedFileName + " " + objFileName;
            auto *pPipe = ::popen(cmd.c_str(), "r");
            if (pPipe == nullptr) {
                has_error = true;
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
                has_error = true;
                error.emit() << "Could not compile query module statically (Pipe could not be closed)";
                return;
            }
            handle = dlopen(linkedFileName.c_str(), RTLD_LAZY);
            if (const char *dlsymError = dlerror()) {
                has_error = true;
                error.emit() << "Cannot open object file: " << std::string(dlsymError) << "\nerror: " <<
                        strerror(errno) << "\n";
                return;
            }
        }

        mainFnType getMainFunction() override {
            const auto mainFunc = reinterpret_cast<mainFnType>(dlsym(handle, "main"));
            if (const char *dlsymError = dlerror()) {
                error.emit() << "Could not load symbol for main function: " << std::string(dlsymError) << "\nerror:"
                        << strerror(errno) << "\n";
                has_error = true;
                return nullptr;
            }
            return mainFunc;
        }

        void teardown() override {
            dlclose(handle);
        }
    };

    class BaselineBackend : public lingodb::execution::ExecutionBackend {
        // lower mlir IR to a form that can be compiled by tpde
        // currently mostly does a SCF to CF conversion
        bool lower(mlir::ModuleOp &moduleOp,
                   std::shared_ptr<lingodb::execution::SnapshotState> serializationState) {
            mlir::PassManager pm2(moduleOp->getContext());
            pm2.enableVerifier(verify);
            lingodb::execution::addLingoDBInstrumentation(pm2, serializationState);
            pm2.addPass(mlir::createConvertSCFToCFPass());
            pm2.addPass(createDecomposeTuplePass());
            pm2.addPass(mlir::createCanonicalizerPass());
            pm2.addPass(mlir::createCSEPass()); // TODO: evaluate whether we need this
            if (mlir::failed(pm2.run(moduleOp))) {
                return false;
            }
            return true;
        }

        void execute(mlir::ModuleOp &moduleOp, lingodb::runtime::ExecutionContext *executionContext) override {
            auto startLowering = std::chrono::high_resolution_clock::now();
            if (!lower(moduleOp, getSerializationState())) {
                error.emit() << "Could not lower module for baseline compilation";
                return;
            }
            auto endLowering = std::chrono::high_resolution_clock::now();
            timing["baselineLowering"] = std::chrono::duration_cast<std::chrono::microseconds>(
                                             endLowering - startLowering).count() / 1000.0;

            SpdLogSpoof logSpoof;
#if defined(__x86_64__)
            IRCompilerX64 compiler{std::make_unique<IRAdaptor>(&moduleOp, error)};
#else
#error "Baseline backend is only supported on x86_64 architecture."
#endif
            const auto baselineCodeGenStart = std::chrono::high_resolution_clock::now();
            if (!compiler.compile()) {
                error.emit() << "Could not compile query module:\n"
                        // << logSpoof.drain_logs() << "\n"
                        << compiler.adaptor->getError().emit().str() << "\n"
                        << compiler.getError().emit().str() << "\n";
                return;
            }
            const auto baselineCodeGenEnd = std::chrono::high_resolution_clock::now();
            const auto baselineEmitStart = std::chrono::high_resolution_clock::now();
            std::unique_ptr<DynamicLoader> loader;
            if (!baselineDebugFileOut.getValue().empty()) {
                loader = std::make_unique<DebugLoader<IRCompilerX64::Assembler> >(compiler.assembler, error,
                    baselineDebugFileOut.getValue());
            } else {
                if (!compiler.localFuncMap.contains("main")) {
                    error.emit() << "No main function found in query module. Please ensure that the module has a "
                            "function named 'main'.\n";
                    return;
                }
                const uint32_t mainFuncIdx = compiler.localFuncMap["main"];
                if (mainFuncIdx >= compiler.func_syms.size()) {
                    error.emit() << "Main function index out of bounds: " << mainFuncIdx << " >= " <<
                            compiler.func_syms.size() << "\n";
                    return;
                }
                loader = std::make_unique<InMemoryLoader<IRCompilerX64::Assembler> >(
                    compiler.assembler, error, compiler.func_syms[mainFuncIdx]);
            }
            if (loader->has_error) return;
            auto mainFunc = loader->getMainFunction();
            if (loader->has_error) return;
            const auto baselineEmitEnd = std::chrono::high_resolution_clock::now();

            const auto executionStart = std::chrono::high_resolution_clock::now();
            utility::Tracer::Trace trace(execution);
            mainFunc();
            trace.stop();
            const auto executionEnd = std::chrono::high_resolution_clock::now();
            loader->teardown();

            timing["baselineCodeGen"] = std::chrono::duration_cast<std::chrono::microseconds>(
                                            baselineCodeGenEnd - baselineCodeGenStart).count() / 1000.0;
            timing["baselineEmit"] = std::chrono::duration_cast<std::chrono::microseconds>(
                                         baselineEmitEnd - baselineEmitStart).count() / 1000.0;
            timing["executionTime"] = std::chrono::duration_cast<std::chrono::microseconds>(
                                          executionEnd - executionStart).count() / 1000.0;
        }
    };
}

std::unique_ptr<lingodb::execution::ExecutionBackend> lingodb::execution::createBaselineBackend() {
    return std::make_unique<baseline::BaselineBackend>();
}
#endif
