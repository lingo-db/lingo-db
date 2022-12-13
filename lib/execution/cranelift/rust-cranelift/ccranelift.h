#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// CPU flags representing the result of an integer comparison. These flags
/// can be tested with an :u8:`intcc` condition code.
constexpr static const uint8_t TypeIFLAGS = 1;

/// CPU flags representing the result of a floating point comparison. These
/// flags can be tested with a :u8:`floatcc` condition code.
constexpr static const uint8_t TypeFFLAGS = 2;

/// A boolean u8 with 1 bits.
constexpr static const uint8_t TypeB1 = 112;

/// A boolean u8 with 8 bits.
constexpr static const uint8_t TypeB8 = 113;

/// A boolean u8 with 16 bits.
constexpr static const uint8_t TypeB16 = 114;

/// A boolean u8 with 32 bits.
constexpr static const uint8_t TypeB32 = 115;

/// A boolean u8 with 64 bits.
constexpr static const uint8_t TypeB64 = 116;

/// A boolean u8 with 128 bits.
constexpr static const uint8_t TypeB128 = 117;

/// An integer u8 with 8 bits.
/// WARNING: arithmetic on 8bit integers is incomplete
constexpr static const uint8_t TypeI8 = 118;

/// An integer u8 with 16 bits.
/// WARNING: arithmetic on 16bit integers is incomplete
constexpr static const uint8_t TypeI16 = 119;

/// An integer u8 with 32 bits.
constexpr static const uint8_t TypeI32 = 120;

/// An integer u8 with 64 bits.
constexpr static const uint8_t TypeI64 = 121;

/// An integer u8 with 128 bits.
constexpr static const uint8_t TypeI128 = 122;

/// A 32-bit floating point u8 represented in the IEEE 754-2008
/// *binary32* interchange format. This corresponds to the :c:u8:`float`
/// u8 in most C implementations.
constexpr static const uint8_t TypeF32 = 123;

/// A 64-bit floating point u8 represented in the IEEE 754-2008
/// *binary64* interchange format. This corresponds to the :c:u8:`double`
/// u8 in most C implementations.
constexpr static const uint8_t TypeF64 = 124;

/// An opaque reference u8 with 32 bits.
constexpr static const uint8_t TypeR32 = 126;

/// An opaque reference u8 with 64 bits.
constexpr static const uint8_t TypeR64 = 127;

/// A SIMD vector with 8 lanes containing a `b8` each.
constexpr static const uint8_t TypeB8X8 = 161;

/// A SIMD vector with 4 lanes containing a `b16` each.
constexpr static const uint8_t TypeB16X4 = 146;

/// A SIMD vector with 2 lanes containing a `b32` each.
constexpr static const uint8_t TypeB32X2 = 131;

/// A SIMD vector with 8 lanes containing a `i8` each.
constexpr static const uint8_t TypeI8X8 = 166;

/// A SIMD vector with 4 lanes containing a `i16` each.
constexpr static const uint8_t TypeI16X4 = 151;

/// A SIMD vector with 2 lanes containing a `i32` each.
constexpr static const uint8_t TypeI32X2 = 136;

/// A SIMD vector with 2 lanes containing a `f32` each.
constexpr static const uint8_t TypeF32X2 = 139;

/// A SIMD vector with 16 lanes containing a `b8` each.
constexpr static const uint8_t TypeB8X16 = 177;

/// A SIMD vector with 8 lanes containing a `b16` each.
constexpr static const uint8_t TypeB16X8 = 162;

/// A SIMD vector with 4 lanes containing a `b32` each.
constexpr static const uint8_t TypeB32X4 = 147;

/// A SIMD vector with 2 lanes containing a `b64` each.
constexpr static const uint8_t TypeB64X2 = 132;

/// A SIMD vector with 16 lanes containing a `i8` each.
constexpr static const uint8_t TypeI8X16 = 182;

/// A SIMD vector with 8 lanes containing a `i16` each.
constexpr static const uint8_t TypeI16X8 = 167;

/// A SIMD vector with 4 lanes containing a `i32` each.
constexpr static const uint8_t TypeI32X4 = 152;

/// A SIMD vector with 2 lanes containing a `i64` each.
constexpr static const uint8_t TypeI64X2 = 137;

/// A SIMD vector with 4 lanes containing a `f32` each.
constexpr static const uint8_t TypeF32X4 = 155;

/// A SIMD vector with 2 lanes containing a `f64` each.
constexpr static const uint8_t TypeF64X2 = 140;

/// A SIMD vector with 32 lanes containing a `b8` each.
constexpr static const uint8_t TypeB8X32 = 193;

/// A SIMD vector with 16 lanes containing a `b16` each.
constexpr static const uint8_t TypeB16X16 = 178;

/// A SIMD vector with 8 lanes containing a `b32` each.
constexpr static const uint8_t TypeB32X8 = 163;

/// A SIMD vector with 4 lanes containing a `b64` each.
constexpr static const uint8_t TypeB64X4 = 148;

/// A SIMD vector with 2 lanes containing a `b128` each.
constexpr static const uint8_t TypeB128X2 = 133;

/// A SIMD vector with 32 lanes containing a `i8` each.
constexpr static const uint8_t TypeI8X32 = 198;

/// A SIMD vector with 16 lanes containing a `i16` each.
constexpr static const uint8_t TypeI16X16 = 183;

/// A SIMD vector with 8 lanes containing a `i32` each.
constexpr static const uint8_t TypeI32X8 = 168;

/// A SIMD vector with 4 lanes containing a `i64` each.
constexpr static const uint8_t TypeI64X4 = 153;

/// A SIMD vector with 2 lanes containing a `i128` each.
constexpr static const uint8_t TypeI128X2 = 138;

/// A SIMD vector with 8 lanes containing a `f32` each.
constexpr static const uint8_t TypeF32X8 = 171;

/// A SIMD vector with 4 lanes containing a `f64` each.
constexpr static const uint8_t TypeF64X4 = 156;

/// A SIMD vector with 64 lanes containing a `b8` each.
constexpr static const uint8_t TypeB8X64 = 209;

/// A SIMD vector with 32 lanes containing a `b16` each.
constexpr static const uint8_t TypeB16X32 = 194;

/// A SIMD vector with 16 lanes containing a `b32` each.
constexpr static const uint8_t TypeB32X16 = 179;

/// A SIMD vector with 8 lanes containing a `b64` each.
constexpr static const uint8_t TypeB64X8 = 164;

/// A SIMD vector with 4 lanes containing a `b128` each.
constexpr static const uint8_t TypeB128X4 = 149;

/// A SIMD vector with 64 lanes containing a `i8` each.
constexpr static const uint8_t TypeI8X64 = 214;

/// A SIMD vector with 32 lanes containing a `i16` each.
constexpr static const uint8_t TypeI16X32 = 199;

/// A SIMD vector with 16 lanes containing a `i32` each.
constexpr static const uint8_t TypeI32X16 = 184;

/// A SIMD vector with 8 lanes containing a `i64` each.
constexpr static const uint8_t TypeI64X8 = 169;

/// A SIMD vector with 4 lanes containing a `i128` each.
constexpr static const uint8_t TypeI128X4 = 154;

/// A SIMD vector with 16 lanes containing a `f32` each.
constexpr static const uint8_t TypeF32X16 = 187;

/// A SIMD vector with 8 lanes containing a `f64` each.
constexpr static const uint8_t TypeF64X8 = 172;

/// The current stack space was exhausted.
///
/// On some platforms, a stack overflow may also be indicated by a segmentation fault from the
/// stack guard page.
constexpr static const uint32_t TrapCodeStackOverflow = (1 << 16);

/// A `heap_addr` instruction detected an out-of-bounds error.
///
/// Note that not all out-of-bounds heap accesses are reported this way;
/// some are detected by a segmentation fault on the heap unmapped or
/// offset-guard pages.
constexpr static const uint32_t TrapCodeHeapOutOfBounds = (2 << 16);

/// A `table_addr` instruction detected an out-of-bounds error.
constexpr static const uint32_t TrapCodeTableOutOfBounds = (3 << 16);

/// Indirect call to a null table entry.
constexpr static const uint32_t TrapCodeIndirectCallToNull = (5 << 16);

/// Signature mismatch on indirect call.
constexpr static const uint32_t TrapCodeBadSignature = (6 << 16);

/// An integer arithmetic operation caused an overflow.
constexpr static const uint32_t TrapCodeIntegerOverflow = (7 << 16);

/// An integer division by zero.
constexpr static const uint32_t TrapCodeIntegerDivisionByZero = (8 << 16);

/// Failed float-to-int conversion.
constexpr static const uint32_t TrapCodeBadConversionToInteger = (9 << 16);

/// Code that was supposed to have been unreachable was reached.
constexpr static const uint32_t TrapCodeUnreachableCodeReached = (10 << 16);

/// Execution has potentially run too long and may be interrupted.
/// This trap is resumable.
constexpr static const uint32_t TrapCodeInterrupt = (11 << 16);

constexpr static const uint8_t MemFlagNoTrap = 1;

constexpr static const uint8_t MemFlagAligned = 2;

constexpr static const uint8_t MemFlagReadonly = 4;

enum class CraneliftCallConv : uint32_t {
  CraneliftCallConvDefault = 4294967295,
  CraneliftCallConvFast = 0,
  CraneliftCallConvCold = 1,
  CraneliftCallConvSystemV = 2,
  CraneliftCallConvWindowsFastcall = 3,
  CraneliftCallConvBaldrdashSystemV = 4,
  CraneliftCallConvBaldrdashWindows = 5,
  CraneliftCallConvProbestack = 6,
};

enum class CraneliftDataFlags : uint32_t {
  None = 0,
  TLS = 1,
  Writable = 2,
};

enum class CraneliftFloatCC : uint32_t {
  CraneliftFloatCCOrdered = 0,
  CraneliftFloatCCUnordered = 1,
  CraneliftFloatCCEqual = 2,
  CraneliftFloatCCNotEqual = 3,
  CraneliftFloatCCOrderedNotEqual = 4,
  CraneliftFloatCCUnorderedOrEqual = 5,
  CraneliftFloatCCLessThan = 6,
  CraneliftFloatCCLessThanOrEqual = 7,
  CraneliftFloatCCGreaterThan = 8,
  CraneliftFloatCCGreaterThanOrEqual = 9,
  CraneliftFloatCCUnorderedOrLessThan = 10,
  CraneliftFloatCCUnorderedOrLessThanOrEqual = 11,
  CraneliftFloatCCUnorderedOrGreaterThan = 12,
  CraneliftFloatCCUnorderedOrGreaterThanOrEqual = 13,
};

enum class CraneliftIntCC : uint32_t {
  CraneliftIntCCEqual = 0,
  CraneliftIntCCNotEqual = 1,
  CraneliftIntCCSignedLessThan = 2,
  CraneliftIntCCSignedGreaterThanOrEqual = 3,
  CraneliftIntCCSignedGreaterThan = 4,
  CraneliftIntCCSignedLessThanOrEqual = 5,
  CraneliftIntCCUnsignedLessThan = 6,
  CraneliftIntCCUnsignedGreaterThanOrEqual = 7,
  CraneliftIntCCUnsignedGreaterThan = 8,
  CraneliftIntCCUnsignedLessThanOrEqual = 9,
  CraneliftIntCCOverflow = 10,
  CraneliftIntCCNotOverflow = 11,
};

enum class CraneliftLinkage : uint32_t {
  Import = 0,
  Local = 1,
  Preemptible = 2,
  Hidden = 3,
  Export = 4,
};

struct FunctionData;

struct ModuleData;

using Type = uint8_t;

using BlockCode = uint32_t;

using ValueCode = uint32_t;

using VariableCode = uint32_t;

using ValueLabelCode = uint32_t;

using JumpTableCode = uint32_t;

using SigRefCode = uint32_t;

using FuncRefCode = uint32_t;

using InstCode = uint32_t;

using TrapCode = uint32_t;

using Uimm8 = uint8_t;

using MemFlagCode = uint8_t;

using Offset32 = int32_t;

using StackSlotCode = uint32_t;

using GlobalValueCode = uint32_t;

using HeapCode = uint32_t;

using Uimm32 = uint32_t;

using TableCode = uint32_t;

using Imm64 = int64_t;

using ConstantCode = uint32_t;

using ImmediateCode = uint32_t;

extern "C" {

void rust_function();

ModuleData *cranelift_module_new(const char *target_triple,
                                 const char *flags,
                                 const char *name,
                                 uintptr_t userdata,
                                 void (*error_cb)(uintptr_t userdata, const char*, const char*),
                                 void (*message_cb)(uintptr_t userdata, const char*, const char*));

bool cranelift_define_data(ModuleData *ptr,
                           const char *name,
                           CraneliftLinkage linkage,
                           CraneliftDataFlags data_flags,
                           uint8_t align,
                           uint32_t *id);

bool cranelift_declare_function(ModuleData *ptr,
                                const char *name,
                                CraneliftLinkage linkage,
                                uint32_t *id);

int32_t cranelift_define_function(ModuleData *ptr, uint32_t func);

bool cranelift_set_data_value(ModuleData *ptr, const uint8_t *content, int32_t length);

void cranelift_set_data_section(ModuleData *ptr, const char *seg, const char *sec);

void cranelift_clear_data(ModuleData *ptr);

bool cranelift_assign_data_to_global(ModuleData *ptr, uint32_t id);

const uint8_t *cranelift_jit(ModuleData *ptr, uint32_t func);

void cranelift_module_delete(ModuleData *ptr);

Type cranelift_get_pointer_type(ModuleData *ptr);

uint8_t cranelift_get_pointer_size_bytes(ModuleData *ptr);

void cranelift_clear_context(ModuleData *ptr);

void cranelift_signature_builder_reset(ModuleData *ptr, CraneliftCallConv cc);

void cranelift_signature_builder_add_param(ModuleData *ptr, Type typ);

void cranelift_signature_builder_add_result(ModuleData *ptr, Type typ);

void cranelift_build_function(ModuleData *ptr,
                              uintptr_t userdata,
                              void (*cb)(uintptr_t userdata, FunctionData *builder));

void cranelift_function_to_string(ModuleData *ptr,
                                  uintptr_t userdata,
                                  void (*cb)(uintptr_t userdata, const char *str));

void cranelift_set_source_loc(FunctionData *ptr, uint32_t loc);

BlockCode cranelift_create_block(FunctionData *ptr);

void cranelift_switch_to_block(FunctionData *ptr, BlockCode block);

void cranelift_seal_block(FunctionData *ptr, BlockCode block);

void cranelift_seal_all_blocks(FunctionData *ptr);

void cranelift_append_block_params_for_function_params(FunctionData *ptr, BlockCode block);

void cranelift_append_block_params_for_function_returns(FunctionData *ptr, BlockCode block);

int32_t cranelift_block_params_count(FunctionData *ptr, BlockCode block);

void cranelift_block_params(FunctionData *ptr, BlockCode block, ValueCode *dest);

VariableCode cranelift_declare_var(FunctionData *ptr, Type typ);

void cranelift_def_var(FunctionData *ptr, VariableCode var, ValueCode val);

ValueCode cranelift_use_var(FunctionData *ptr, VariableCode var);

void cranelift_set_val_label(FunctionData *ptr, ValueCode val, ValueLabelCode label);

JumpTableCode cranelift_create_jump_table(FunctionData *ptr, uint32_t count, BlockCode *targets);

SigRefCode cranelift_import_signature(FunctionData *ptr,
                                      CraneliftCallConv cc,
                                      uint32_t argscount,
                                      Type *args,
                                      uint32_t retcount,
                                      Type *rets);

FuncRefCode cranelift_declare_func_in_current_func(FunctionData *ptr, uint32_t source_id);

InstCode cranelift_ins_jump(FunctionData *ptr, BlockCode block, uint32_t count, ValueCode *args);

InstCode cranelift_ins_brz(FunctionData *ptr,
                           ValueCode c,
                           BlockCode block,
                           uint32_t count,
                           ValueCode *args);

InstCode cranelift_ins_brnz(FunctionData *ptr,
                            ValueCode c,
                            BlockCode block,
                            uint32_t count,
                            ValueCode *args);

InstCode cranelift_ins_br_icmp(FunctionData *ptr,
                               CraneliftIntCC cond,
                               ValueCode x,
                               ValueCode y,
                               BlockCode block,
                               uint32_t count,
                               ValueCode *args);

InstCode cranelift_ins_brif(FunctionData *ptr,
                            CraneliftIntCC cond,
                            ValueCode c,
                            BlockCode block,
                            uint32_t count,
                            ValueCode *args);

InstCode cranelift_ins_brff(FunctionData *ptr,
                            CraneliftFloatCC cond,
                            ValueCode c,
                            BlockCode block,
                            uint32_t count,
                            ValueCode *args);

InstCode cranelift_ins_br_table(FunctionData *ptr, ValueCode x, BlockCode block, JumpTableCode jt);

InstCode cranelift_ins_trap(FunctionData *ptr, TrapCode code);

InstCode cranelift_ins_resumable_trap(FunctionData *ptr, TrapCode code);

InstCode cranelift_ins_trapz(FunctionData *ptr, ValueCode c, TrapCode code);

InstCode cranelift_ins_trapnz(FunctionData *ptr, ValueCode v, TrapCode code);

InstCode cranelift_ins_trapif(FunctionData *ptr, CraneliftIntCC cond, ValueCode f, TrapCode code);

InstCode cranelift_ins_trapff(FunctionData *ptr, CraneliftFloatCC cond, ValueCode f, TrapCode code);

InstCode cranelift_return(FunctionData *ptr, uint32_t count, ValueCode *args);

InstCode cranelift_fallthrough_return(FunctionData *ptr, uint32_t count, ValueCode *args);

InstCode cranelift_call(FunctionData *ptr, FuncRefCode fnn, uint32_t count, ValueCode *args);

InstCode cranelift_call_indirect(FunctionData *ptr,
                                 SigRefCode sig,
                                 ValueCode v,
                                 uint32_t count,
                                 ValueCode *args);

ValueCode cranelift_func_addr(FunctionData *ptr, Type iaddr, FuncRefCode fnn);

ValueCode cranelift_splat(FunctionData *ptr, Type txn, ValueCode x);

ValueCode cranelift_swizzle(FunctionData *ptr, Type txn, ValueCode x, ValueCode y);

ValueCode cranelift_insertlane(FunctionData *ptr, ValueCode x, ValueCode y, Uimm8 idx);

ValueCode cranelift_extractlane(FunctionData *ptr, ValueCode x, Uimm8 idx);

ValueCode cranelift_imin(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_umin(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_imax(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_umax(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_avg_round(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_load(FunctionData *ptr,
                         Type mem,
                         MemFlagCode memflags,
                         ValueCode p,
                         Offset32 offset);

ValueCode cranelift_load_complex(FunctionData *ptr,
                                 Type mem,
                                 MemFlagCode memflags,
                                 uint32_t count,
                                 ValueCode *args,
                                 Offset32 offset);

InstCode cranelift_store(FunctionData *ptr,
                         MemFlagCode memflags,
                         ValueCode x,
                         ValueCode p,
                         Offset32 offset);

InstCode cranelift_store_complex(FunctionData *ptr,
                                 MemFlagCode memflags,
                                 ValueCode x,
                                 uint32_t count,
                                 ValueCode *args,
                                 Offset32 offset);

ValueCode cranelift_uload8(FunctionData *ptr,
                           Type iext8,
                           MemFlagCode memflags,
                           ValueCode p,
                           Offset32 offset);

ValueCode cranelift_uload8_complex(FunctionData *ptr,
                                   Type iext8,
                                   MemFlagCode memflags,
                                   uint32_t count,
                                   ValueCode *args,
                                   Offset32 offset);

ValueCode cranelift_sload8(FunctionData *ptr,
                           Type iext8,
                           MemFlagCode memflags,
                           ValueCode p,
                           Offset32 offset);

ValueCode cranelift_sload8_complex(FunctionData *ptr,
                                   Type iext8,
                                   MemFlagCode memflags,
                                   uint32_t count,
                                   ValueCode *args,
                                   Offset32 offset);

InstCode cranelift_istore8(FunctionData *ptr,
                           MemFlagCode memflags,
                           ValueCode x,
                           ValueCode p,
                           Offset32 offset);

InstCode cranelift_istore8_complex(FunctionData *ptr,
                                   MemFlagCode memflags,
                                   ValueCode x,
                                   uint32_t count,
                                   ValueCode *args,
                                   Offset32 offset);

ValueCode cranelift_uload16(FunctionData *ptr,
                            Type iext16,
                            MemFlagCode memflags,
                            ValueCode p,
                            Offset32 offset);

ValueCode cranelift_uload16_complex(FunctionData *ptr,
                                    Type iext16,
                                    MemFlagCode memflags,
                                    uint32_t count,
                                    ValueCode *args,
                                    Offset32 offset);

ValueCode cranelift_sload16(FunctionData *ptr,
                            Type iext16,
                            MemFlagCode memflags,
                            ValueCode p,
                            Offset32 offset);

ValueCode cranelift_sload16_complex(FunctionData *ptr,
                                    Type iext16,
                                    MemFlagCode memflags,
                                    uint32_t count,
                                    ValueCode *args,
                                    Offset32 offset);

InstCode cranelift_istore16(FunctionData *ptr,
                            MemFlagCode memflags,
                            ValueCode x,
                            ValueCode p,
                            Offset32 offset);

InstCode cranelift_istore16_complex(FunctionData *ptr,
                                    MemFlagCode memflags,
                                    ValueCode x,
                                    uint32_t count,
                                    ValueCode *args,
                                    Offset32 offset);

ValueCode cranelift_uload32(FunctionData *ptr, MemFlagCode memflags, ValueCode p, Offset32 offset);

ValueCode cranelift_uload32_complex(FunctionData *ptr,
                                    MemFlagCode memflags,
                                    uint32_t count,
                                    ValueCode *args,
                                    Offset32 offset);

ValueCode cranelift_sload32(FunctionData *ptr, MemFlagCode memflags, ValueCode p, Offset32 offset);

ValueCode cranelift_sload32_complex(FunctionData *ptr,
                                    MemFlagCode memflags,
                                    uint32_t count,
                                    ValueCode *args,
                                    Offset32 offset);

InstCode cranelift_istore32(FunctionData *ptr,
                            MemFlagCode memflags,
                            ValueCode x,
                            ValueCode p,
                            Offset32 offset);

InstCode cranelift_istore32_complex(FunctionData *ptr,
                                    MemFlagCode memflags,
                                    ValueCode x,
                                    uint32_t count,
                                    ValueCode *args,
                                    Offset32 offset);

ValueCode cranelift_uload8x8(FunctionData *ptr, MemFlagCode memflags, ValueCode p, Offset32 offset);

ValueCode cranelift_sload8x8(FunctionData *ptr, MemFlagCode memflags, ValueCode p, Offset32 offset);

ValueCode cranelift_uload16x4(FunctionData *ptr,
                              MemFlagCode memflags,
                              ValueCode p,
                              Offset32 offset);

ValueCode cranelift_sload16x4(FunctionData *ptr,
                              MemFlagCode memflags,
                              ValueCode p,
                              Offset32 offset);

ValueCode cranelift_uload32x2(FunctionData *ptr,
                              MemFlagCode memflags,
                              ValueCode p,
                              Offset32 offset);

ValueCode cranelift_sload32x2(FunctionData *ptr,
                              MemFlagCode memflags,
                              ValueCode p,
                              Offset32 offset);

ValueCode cranelift_stack_load(FunctionData *ptr, Type mem, StackSlotCode ss, Offset32 offset);

InstCode cranelift_stack_store(FunctionData *ptr, ValueCode x, StackSlotCode ss, Offset32 offset);

ValueCode cranelift_stack_addr(FunctionData *ptr, Type iaddr, StackSlotCode ss, Offset32 offset);

ValueCode cranelift_global_value(FunctionData *ptr, Type mem, GlobalValueCode gv);

ValueCode cranelift_symbol_value(FunctionData *ptr, Type mem, GlobalValueCode gv);

ValueCode cranelift_tls_value(FunctionData *ptr, Type mem, GlobalValueCode gv);

ValueCode cranelift_heap_addr(FunctionData *ptr, Type iaddr, HeapCode h, ValueCode p, Uimm32 size);

ValueCode cranelift_get_pinned_reg(FunctionData *ptr, Type iaddr);

InstCode cranelift_set_pinned_reg(FunctionData *ptr, ValueCode addr);

ValueCode cranelift_table_addr(FunctionData *ptr,
                               Type iaddr,
                               TableCode t,
                               ValueCode p,
                               Offset32 offset);

ValueCode cranelift_iconst(FunctionData *ptr, Type int_, Imm64 n);

ValueCode cranelift_f32const(FunctionData *ptr, float n);

ValueCode cranelift_f64const(FunctionData *ptr, double n);

ValueCode cranelift_bconst(FunctionData *ptr, Type bool_, bool n);

ValueCode cranelift_vconst(FunctionData *ptr, Type txn, ConstantCode n);

ValueCode cranelift_const_addr(FunctionData *ptr, Type iaddr, ConstantCode constant);

ValueCode cranelift_shuffle(FunctionData *ptr, ValueCode a, ValueCode b, ImmediateCode mask);

ValueCode cranelift_null(FunctionData *ptr, Type _ref);

InstCode cranelift_nop(FunctionData *ptr);

ValueCode cranelift_select(FunctionData *ptr, ValueCode c, ValueCode x, ValueCode y);

ValueCode cranelift_selectif(FunctionData *ptr,
                             Type any,
                             CraneliftIntCC cc,
                             ValueCode flags,
                             ValueCode x,
                             ValueCode y);

ValueCode cranelift_bitselect(FunctionData *ptr, ValueCode c, ValueCode x, ValueCode y);

ValueCode cranelift_copy(FunctionData *ptr, ValueCode x);

void cranelift_vsplit(FunctionData *ptr, ValueCode x, ValueCode *res1, ValueCode *res2);

ValueCode cranelift_vconcat(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_vselect(FunctionData *ptr, ValueCode c, ValueCode x, ValueCode y);

ValueCode cranelift_vany_true(FunctionData *ptr, ValueCode a);

ValueCode cranelift_vall_true(FunctionData *ptr, ValueCode a);

ValueCode cranelift_icmp(FunctionData *ptr, CraneliftIntCC cond, ValueCode x, ValueCode y);

ValueCode cranelift_icmp_imm(FunctionData *ptr, CraneliftIntCC cond, ValueCode x, Imm64 y);

ValueCode cranelift_ifcmp(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_ifcmp_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_iadd(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_uadd_sat(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_sadd_sat(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_isub(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_usub_sat(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_ssub_sat(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_ineg(FunctionData *ptr, ValueCode x);

ValueCode cranelift_imul(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_umulhi(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_smulhi(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_udiv(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_sdiv(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_urem(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_srem(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_iadd_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_imul_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_udiv_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_sdiv_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_urem_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_srem_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_irsub_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_iadd_cin(FunctionData *ptr, ValueCode x, ValueCode y, ValueCode c_in);

ValueCode cranelift_iadd_ifcin(FunctionData *ptr, ValueCode x, ValueCode y, ValueCode c_in);

void cranelift_iadd_cout(FunctionData *ptr,
                         ValueCode x,
                         ValueCode y,
                         ValueCode *res1,
                         ValueCode *res2);

void cranelift_iadd_ifcout(FunctionData *ptr,
                           ValueCode x,
                           ValueCode y,
                           ValueCode *res1,
                           ValueCode *res2);

void cranelift_iadd_carry(FunctionData *ptr,
                          ValueCode x,
                          ValueCode y,
                          ValueCode c_in,
                          ValueCode *res1,
                          ValueCode *res2);

void cranelift_iadd_ifcarry(FunctionData *ptr,
                            ValueCode x,
                            ValueCode y,
                            ValueCode c_in,
                            ValueCode *res1,
                            ValueCode *res2);

ValueCode cranelift_isub_bin(FunctionData *ptr, ValueCode x, ValueCode y, ValueCode b_in);

ValueCode cranelift_isub_ifbin(FunctionData *ptr, ValueCode x, ValueCode y, ValueCode b_in);

void cranelift_isub_bout(FunctionData *ptr,
                         ValueCode x,
                         ValueCode y,
                         ValueCode *res1,
                         ValueCode *res2);

void cranelift_isub_ifbout(FunctionData *ptr,
                           ValueCode x,
                           ValueCode y,
                           ValueCode *res1,
                           ValueCode *res2);

void cranelift_isub_borrow(FunctionData *ptr,
                           ValueCode x,
                           ValueCode y,
                           ValueCode b_in,
                           ValueCode *res1,
                           ValueCode *res2);

void cranelift_isub_ifborrow(FunctionData *ptr,
                             ValueCode x,
                             ValueCode y,
                             ValueCode b_in,
                             ValueCode *res1,
                             ValueCode *res2);

ValueCode cranelift_band(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_bor(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_bxor(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_bnot(FunctionData *ptr, ValueCode x);

ValueCode cranelift_band_not(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_bor_not(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_bxor_not(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_band_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_bor_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_bxor_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_rotl(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_rotr(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_rotl_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_rotr_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_ishl(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_ushr(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_sshr(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_ishl_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_ushr_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_sshr_imm(FunctionData *ptr, ValueCode x, Imm64 y);

ValueCode cranelift_bitrev(FunctionData *ptr, ValueCode x);

ValueCode cranelift_clz(FunctionData *ptr, ValueCode x);

ValueCode cranelift_cls(FunctionData *ptr, ValueCode x);

ValueCode cranelift_ctz(FunctionData *ptr, ValueCode x);

ValueCode cranelift_popcnt(FunctionData *ptr, ValueCode x);

ValueCode cranelift_fcmp(FunctionData *ptr, CraneliftFloatCC cond, ValueCode x, ValueCode y);

ValueCode cranelift_ffcmp(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_fadd(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_fsub(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_fmul(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_fdiv(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_sqrt(FunctionData *ptr, ValueCode x);

ValueCode cranelift_fma(FunctionData *ptr, ValueCode x, ValueCode y, ValueCode z);

ValueCode cranelift_fneg(FunctionData *ptr, ValueCode x);

ValueCode cranelift_fabs(FunctionData *ptr, ValueCode x);

ValueCode cranelift_fcopysign(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_fmin(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_fmax(FunctionData *ptr, ValueCode x, ValueCode y);

ValueCode cranelift_ceil(FunctionData *ptr, ValueCode x);

ValueCode cranelift_floor(FunctionData *ptr, ValueCode x);

ValueCode cranelift_trunc(FunctionData *ptr, ValueCode x);

ValueCode cranelift_nearest(FunctionData *ptr, ValueCode x);

ValueCode cranelift_is_null(FunctionData *ptr, ValueCode x);

ValueCode cranelift_is_invalid(FunctionData *ptr, ValueCode x);

ValueCode cranelift_trueif(FunctionData *ptr, CraneliftIntCC cond, ValueCode f);

ValueCode cranelift_trueff(FunctionData *ptr, CraneliftFloatCC cond, ValueCode f);

ValueCode cranelift_bitcast(FunctionData *ptr, Type memto, ValueCode x);

ValueCode cranelift_raw_bitcast(FunctionData *ptr, Type anyto, ValueCode x);

ValueCode cranelift_scalar_to_vector(FunctionData *ptr, Type txn, ValueCode s);

ValueCode cranelift_breduce(FunctionData *ptr, Type boolto, ValueCode x);

ValueCode cranelift_bextend(FunctionData *ptr, Type boolto, ValueCode x);

ValueCode cranelift_bint(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_bmask(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_ireduce(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_uextend(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_sextend(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_fpromote(FunctionData *ptr, Type floatto, ValueCode x);

ValueCode cranelift_fdemote(FunctionData *ptr, Type floatto, ValueCode x);

ValueCode cranelift_fcvt_to_uint(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_fcvt_to_uint_sat(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_fcvt_to_sint(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_fcvt_to_sint_sat(FunctionData *ptr, Type intto, ValueCode x);

ValueCode cranelift_fcvt_from_uint(FunctionData *ptr, Type floatto, ValueCode x);

ValueCode cranelift_fcvt_from_sint(FunctionData *ptr, Type floatto, ValueCode x);

void cranelift_isplit(FunctionData *ptr, ValueCode x, ValueCode *res1, ValueCode *res2);

ValueCode cranelift_iconcat(FunctionData *ptr, ValueCode lo, ValueCode hi);

} // extern "C"
