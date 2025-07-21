#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

uint32_t arith_add_i32(uint32_t a, uint32_t b) { return (a + b); }
uint32_t arith_sub_i32(uint32_t a, uint32_t b) { return (a - b); }
uint32_t arith_mul_i32(uint32_t a, uint32_t b) { return (a * b); }
int32_t arith_sdiv_i32(int32_t a, int32_t b) { return (a / b); }
uint32_t arith_udiv_i32(uint32_t a, uint32_t b) { return (a / b); }
int32_t arith_srem_i32(int32_t a, int32_t b) { return (a % b); }
uint32_t arith_urem_i32(uint32_t a, uint32_t b) { return (a % b); }
uint32_t arith_land_i32(uint32_t a, uint32_t b) { return (a & b); }
uint32_t arith_lor_i32(uint32_t a, uint32_t b) { return (a | b); }
uint32_t arith_lxor_i32(uint32_t a, uint32_t b) { return (a ^ b); }
uint32_t arith_shl_i32(uint32_t a, uint32_t b) { return (a << b); }
uint32_t arith_shr_u32(uint32_t a, uint32_t b) { return (a >> b); }

uint64_t arith_add_i64(uint64_t a, uint64_t b) { return (a + b); }
uint64_t arith_sub_i64(uint64_t a, uint64_t b) { return (a - b); }
uint64_t arith_mul_i64(uint64_t a, uint64_t b) { return (a * b); }
int64_t arith_sdiv_i64(int64_t a, int64_t b) { return (a / b); }
uint64_t arith_udiv_i64(uint64_t a, uint64_t b) { return (a / b); }
int64_t arith_srem_i64(int64_t a, int64_t b) { return (a % b); }
uint64_t arith_urem_i64(uint64_t a, uint64_t b) { return (a % b); }
uint64_t arith_land_i64(uint64_t a, uint64_t b) { return (a & b); }
uint64_t arith_lor_i64(uint64_t a, uint64_t b) { return (a | b); }
uint64_t arith_lxor_i64(uint64_t a, uint64_t b) { return (a ^ b); }
uint64_t arith_shl_i64(uint64_t a, uint64_t b) { return (a << b); }
uint64_t arith_shr_u64(uint64_t a, uint64_t b) { return (a >> b); }

__uint128_t arith_add_i128(__uint128_t a, __uint128_t b) { return (a + b); }
__uint128_t arith_sub_i128(__uint128_t a, __uint128_t b) { return (a - b); }
__uint128_t arith_mul_i128(__uint128_t a, __uint128_t b) { return (a * b); }
// Division and remainder operations are not supported by tpde_encoder. We need to call builtins for these.
//__int128_t arith_sdiv_i128(__int128_t a, __int128_t b) { return (a / b); }
//__uint128_t arith_udiv_i128(__uint128_t a, __uint128_t b) { return (a / b); }
//__int128_t arith_srem_i128(__int128_t a, __int128_t b) { return (a % b); }
//__uint128_t arith_urem_i128(__uint128_t a, __uint128_t b) { return (a % b); }
__uint128_t arith_land_i128(__uint128_t a, __uint128_t b) { return (a & b); }
__uint128_t arith_lor_i128(__uint128_t a, __uint128_t b) { return (a | b); }
__uint128_t arith_lxor_i128(__uint128_t a, __uint128_t b) { return (a ^ b); }

float arith_add_f32(float a, float b) { return (a + b); }
float arith_sub_f32(float a, float b) { return (a - b); }
float arith_mul_f32(float a, float b) { return (a * b); }
float arith_div_f32(float a, float b) { return (a / b); }
// float arith_rem_f32(float a, float b) { return __builtin_fmodf(a, b); }

double arith_add_f64(double a, double b) { return (a + b); }
double arith_sub_f64(double a, double b) { return (a - b); }
double arith_mul_f64(double a, double b) { return (a * b); }
double arith_div_f64(double a, double b) { return (a / b); }
// double arith_rem_f64(double a, double b) { return __builtin_fmod(a, b); }

bool util_load_i1(bool* ptr) { return *ptr; }
uint8_t util_load_i8(uint8_t* ptr) { return *ptr; }
uint16_t util_load_i16(uint16_t* ptr) { return *ptr; }
uint32_t util_load_i32(uint32_t* ptr) { return *ptr; }
uint64_t util_load_i64(uint64_t* ptr) { return *ptr; }
__uint128_t load_i128(__uint128_t* ptr) { return *ptr; }

void util_store_i1(bool* ptr, bool value) { *ptr = value; }
void util_store_i8(uint8_t* ptr, uint8_t value) { *ptr = value; }
void util_store_i16(uint16_t* ptr, uint16_t value) { *ptr = value; }
void util_store_i32(uint32_t* ptr, uint32_t value) { *ptr = value; }
void util_store_i64(uint64_t* ptr, uint64_t value) { *ptr = value; }
void store_i128(__uint128_t* ptr, __uint128_t value) { *ptr = value; }

int32_t arith_select_i32(uint8_t cond, int32_t val1, int32_t val2) { return ((cond & 1) ? val1 : val2); }
int64_t arith_select_i64(uint8_t cond, int64_t val1, int64_t val2) { return ((cond & 1) ? val1 : val2); }
__uint128_t arith_select_i128(uint8_t cond, __uint128_t val1, __uint128_t val2) { return ((cond & 1) ? val1 : val2); }

__int128_t arith_sext_i64_i128(int64_t in) { return (__int128_t)in; }
__uint128_t arith_zext_i64_i128(uint64_t in) { return (__uint128_t)in; }

typedef struct UtilVarLenRes { uint64_t totalEqual; uint64_t needsDetailedComp; } UtilVarLenRes;
UtilVarLenRes util_varlen_cmp(__uint128_t lhs, __uint128_t rhs) {
    // cmp lengths + first 4 chars
    uint64_t first64Left = (uint64_t)(lhs);
    uint64_t first64Right = (uint64_t)(rhs);
    bool first64Equal = (first64Left == first64Right);

    // cmp chars 5-8 or pointers (bad)
    uint64_t left64Left = (uint64_t)(lhs >> 64);
    uint64_t left64Right = (uint64_t)(rhs >> 64);
    bool last64Equal = (left64Left == left64Right);

    bool totalEqual = first64Equal && last64Equal;
    uint32_t len = first64Left & 0xFFFFFFFF;
    bool lenGt12 = len > 12;
    bool needsDetailedComp = first64Equal && lenGt12;
    return (UtilVarLenRes){totalEqual, needsDetailedComp};
}

// --------------------------
// float comparisons
// --------------------------

#define FOP_ORD(ty, ty2, name, op) uint32_t arith_cmp_##ty2##_##name(ty a, ty b) { return !__builtin_isunordered(a, b) && (a op b); };
#define FOP_UNRD(ty, ty2, name, op) uint32_t arith_cmp_##ty2##_##name(ty a, ty b) { return __builtin_isunordered(a, b) || (a op b); };
#define FOPS(ty, ty2) FOP_ORD(ty, ty2, oeq, ==) \
FOP_ORD(ty, ty2, ogt, >) \
FOP_ORD(ty, ty2, oge, >=) \
FOP_ORD(ty, ty2, olt, <) \
FOP_ORD(ty, ty2, ole, <=) \
FOP_ORD(ty, ty2, one, !=) \
uint32_t arith_cmp_##ty2##_ord(ty a, ty b) { return !__builtin_isunordered(a, b); } \
FOP_UNRD(ty, ty2, ueq, ==) \
FOP_UNRD(ty, ty2, ugt, >) \
FOP_UNRD(ty, ty2, uge, >=) \
FOP_UNRD(ty, ty2, ult, <) \
FOP_UNRD(ty, ty2, ule, <=) \
FOP_UNRD(ty, ty2, une, !=) \
uint32_t arith_cmp_##ty2##_uno(ty a, ty b) { return __builtin_isunordered(a, b); }

FOPS(float, f32)
FOPS(double, f64)

#undef FOP_ORD
#undef FOP_UNORD
#undef FOPS

// --------------------------
// special operations
// --------------------------

uint64_t util_hash_64(uint64_t val) {
    uint64_t p1 = 11400714819323198549ull;
    uint64_t m1 = p1 * val;
    uint64_t reversed = __builtin_bswap64(m1);
    return m1 ^ reversed;
}

bool util_ptr_tag_matches(void* ref, uint64_t hash, void* bloomMaskPtr) {
    uint64_t shiftAmount = 53;
    uint64_t slot = hash >> shiftAmount;
    uint64_t tag = ((uint16_t*)bloomMaskPtr)[slot];
    uint16_t entry = (uint16_t)(uintptr_t)ref;
    uint16_t negatedEntry = entry ^ 0xFFFF;
    uint16_t anded = tag & negatedEntry;
    bool isMatch = anded == 0;
    return isMatch;
}

void* util_untag_ptr(void* ref) {
    uintptr_t ptrAsInt = (uintptr_t)ref;
    uint64_t shiftAmount = 16;
    uintptr_t ptrWithoutTag = ptrAsInt >> shiftAmount;
    return (void*)ptrWithoutTag;
}

bool util_is_ref_valid(void* ref) {
    return ref != NULL;
}

uint64_t util_hash_combine(uint64_t h1, uint64_t h2) {
    uint64_t reversed = __builtin_bswap64(h1);
    return h2 ^ reversed;
}


typedef struct UtilTryCheapHashRes { bool lenLt13; uint64_t hash; } UtilTryCheapHashRes;
UtilTryCheapHashRes util_varlen_try_cheap_hash(__uint128_t varlen) {
    uint64_t first64 = (uint64_t)(varlen);
    uint64_t last64 = (uint64_t)(varlen >> 64);

    uint64_t mask = 0xFFFFFFFF;
    uint64_t len = first64 & mask;
    bool lenLt13 = len < 13;
    uint64_t fHash = util_hash_64(first64);
    uint64_t lHash = util_hash_64(last64);
    uint64_t hash = util_hash_combine(fHash, lHash);
    return (UtilTryCheapHashRes){lenLt13, hash};
}
