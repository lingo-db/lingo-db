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
uint32_t arith_shr_i32(uint32_t a, uint32_t b) { return (a >> b); }

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
uint64_t arith_shr_i64(uint64_t a, uint64_t b) { return (a >> b); }

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

__uint128_t arith_add_i128(__uint128_t a, __uint128_t b) { return (a + b); }
__uint128_t arith_land_i128(__uint128_t a, __uint128_t b) { return (a & b); }

uint32_t util_load_i8(uint8_t* ptr) { return *ptr; }
uint32_t util_load_i16(uint16_t* ptr) { return *ptr; }
uint32_t util_load_i32(uint32_t* ptr) { return *ptr; }
uint64_t util_load_i64(uint64_t* ptr) { return *ptr; }
__uint128_t load_i128(__uint128_t* ptr) { return *ptr; }

void util_store_i8(uint8_t* ptr, uint8_t value) { *ptr = value; }
void util_store_i16(uint16_t* ptr, uint16_t value) { *ptr = value; }
void util_store_i32(uint32_t* ptr, uint32_t value) { *ptr = value; }
void util_store_i64(uint64_t* ptr, uint64_t value) { *ptr = value; }
void store_i128(__uint128_t* ptr, __uint128_t value) { *ptr = value; }

int32_t arith_select_i32(uint8_t cond, int32_t val1, int32_t val2) { return ((cond & 1) ? val1 : val2); }
int64_t arith_select_i64(uint8_t cond, int64_t val1, int64_t val2) { return ((cond & 1) ? val1 : val2); }
__uint128_t arith_select_i128(uint8_t cond, __uint128_t val1, __uint128_t val2) { return ((cond & 1) ? val1 : val2); }
