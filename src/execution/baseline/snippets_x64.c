#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

uint32_t arith_add_i32(uint32_t a, uint32_t b) { return (a + b); }
uint32_t arith_sub_i32(uint32_t a, uint32_t b) { return (a - b); }
uint32_t arith_mul_i32(uint32_t a, uint32_t b) { return (a * b); }
int32_t arith_sdiv_i32(int32_t a, int32_t b) { return (a / b); }
uint32_t arith_land_i32(uint32_t a, uint32_t b) { return (a & b); }
uint32_t arith_lor_i32(uint32_t a, uint32_t b) { return (a | b); }
uint32_t arith_lxor_i32(uint32_t a, uint32_t b) { return (a ^ b); }
uint32_t arith_shl_i32(uint32_t a, uint32_t b) { return (a << b); }
uint32_t arith_shr_i32(uint32_t a, uint32_t b) { return (a >> b); }

uint64_t arith_add_i64(uint64_t a, uint64_t b) { return (a + b); }
uint64_t arith_sub_i64(uint64_t a, uint64_t b) { return (a - b); }
uint64_t arith_mul_i64(uint64_t a, uint64_t b) { return (a * b); }
int64_t arith_sdiv_i64(int64_t a, int64_t b) { return (a / b); }
uint64_t arith_land_i64(uint64_t a, uint64_t b) { return (a & b); }
uint64_t arith_lor_i64(uint64_t a, uint64_t b) { return (a | b); }
uint64_t arith_lxor_i64(uint64_t a, uint64_t b) { return (a ^ b); }
uint64_t arith_shl_i64(uint64_t a, uint64_t b) { return (a << b); }
uint64_t arith_shr_i64(uint64_t a, uint64_t b) { return (a >> b); }

__uint128_t arith_add_i128(__uint128_t a, __uint128_t b) { return (a + b); }
__uint128_t arith_land_i128(__uint128_t a, __uint128_t b) { return (a & b); }

uint32_t util_load_i8(uint8_t* ptr) { return *ptr; }
uint32_t util_load_i16(uint16_t* ptr) { return *ptr; }
uint32_t util_load_i32(uint32_t* ptr) { return *ptr; }
uint64_t util_load_i64(uint64_t* ptr) { return *ptr; }
__uint128_t loadi128(__uint128_t* ptr) { return *ptr; }