// RUN: LINGODB_EXECUTION_MODE=DEFAULT LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s %}

// Unsigned arithmetic on i8 / i16 values (computed in 32-bit registers by the
// baseline backend). The operands used to be sign-extended for every op, so
// the unsigned ops silently got wrong results for values with the top bit set
// (200 >> 1 = 228); signed ops still need the sign extension.

//CHECK: int(100)
//CHECK-NEXT: int(100)
//CHECK-NEXT: int(6)
//CHECK-NEXT: int(200)
//CHECK-NEXT: int(1)
//CHECK-NEXT: int(32766)
//CHECK-NEXT: int(-28)
//CHECK-NEXT: int(0)
//CHECK-NEXT: int(-56)
module {
  func.func private @dumpI64(i64)
  func.func @f(%x: i8, %y: i16) {
    %c1_i8 = arith.constant 1 : i8
    %c2_i8 = arith.constant 2 : i8
    %c7_i8 = arith.constant 7 : i8
    %c7_i16 = arith.constant 7 : i16
    %shr = arith.shrui %x, %c1_i8 : i8
    %div = arith.divui %x, %c2_i8 : i8
    %rem = arith.remui %y, %c7_i16 : i16
    %max = arith.maxui %x, %c1_i8 : i8
    %min = arith.minui %x, %c1_i8 : i8
    %c1_i16 = arith.constant 1 : i16
    %shr16 = arith.shrui %y, %c1_i16 : i16
    %sdiv = arith.divsi %x, %c2_i8 : i8
    %srem = arith.remsi %x, %c7_i8 : i8
    %smin = arith.minsi %x, %c1_i8 : i8
    %e0 = arith.extui %shr : i8 to i64
    %e1 = arith.extui %div : i8 to i64
    %e2 = arith.extui %rem : i16 to i64
    %e3 = arith.extui %max : i8 to i64
    %e4 = arith.extui %min : i8 to i64
    %e5 = arith.extui %shr16 : i16 to i64
    %e6 = arith.extsi %sdiv : i8 to i64
    %e7 = arith.extsi %srem : i8 to i64
    %e8 = arith.extsi %smin : i8 to i64
    call @dumpI64(%e0) : (i64) -> ()
    call @dumpI64(%e1) : (i64) -> ()
    call @dumpI64(%e2) : (i64) -> ()
    call @dumpI64(%e3) : (i64) -> ()
    call @dumpI64(%e4) : (i64) -> ()
    call @dumpI64(%e5) : (i64) -> ()
    call @dumpI64(%e6) : (i64) -> ()
    call @dumpI64(%e7) : (i64) -> ()
    call @dumpI64(%e8) : (i64) -> ()
    return
  }
  func.func @main() {
    %x = arith.constant 200 : i8
    %y = arith.constant 65533 : i16
    call @f(%x, %y) : (i8, i16) -> ()
    return
  }
}
