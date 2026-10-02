// RUN: LINGODB_EXECUTION_MODE=DEFAULT LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s %}

// util.load / util.store of a whole (nested) tuple that is not taken apart
// right away (no util.pack / util.get_tuple to fold with), e.g. the closure
// argument that the try function of a hipy try/except loads and passes on.
// The baseline compiler loads/stores scalars only, so its legalization splits
// such accesses into one per element (used to fail with "Failed to compile
// instruction util.load" / "util.store").

//CHECK: int(1)
//CHECK-NEXT: int(100)
//CHECK-NEXT: string("a string longer than twelve bytes")
//CHECK-NEXT: int(7)
//CHECK-NEXT: int(1)
//CHECK-NEXT: int(100)
//CHECK-NEXT: string("a string longer than twelve bytes")
//CHECK-NEXT: int(7)
module {
  func.func private @dumpI64(i64)
  func.func private @dumpString(!util.varlen32)
  func.func @print(%t: tuple<i64, tuple<i64, !util.varlen32>, i64>) {
    %a = util.get_tuple %t[0] : (tuple<i64, tuple<i64, !util.varlen32>, i64>) -> i64
    %n = util.get_tuple %t[1] : (tuple<i64, tuple<i64, !util.varlen32>, i64>) -> tuple<i64, !util.varlen32>
    %b = util.get_tuple %n[0] : (tuple<i64, !util.varlen32>) -> i64
    %s = util.get_tuple %n[1] : (tuple<i64, !util.varlen32>) -> !util.varlen32
    %c = util.get_tuple %t[2] : (tuple<i64, tuple<i64, !util.varlen32>, i64>) -> i64
    call @dumpI64(%a) : (i64) -> ()
    call @dumpI64(%b) : (i64) -> ()
    call @dumpString(%s) : (!util.varlen32) -> ()
    call @dumpI64(%c) : (i64) -> ()
    return
  }
  func.func @roundtrip(%t: tuple<i64, tuple<i64, !util.varlen32>, i64>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    // whole-tuple store/load, without and with an index
    %mem = util.alloca(%c2) : !util.ref<tuple<i64, tuple<i64, !util.varlen32>, i64>>
    util.store %t : tuple<i64, tuple<i64, !util.varlen32>, i64>, %mem[] : !util.ref<tuple<i64, tuple<i64, !util.varlen32>, i64>>
    util.store %t : tuple<i64, tuple<i64, !util.varlen32>, i64>, %mem[%c1] : !util.ref<tuple<i64, tuple<i64, !util.varlen32>, i64>>
    %l0 = util.load %mem[] : !util.ref<tuple<i64, tuple<i64, !util.varlen32>, i64>> -> tuple<i64, tuple<i64, !util.varlen32>, i64>
    %l1 = util.load %mem[%c1] : !util.ref<tuple<i64, tuple<i64, !util.varlen32>, i64>> -> tuple<i64, tuple<i64, !util.varlen32>, i64>
    call @print(%l0) : (tuple<i64, tuple<i64, !util.varlen32>, i64>) -> ()
    call @print(%l1) : (tuple<i64, tuple<i64, !util.varlen32>, i64>) -> ()
    return
  }
  func.func @main() {
    %i1 = arith.constant 1 : i64
    %i7 = arith.constant 7 : i64
    %i100 = arith.constant 100 : i64
    %s = util.varlen32_create_const "a string longer than twelve bytes"
    %n = util.pack %i100, %s : i64, !util.varlen32 -> tuple<i64, !util.varlen32>
    %t = util.pack %i1, %n, %i7 : i64, tuple<i64, !util.varlen32>, i64 -> tuple<i64, tuple<i64, !util.varlen32>, i64>
    call @roundtrip(%t) : (tuple<i64, tuple<i64, !util.varlen32>, i64>) -> ()
    return
  }
}
