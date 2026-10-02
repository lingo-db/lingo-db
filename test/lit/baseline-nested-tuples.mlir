// RUN: LINGODB_EXECUTION_MODE=DEFAULT LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s %}

// util.pack / util.get_tuple with nested tuples, not folded away (passed to
// a function; hipy passes closures like this). The baseline backend used to
// ignore nested tuples in util.pack (all later elements shifted) and copied
// only the first part of a nested tuple in util.get_tuple: silently wrong
// values.

//CHECK: int(1)
//CHECK-NEXT: int(100)
//CHECK-NEXT: int(101)
//CHECK-NEXT: string("a string longer than twelve bytes")
//CHECK-NEXT: int(102)
//CHECK-NEXT: int(7)
module {
  func.func private @dumpI64(i64)
  func.func private @dumpString(!util.varlen32)
  func.func @callee(%t: tuple<i64, tuple<tuple<i64, i64>, !util.varlen32, i64>, i64>) {
    %a = util.get_tuple %t[0] : (tuple<i64, tuple<tuple<i64, i64>, !util.varlen32, i64>, i64>) -> i64
    %n = util.get_tuple %t[1] : (tuple<i64, tuple<tuple<i64, i64>, !util.varlen32, i64>, i64>) -> tuple<tuple<i64, i64>, !util.varlen32, i64>
    %z = util.get_tuple %t[2] : (tuple<i64, tuple<tuple<i64, i64>, !util.varlen32, i64>, i64>) -> i64
    %p = util.get_tuple %n[0] : (tuple<tuple<i64, i64>, !util.varlen32, i64>) -> tuple<i64, i64>
    %s = util.get_tuple %n[1] : (tuple<tuple<i64, i64>, !util.varlen32, i64>) -> !util.varlen32
    %c = util.get_tuple %n[2] : (tuple<tuple<i64, i64>, !util.varlen32, i64>) -> i64
    %p0 = util.get_tuple %p[0] : (tuple<i64, i64>) -> i64
    %p1 = util.get_tuple %p[1] : (tuple<i64, i64>) -> i64
    call @dumpI64(%a) : (i64) -> ()
    call @dumpI64(%p0) : (i64) -> ()
    call @dumpI64(%p1) : (i64) -> ()
    call @dumpString(%s) : (!util.varlen32) -> ()
    call @dumpI64(%c) : (i64) -> ()
    call @dumpI64(%z) : (i64) -> ()
    return
  }
  func.func @main() {
    %c1 = arith.constant 1 : i64
    %c7 = arith.constant 7 : i64
    %c100 = arith.constant 100 : i64
    %c101 = arith.constant 101 : i64
    %c102 = arith.constant 102 : i64
    %s = util.varlen32_create_const "a string longer than twelve bytes"
    %p = util.pack %c100, %c101 : i64, i64 -> tuple<i64, i64>
    %n = util.pack %p, %s, %c102 : tuple<i64, i64>, !util.varlen32, i64 -> tuple<tuple<i64, i64>, !util.varlen32, i64>
    %t = util.pack %c1, %n, %c7 : i64, tuple<tuple<i64, i64>, !util.varlen32, i64>, i64 -> tuple<i64, tuple<tuple<i64, i64>, !util.varlen32, i64>, i64>
    call @callee(%t) : (tuple<i64, tuple<tuple<i64, i64>, !util.varlen32, i64>, i64>) -> ()
    return
  }
}
