// RUN: LINGODB_EXECUTION_MODE=DEFAULT LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE LINGODB_BACKEND_ONLY=ON run-mlir %s | FileCheck %s %}

// Type conversions of earlier lowerings leave pairs of casts that cancel out
// (e.g. !util.ref<i8> -> !arrow.table -> !util.ref<i8> from lower-arrow /
// lower-py-interp, as for tabular Python UDFs). The LLVM backends reconcile
// them; the baseline backend used to fail with "Encountered unimplemented
// instruction: builtin.unrealized_conversion_cast".

//CHECK: int(42)
module {
  func.func private @dumpI64(i64)
  func.func @main() {
    %c42 = arith.constant 42 : i64
    %mem = util.alloca() : !util.ref<i64>
    util.store %c42 : i64, %mem[] : !util.ref<i64>
    %raw = util.generic_memref_cast %mem : !util.ref<i64> -> !util.ref<i8>
    %table = builtin.unrealized_conversion_cast %raw : !util.ref<i8> to !arrow.table
    %back = builtin.unrealized_conversion_cast %table : !arrow.table to !util.ref<i8>
    %ref = util.generic_memref_cast %back : !util.ref<i8> -> !util.ref<i64>
    %v = util.load %ref[] : !util.ref<i64> -> i64
    call @dumpI64(%v) : (i64) -> ()
    return
  }
}
