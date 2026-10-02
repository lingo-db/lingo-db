//RUN: mlir-db-opt %s -subop-organize-execution-steps -subop-prepare-lowering -subop-split-into-nested-steps -lower-subop-to-cf | FileCheck %s --check-prefix=LOWERED
//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

// The lowering of subop.map folds the ops of the lambda while inlining it. A
// commutative op with a constant left operand (here: arith.addi %c5, %n and
// arith.ori %false, %gt) is folded *in place* (operands swapped, no
// replacement values); it must be kept, not erased (used to fail with "null
// operand found"). Directly lowered, as the SQL pipeline doesn't canonicalize
// after to-subop (run-mlir does, which hides the bug).
// Rows n = 0..3: n + 5 = 5..8, n > 1 = f f t t.

//LOWERED: %[[C5:.*]] = arith.constant 5 : i64
//LOWERED: arith.addi %{{.*}}, %[[C5]] : i64
//LOWERED: %[[FALSE:.*]] = arith.constant false
//LOWERED: %[[GT:.*]] = arith.cmpi sgt
//LOWERED: arith.ori %[[GT]], %[[FALSE]] : i1

//CHECK: |                             s  |                             b  |
//CHECK: ------------------------------------------------------------------
//CHECK: |                             5  |                         false  |
//CHECK: |                             6  |                         false  |
//CHECK: |                             7  |                          true  |
//CHECK: |                             8  |                          true  |
module {
  func.func @main() {
    %result_table = subop.create !subop.result_table<[s : i64, b : i1]>
    %generated, %streams = subop.generate [@t::@n({type = i64})] {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c4 = arith.constant 4 : i64
      scf.for %i = %c0 to %c4 step %c1 : i64 {
        subop.generate_emit %i : i64
      }
      tuples.return
    }
    %mapped = subop.map %generated computes : [@m::@s({type = i64}), @m::@b({type = i1})] input : [@t::@n] (%n: i64) {
      %c5 = arith.constant 5 : i64
      %s = arith.addi %c5, %n : i64
      %false = arith.constant false
      %c1 = arith.constant 1 : i64
      %gt = arith.cmpi sgt, %n, %c1 : i64
      %b = arith.ori %false, %gt : i1
      tuples.return %s, %b : i64, i1
    }
    subop.materialize %mapped {@m::@s => s, @m::@b => b}, %result_table : !subop.result_table<[s : i64, b : i1]>
    %local_table = subop.create_from ["s", "b"] %result_table : !subop.result_table<[s : i64, b : i1]> -> !subop.local_table<[s : i64, b : i1], ["s", "b"]>
    subop.set_result 0 %local_table : !subop.local_table<[s : i64, b : i1], ["s", "b"]>
    return
  }
}
