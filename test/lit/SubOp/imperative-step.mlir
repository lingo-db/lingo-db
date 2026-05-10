// RUN: mlir-db-opt %s -mlir-print-local-scope | FileCheck %s
// RUN: mlir-db-opt %s -subop-organize-execution-steps -mlir-print-local-scope | FileCheck %s --check-prefix=ORGANIZED
//
// End-to-end shape: real subop pipeline → imperative DAG → real subop
// pipeline. Stage A counts rows of a buffer into a simple_state. Stage B
// reads that count via `subop.state_to_native`, runs an iterative
// Fibonacci over scf.for, and writes the result back via
// `subop.state_from_native` into a fresh simple_state. Stage C scans
// the fib state and materializes a one-row result table.

!s_in   = !subop.simple_state<[n : i32]>
!s_out  = !subop.simple_state<[fib : i32]>
!buf    = !subop.buffer<[v : i32]>
!result_table = !subop.result_table<[fib : i32]>
!local_table  = !subop.local_table<[fib : i32], ["fib"]>

// CHECK-LABEL: func.func @fib_pipeline
func.func @fib_pipeline() {

  // ----- Stage A: real subop pipeline — count(*) of a buffer scan into n_state.
  %src = subop.create !buf
  %n_state = subop.create_simple_state !s_in initial : {
    %z = arith.constant 0 : i32
    tuples.return %z : i32
  }
  // CHECK: subop.scan
  %scan = subop.scan %src : !buf {v => @s::@v({type = i32})}
  %lk = subop.lookup %scan %n_state[] : !s_in @s::@ref({type = !subop.entry_ref<!s_in>})
  // CHECK: subop.reduce
  subop.reduce %lk @s::@ref [] ["n"] ([], [%cur]) {
    %c1 = arith.constant 1 : i32
    %nn = arith.addi %cur, %c1 : i32
    tuples.return %nn : i32
  }

  // ----- Stage B: imperative iterative Fibonacci over scf.for.
  // CHECK: subop.state_to_native
  %t_in = subop.state_to_native %n_state : !s_in -> tuple<i32>
  %n_val = util.unpack %t_in : tuple<i32> -> i32

  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %ub = arith.index_cast %n_val : i32 to index

  // Two iter_args (%a, %b) — body uses BOTH, then yields (%b, %t) so each
  // iter_arg's value next step is a different live value.
  %fib:2 = scf.for %i = %c0_idx to %ub step %c1_idx
      iter_args(%a = %c0_i32, %b = %c1_i32) -> (i32, i32) {
    %t = arith.addi %a, %b : i32
    scf.yield %b, %t : i32, i32
  }

  // Diamond: both loop results feed an extra computation.
  %sum_chk = arith.addi %fib#0, %fib#1 : i32
  %t_out = util.pack %sum_chk : i32 -> tuple<i32>
  // CHECK: subop.state_from_native
  %fib_state = subop.state_from_native %t_out : tuple<i32> -> !s_out

  // ----- Stage C: real subop pipeline — scan fib state into a result table.
  // CHECK: subop.scan
  %scan_fib = subop.scan %fib_state : !s_out {fib => @r::@fib({type = i32})}
  %table = subop.create !result_table
  // CHECK: subop.materialize
  subop.materialize %scan_fib {@r::@fib => fib}, %table : !result_table
  %local = subop.create_from ["fib"] %table : !result_table -> !local_table
  // CHECK: subop.set_result
  subop.set_result 0 %local : !local_table

  return
}

// After OrganizeExecutionStepsPass, expect (in order):
//   - state-creator steps for `!buf` and `!s_in` (each in its own step,
//     since CreateSimpleStateOp / GenericCreateOp are SubOp-interface ops).
//   - Stage A: scan + lookup + reduce (count) — one step.
//   - Imperative middle: state_to_native + util.unpack + scf.for + util.pack
//     + state_from_native — collapsed into a single step via SSA closure
//     over non-SubOperator ops (the `state_to_native`/`state_from_native`
//     bridge ops do NOT implement SubOperator, so they merge into the
//     imperative pipeline rather than living in their own steps).
//   - Stage C: scan + materialize for the fib state — one step. The
//     surrounding result_table create / create_from / set_result each end
//     up in their own little step.
//
// Walk-position tiebreak preserves source order, so the steps appear in
// stage order even though there are no member-conflict edges between
// `subop.reduce` (writes `n`) and `subop.state_to_native` (reads `n`).

// ORGANIZED-LABEL: func.func @fib_pipeline
// ORGANIZED:       subop.execution_group
// Stage A: count pipeline.
// ORGANIZED:         subop.execution_step
// ORGANIZED:           subop.scan
// ORGANIZED:           subop.lookup
// ORGANIZED:           subop.reduce
// Imperative middle.
// ORGANIZED:         subop.execution_step
// ORGANIZED:           subop.state_to_native
// ORGANIZED:           util.unpack
// ORGANIZED:           scf.for
// ORGANIZED:           util.pack
// ORGANIZED:           subop.state_from_native
// Stage C: scan fib state + materialize.
// ORGANIZED:         subop.execution_step
// ORGANIZED:           subop.scan
// ORGANIZED:           subop.materialize
// ORGANIZED:         subop.set_result
