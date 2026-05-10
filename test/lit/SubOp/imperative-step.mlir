// RUN: mlir-db-opt %s -mlir-print-local-scope | FileCheck %s
// RUN: mlir-db-opt %s -subop-organize-execution-steps -mlir-print-local-scope | FileCheck %s --check-prefix=ORGANIZED
//
// Demonstrates the pattern: subop pipeline → imperative DAG → subop pipeline.
// The imperative section reads from a `subop.simple_state` via
// `subop.state_to_native`, runs an iterative Fibonacci computation
// (whose body has multiple SSA values flowing through scf.for iter_args, so
// the data flow is not a single SSA chain), and writes back a fresh
// `subop.simple_state` via `subop.state_from_native`. A follow-up
// `subop.state_to_native` extracts the result.

!s_in  = !subop.simple_state<[n : i32]>
!s_out = !subop.simple_state<[fib : i32, n : i32]>

// CHECK-LABEL: func.func @fib_pipeline
func.func @fib_pipeline() -> i32 {

  // ----- Stage 1: subop creates a simple_state holding `n`.
  %st_in = subop.create_simple_state !s_in initial : {
    %n0 = arith.constant 10 : i32
    tuples.return %n0 : i32
  }

  // ----- Stage 2: imperative iterative Fibonacci over scf.for.
  // CHECK: subop.state_to_native
  %t_in = subop.state_to_native %st_in : !s_in -> tuple<i32>
  %n_val = util.unpack %t_in : tuple<i32> -> i32

  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %ub = arith.index_cast %n_val : i32 to index

  // Two iter_args (%a, %b) — body computes %t from BOTH, then yields
  // (%b, %t) so each iter_arg's value next step is a different live value.
  %fib:2 = scf.for %i = %c0_idx to %ub step %c1_idx
      iter_args(%a = %c0_i32, %b = %c1_i32) -> (i32, i32) {
    %t = arith.addi %a, %b : i32
    scf.yield %b, %t : i32, i32
  }

  // Diamond: both loop results feed an extra computation.
  %sum_chk = arith.addi %fib#0, %fib#1 : i32
  // Pack TWO independent values into the output tuple: the n we read back,
  // and the diamond result.
  %t_out = util.pack %sum_chk, %n_val : i32, i32 -> tuple<i32, i32>
  // CHECK: subop.state_from_native
  %st_out = subop.state_from_native %t_out : tuple<i32, i32> -> !s_out

  // ----- Stage 3: read the result back through state_to_native + unpack.
  // CHECK: subop.state_to_native
  %t_out_read = subop.state_to_native %st_out : !s_out -> tuple<i32, i32>
  %fib_val, %n_unused = util.unpack %t_out_read : tuple<i32, i32> -> i32, i32
  return %fib_val : i32
}

// After OrganizeExecutionStepsPass, the imperative DAG collapses into a
// SINGLE execution_step via SSA closure — even though the ops have no
// tuple-stream chain between them. `subop.state_to_native` /
// `subop.state_from_native` and `subop.create_simple_state` are
// non-`SubOperator` ops, so they merge into the same imperative pipeline.

// ORGANIZED-LABEL: func.func @fib_pipeline
// ORGANIZED:       subop.execution_step
// ORGANIZED:         subop.create_simple_state
// ORGANIZED:         subop.state_to_native
// ORGANIZED:         util.unpack
// ORGANIZED:         scf.for
// ORGANIZED:           arith.addi
// ORGANIZED:           scf.yield
// ORGANIZED:         arith.addi
// ORGANIZED:         util.pack
// ORGANIZED:         subop.state_from_native
// ORGANIZED:         subop.state_to_native
// ORGANIZED:         util.unpack
// ORGANIZED:         subop.execution_step_return
// ORGANIZED-NEXT:   } -> i32
// ORGANIZED-NEXT:   subop.execution_group_return
