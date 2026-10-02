//RUN: mlir-db-opt %s -subop-organize-execution-steps -subop-split-into-nested-steps -mlir-print-local-scope | FileCheck %s --check-prefix=SPLIT
//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

// A nested query inside imperative code: per iteration of an scf.for inside a
// subop.map lambda (the shape nested SQL issued from a hipy UDF loop lowers
// to), generate the two rows i and i+1, sum them into a fresh simple_state and
// read the sum back via subop.state_to_native. The loop runs 200k iterations,
// so per-iteration stack growth (e.g. an alloca inside the loop body) would
// overflow the worker's fiber stack.
// Result: sum_{i<N} (2i+1) = N^2.

// The subop ops in the loop body form an "island" that gets wrapped into a
// nested execution group (with the induction variable as input) right where
// it was, i.e. inside the loop.
//SPLIT: subop.map
//SPLIT: scf.for %[[IV:[a-z0-9_]+]] =
//SPLIT: subop.nested_execution_group %[[IV]]
//SPLIT: subop.execution_step
//SPLIT: subop.create_simple_state
//SPLIT: subop.execution_step
//SPLIT: subop.reduce
//SPLIT: subop.execution_step
//SPLIT: subop.state_to_native
//SPLIT: subop.nested_execution_group_return
//SPLIT: util.unpack
//SPLIT: scf.yield

//CHECK: |                         total  |
//CHECK: ----------------------------------
//CHECK: |                   40000000000  |
module {
  func.func @main() {
    %result_table = subop.create !subop.result_table<[total : i64]>
    %generated, %streams = subop.generate [@t::@n({type = i64})] {
      %c = arith.constant 200000 : i64
      subop.generate_emit %c : i64
      tuples.return
    }
    %mapped = subop.map %generated computes : [@m::@total({type = i64})] input : [@t::@n] (%n: i64) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> (i64) : i64 {
        %state = subop.create_simple_state !subop.simple_state<[s : i64]> initial : {
          %z = arith.constant 0 : i64
          tuples.return %z : i64
        }
        %rows, %row_streams:2 = subop.generate [@g::@v({type = i64})] {
          subop.generate_emit %i : i64
          %c1_inner = arith.constant 1 : i64
          %next = arith.addi %i, %c1_inner : i64
          subop.generate_emit %next : i64
          tuples.return
        }
        %lk = subop.lookup %rows %state[] : !subop.simple_state<[s : i64]> @g::@ref({type = !subop.entry_ref<!subop.simple_state<[s : i64]>>})
        subop.reduce %lk @g::@ref [@g::@v] ["s"] ([%v], [%cur]) {
          %sum = arith.addi %cur, %v : i64
          tuples.return %sum : i64
        }
        %native = subop.state_to_native %state : !subop.simple_state<[s : i64]> -> tuple<i64>
        %val = util.unpack %native : tuple<i64> -> i64
        %acc2 = arith.addi %acc, %val : i64
        scf.yield %acc2 : i64
      }
      tuples.return %r : i64
    }
    subop.materialize %mapped {@m::@total => total}, %result_table : !subop.result_table<[total : i64]>
    %local_table = subop.create_from ["total"] %result_table : !subop.result_table<[total : i64]> -> !subop.local_table<[total : i64], ["total"]>
    subop.set_result 0 %local_table : !subop.local_table<[total : i64], ["total"]>
    return
  }
}
