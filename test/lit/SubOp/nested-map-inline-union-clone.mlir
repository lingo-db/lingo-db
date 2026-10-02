//RUN: mlir-db-opt %s -subop-nested-map-inline | FileCheck %s --check-prefix=INLINE
//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

// The result of the outer nested_map flows into a subop.union, so
// InlineNestedMapPass clones the union's consumer (a subop.map) into the
// nested_map body. That map contains a nested query evaluated in place inside
// a loop (like nested SQL in a hipy UDF loop), with another nested_map.
// The nested_map inside the *clone* must be inlined as well (it used to be
// left alone, which later failed in subop-split-into-nested-steps with
// "operand #0 does not dominate this use").
// Per row with value m: sum_{i<m} sum_{j<i} j = m(m-1)(m-2)/6.
// Rows: m = 30, 40 (via the outer nested_map), 3, 4 (other union input).

// after inlining, no nested_map returns a stream anymore
//INLINE-NOT: tuples.return %{{.*}} : !tuples.tuplestream

//CHECK: |                         total  |
//CHECK: ----------------------------------
//CHECK-DAG: |                             1  |
//CHECK-DAG: |                             4  |
//CHECK-DAG: |                          4060  |
//CHECK-DAG: |                          9880  |
module {
  func.func @main() {
    %result_table = subop.create !subop.result_table<[total : i64]>
    %rows, %rows_streams:2 = subop.generate [@t::@n({type = i64})] {
      %c3 = arith.constant 3 : i64
      %c4 = arith.constant 4 : i64
      subop.generate_emit %c3 : i64
      subop.generate_emit %c4 : i64
      tuples.return
    }
    %outer = subop.nested_map %rows [@t::@n] (%t, %n) {
      %scaled, %scaled_streams = subop.generate [@u::@m({type = i64})] {
        %c10 = arith.constant 10 : i64
        %m = arith.muli %n, %c10 : i64
        subop.generate_emit %m : i64
        tuples.return
      }
      tuples.return %scaled : !tuples.tuplestream
    }
    %rows2, %rows2_streams:2 = subop.generate [@t2::@n({type = i64})] {
      %c3 = arith.constant 3 : i64
      %c4 = arith.constant 4 : i64
      subop.generate_emit %c3 : i64
      subop.generate_emit %c4 : i64
      tuples.return
    }
    %other = subop.map %rows2 computes : [@u::@m({type = i64})] input : [@t2::@n] (%n: i64) {
      tuples.return %n : i64
    }
    %union = subop.union %outer, %other
    %mapped = subop.map %union computes : [@r::@total({type = i64})] input : [@u::@m] (%m: i64) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %r = scf.for %i = %c0 to %m step %c1 iter_args(%acc = %c0) -> (i64) : i64 {
        %state = subop.create_simple_state !subop.simple_state<[s : i64]> initial : {
          %z = arith.constant 0 : i64
          tuples.return %z : i64
        }
        %g, %g_streams = subop.generate [@g::@k({type = index})] {
          %k = arith.index_cast %i : i64 to index
          subop.generate_emit %k : index
          tuples.return
        }
        %inner = subop.nested_map %g [@g::@k] (%gt, %k) {
          %vals, %val_streams = subop.generate [@h::@v({type = i64})] {
            %ci0 = arith.constant 0 : index
            %ci1 = arith.constant 1 : index
            scf.for %j = %ci0 to %k step %ci1 {
              %jv = arith.index_cast %j : index to i64
              subop.generate_emit %jv : i64
            }
            tuples.return
          }
          tuples.return %vals : !tuples.tuplestream
        }
        %lk = subop.lookup %inner %state[] : !subop.simple_state<[s : i64]> @g::@ref({type = !subop.entry_ref<!subop.simple_state<[s : i64]>>})
        subop.reduce %lk @g::@ref [@h::@v] ["s"] ([%v], [%cur]) {
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
    subop.materialize %mapped {@r::@total => total}, %result_table : !subop.result_table<[total : i64]>
    %local_table = subop.create_from ["total"] %result_table : !subop.result_table<[total : i64]> -> !subop.local_table<[total : i64], ["total"]>
    subop.set_result 0 %local_table : !subop.local_table<[total : i64], ["total"]>
    return
  }
}
