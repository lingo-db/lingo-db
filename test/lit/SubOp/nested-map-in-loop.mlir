//RUN: mlir-db-opt %s -subop-nested-map-inline | FileCheck %s --check-prefix=INLINE
//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

// A nested query with a subop.nested_map (e.g. a hash-join probe) inside
// imperative code (per iteration of an scf.for inside a subop.map lambda; the
// shape of a nested SQL join issued from a hipy UDF loop). The state of the
// aggregation is created after the nested_map, as RelAlgToSubOp emits it.
// Such blocks are not organized into execution steps before the nested_map's
// consumers (lookup + reduce) get inlined into its body, so the state creation
// has to be hoisted before the nested_map (used to fail with "operand #1 does
// not dominate this use").
// Per iteration i: the nested_map emits 0..i-1, summed up into the state.
// Result: sum_{i<10} i*(i-1)/2 = 120.

//INLINE: scf.for
//INLINE: %[[STATE:.*]] = subop.create_simple_state
//INLINE: subop.nested_map
//INLINE: subop.lookup %{{.*}}%[[STATE]]
//INLINE: subop.reduce
//INLINE: subop.state_to_native %[[STATE]]

//CHECK: |                         total  |
//CHECK: ----------------------------------
//CHECK: |                           120  |
module {
  func.func @main() {
    %result_table = subop.create !subop.result_table<[total : i64]>
    %generated, %streams = subop.generate [@t::@n({type = i64})] {
      %c = arith.constant 10 : i64
      subop.generate_emit %c : i64
      tuples.return
    }
    %mapped = subop.map %generated computes : [@m::@total({type = i64})] input : [@t::@n] (%n: i64) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> (i64) : i64 {
        %rows, %row_streams = subop.generate [@g::@k({type = index})] {
          %k = arith.index_cast %i : i64 to index
          subop.generate_emit %k : index
          tuples.return
        }
        %inner = subop.nested_map %rows [@g::@k] (%t, %k) {
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
        %state = subop.create_simple_state !subop.simple_state<[s : i64]> initial : {
          %z = arith.constant 0 : i64
          tuples.return %z : i64
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
    subop.materialize %mapped {@m::@total => total}, %result_table : !subop.result_table<[total : i64]>
    %local_table = subop.create_from ["total"] %result_table : !subop.result_table<[total : i64]> -> !subop.local_table<[total : i64], ["total"]>
    subop.set_result 0 %local_table : !subop.local_table<[total : i64], ["total"]>
    return
  }
}
