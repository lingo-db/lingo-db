// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins --lower-relalg-to-subop | FileCheck %s

// A subquery inside a loop of a map lambda stays in the loop (see
// nested-ops-in-loop.mlir) and is lowered to a nested query inside the loop:
// the induction variable becomes a
// column (combine_tuple_with_values) so that the selection's subop.map stays
// isolated from above; the scalar is scattered into a fresh simple_state
// (initialized to NULL = "no row") and read back via state_to_native.
// CHECK-LABEL: func.func @in_loop
// CHECK: subop.map
// CHECK: scf.for %[[IV:[a-z0-9_]+]] =
// CHECK: subop.generate
// CHECK: subop.combine_tuple_with_values %{{.*}}, %[[IV]] : i64 => [@captured::@v0({type = i64})]
// CHECK: subop.map {{.*}} input : [@inner::@k,@captured::@v0]
// CHECK: db.compare eq
// CHECK: subop.filter
// CHECK: subop.create_simple_state <[scalar$0 : !db.nullable<i64>]> initial
// CHECK: db.null : <i64>
// CHECK: subop.map {{.*}}@getscalar::@casted
// CHECK: db.as_nullable
// CHECK: subop.lookup
// CHECK: subop.scatter {{.*}} {@getscalar::@casted => scalar$0}
// CHECK: subop.state_to_native {{.*}} -> tuple<!db.nullable<i64>>
// CHECK: util.unpack {{.*}} -> !db.nullable<i64>
// CHECK: scf.yield
module {
  func.func @in_loop() {
    %0 = relalg.const_relation columns : [@outer::@x({type = i64})] values : [[3]]
    %1 = relalg.map %0 computes : [@m::@res({type = i64})] (%t: !tuples.tuple) {
      %n = tuples.getcol %t @outer::@x : i64
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> (i64) : i64 {
        %rel = relalg.const_relation columns : [@inner::@k({type = i64}), @inner::@v({type = i64})] values : [[0, 10], [1, 11], [2, 12]]
        %sel = relalg.selection %rel (%t2: !tuples.tuple) {
          %k = tuples.getcol %t2 @inner::@k : i64
          %eq = db.compare eq %k : i64, %i : i64
          tuples.return %eq : i1
        }
        %v = relalg.getscalar @inner::@v %sel : !db.nullable<i64>
        %vv = db.nullable_get_val %v : <i64>
        %acc2 = arith.addi %acc, %vv : i64
        scf.yield %acc2 : i64
      }
      tuples.return %r : i64
    }
    %2 = relalg.materialize %1 [@m::@res] => ["res"] : !subop.local_table<[res$0 : i64], ["res"]>
    subop.set_result 0 %2 : !subop.local_table<[res$0 : i64], ["res"]>
    return
  }
}

