// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins | FileCheck %s --check-prefix=INPLACE
// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins --lower-relalg-to-subop | FileCheck %s --check-prefix=SUBOP
// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins --lower-relalg-to-subop -subop-organize-execution-steps -subop-split-into-nested-steps | FileCheck %s --check-prefix=SPLIT

// relalg.getfirstrow (row-valued nested SQL, e.g. hipy's sql.row(...)) returns
// the first row of its input as a tuple and raises a runtime error if there is
// none. It is evaluated in place: even outside of a loop, the subtree it
// consumes is neither extracted before the map nor turned into a join.

// INPLACE-LABEL: func.func @first_row
// INPLACE-NOT: relalg.singlejoin
// INPLACE: relalg.map
// INPLACE: relalg.const_relation columns : [@inner::@k
// INPLACE: relalg.selection
// INPLACE: relalg.getfirstrow {{.*}} [@inner::@k,@inner::@v] : tuple<i64, i64>

// Lowered to a nested query in the map lambda: the row's columns and a
// "found" flag (initially false, set by every row) are scattered into a fresh
// simple_state, read back via state_to_native, and an empty result raises an
// error at runtime.
// SUBOP-LABEL: func.func @first_row
// SUBOP: subop.map
// SUBOP: subop.combine_tuple_with_values
// SUBOP: subop.filter
// SUBOP: subop.create_simple_state <[found$0 : i1, col$0 : i64, col$1 : i64]> initial
// SUBOP: db.constant(false) : i1
// SUBOP: subop.map {{.*}}@getfirstrow::@found
// SUBOP: db.constant(true) : i1
// SUBOP: subop.lookup
// SUBOP: subop.scatter {{.*}} {@inner::@k => col$0, @inner::@v => col$1, @getfirstrow::@found => found$0}
// SUBOP: subop.state_to_native {{.*}} -> tuple<i1, i64, i64>
// SUBOP: %[[ROW:[0-9]+]]:3 = util.unpack
// SUBOP: %[[NOTFOUND:[0-9]+]] = arith.xori %[[ROW]]#0
// SUBOP: scf.if %[[NOTFOUND]]
// SUBOP: util.varlen32_create_const "nested SQL query returned no row"
// SUBOP: raiseError
// SUBOP: util.pack %[[ROW]]#1, %[[ROW]]#2 : i64, i64 -> tuple<i64, i64>
// The nested query sits directly in the map lambda (no loop around it); it is
// still wrapped into a nested execution group in place.
// SPLIT-LABEL: func.func @first_row
// SPLIT: subop.map
// SPLIT: subop.nested_execution_group %arg
// SPLIT: subop.create_simple_state
// SPLIT: subop.scatter
// SPLIT: subop.state_to_native
// SPLIT: subop.nested_execution_group_return
// SPLIT: util.unpack
// SPLIT: scf.if
module {
  func.func @first_row() {
    %0 = relalg.const_relation columns : [@outer::@x({type = i64})] values : [[1]]
    %1 = relalg.map %0 computes : [@m::@res({type = i64})] (%t: !tuples.tuple) {
      %n = tuples.getcol %t @outer::@x : i64
      %rel = relalg.const_relation columns : [@inner::@k({type = i64}), @inner::@v({type = i64})] values : [[0, 10], [1, 11], [2, 12]]
      %sel = relalg.selection %rel (%t2: !tuples.tuple) {
        %k = tuples.getcol %t2 @inner::@k : i64
        %eq = db.compare eq %k : i64, %n : i64
        tuples.return %eq : i1
      }
      %row = relalg.getfirstrow %sel [@inner::@k, @inner::@v] : tuple<i64, i64>
      %kv:2 = util.unpack %row : tuple<i64, i64> -> i64, i64
      %sum = arith.addi %kv#0, %kv#1 : i64
      tuples.return %sum : i64
    }
    %2 = relalg.materialize %1 [@m::@res] => ["res"] : !subop.local_table<[res$0 : i64], ["res"]>
    subop.set_result 0 %2 : !subop.local_table<[res$0 : i64], ["res"]>
    return
  }
}
