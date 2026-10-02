// RUN: mlir-db-opt %s -split-input-file -mlir-print-local-scope --relalg-extract-nested-operators | FileCheck %s --check-prefix=EXTRACT
// RUN: mlir-db-opt %s -split-input-file -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins | FileCheck %s --check-prefix=JOINS

// A subquery inside a loop of a map lambda (e.g. nested SQL issued per
// iteration by a hipy UDF) uses the loop induction variable. It is evaluated
// once per iteration, so it must stay inside the loop: it is neither extracted
// before the map nor turned into a join with the map's input.

// EXTRACT-LABEL: func.func @in_loop
// EXTRACT: relalg.map
// EXTRACT: scf.for
// EXTRACT: relalg.const_relation columns : [@inner::@k
// EXTRACT: relalg.selection
// EXTRACT: relalg.getscalar @inner::@v
// EXTRACT: scf.yield

// JOINS-LABEL: func.func @in_loop
// JOINS-NOT: relalg.singlejoin
// JOINS: scf.for
// JOINS: relalg.getscalar @inner::@v

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

// -----

// Control: a subquery that is *not* inside a loop is still extracted before
// the map and decorrelated into a join.

// EXTRACT-LABEL: func.func @not_in_loop
// EXTRACT: relalg.const_relation columns : [@inner::@k
// EXTRACT: relalg.selection
// EXTRACT: relalg.map
// EXTRACT: relalg.getscalar @inner::@v

// JOINS-LABEL: func.func @not_in_loop
// JOINS: relalg.singlejoin
// JOINS: relalg.map
// JOINS-NOT: relalg.getscalar

module {
  func.func @not_in_loop() {
    %0 = relalg.const_relation columns : [@outer::@x({type = i64})] values : [[1]]
    %1 = relalg.map %0 computes : [@m::@res({type = !db.nullable<i64>})] (%t: !tuples.tuple) {
      %n = tuples.getcol %t @outer::@x : i64
      %rel = relalg.const_relation columns : [@inner::@k({type = i64}), @inner::@v({type = i64})] values : [[0, 10], [1, 11], [2, 12]]
      %sel = relalg.selection %rel (%t2: !tuples.tuple) {
        %k = tuples.getcol %t2 @inner::@k : i64
        %eq = db.compare eq %k : i64, %n : i64
        tuples.return %eq : i1
      }
      %v = relalg.getscalar @inner::@v %sel : !db.nullable<i64>
      tuples.return %v : !db.nullable<i64>
    }
    %2 = relalg.materialize %1 [@m::@res] => ["res"] : !subop.local_table<[res$0 : !db.nullable<i64>], ["res"]>
    subop.set_result 0 %2 : !subop.local_table<[res$0 : !db.nullable<i64>], ["res"]>
    return
  }
}
