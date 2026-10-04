// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins | FileCheck %s --check-prefix=INPLACE
// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins --lower-relalg-to-subop | FileCheck %s --check-prefix=SUBOP
// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins --lower-relalg-to-subop -subop-organize-execution-steps -subop-split-into-nested-steps | FileCheck %s --check-prefix=SPLIT
// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s --check-prefix=EXEC
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s --check-prefix=EXEC %}

// relalg.getrows (list-valued nested SQL, e.g. hipy's sql.rows(...)) returns
// all rows of its input as a !db.list of tuples. It is evaluated in place:
// the subtree it consumes is neither extracted before the map nor turned into
// a join.

// INPLACE-LABEL: func.func @main
// INPLACE-NOT: relalg.singlejoin
// INPLACE: relalg.map
// INPLACE: relalg.const_relation columns : [@inner::@k
// INPLACE: relalg.selection
// INPLACE: relalg.getrows {{.*}} [@inner::@k,@inner::@v] : !db.list<tuple<i64, !db.nullable<i64>>>

// Lowered to a nested query in the map lambda: the columns are materialized
// into a fresh buffer, which state_to_native turns into the list.
// SUBOP-LABEL: func.func @main
// SUBOP: subop.map
// SUBOP: %[[BUF:.+]] = subop.create !subop.buffer<[col$0 : i64, col$1 : !db.nullable<i64>]>
// SUBOP: subop.materialize {{.*}} {@inner::@k => col$0, @inner::@v => col$1}
// SUBOP: subop.state_to_native %[[BUF]] : !subop.buffer<[col$0 : i64, col$1 : !db.nullable<i64>]> -> !db.list<tuple<i64, !db.nullable<i64>>>
// SPLIT-LABEL: func.func @main
// SPLIT: subop.map
// SPLIT: subop.nested_execution_group
// SPLIT: subop.create !subop.buffer
// SPLIT: subop.materialize
// SPLIT: subop.state_to_native
// SPLIT: subop.nested_execution_group_return
// SPLIT: db.list_length

// per outer x: 100 * #rows + sum of k + 1000 * #NULLs in v, for the rows k <= x
// EXEC: |                           res  |
// EXEC: |                             0  |
// EXEC: |                          1203  |
// EXEC: |                          1306  |
module {
  func.func @main() {
    %0 = relalg.const_relation columns : [@outer::@x({type = i64})] values : [[3], [0], [2]]
    %1 = relalg.map %0 computes : [@m::@res({type = i64})] (%t: !tuples.tuple) {
      %n = tuples.getcol %t @outer::@x : i64
      %rel = relalg.const_relation columns : [@inner::@k({type = i64}), @inner::@v({type = !db.nullable<i64>})] values : [[1, 10], [2, unit], [3, 30]]
      %sel = relalg.selection %rel (%t2: !tuples.tuple) {
        %k = tuples.getcol %t2 @inner::@k : i64
        %le = db.compare lte %k : i64, %n : i64
        tuples.return %le : i1
      }
      %rows = relalg.getrows %sel [@inner::@k, @inner::@v] : !db.list<tuple<i64, !db.nullable<i64>>>
      %len = db.list_length %rows : !db.list<tuple<i64, !db.nullable<i64>>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant 0 : i64
      %acc = scf.for %i = %c0 to %len step %c1 iter_args(%a = %zero) -> (i64) {
        %row = db.list_get %rows : !db.list<tuple<i64, !db.nullable<i64>>>[%i] : tuple<i64, !db.nullable<i64>>
        %kv:2 = util.unpack %row : tuple<i64, !db.nullable<i64>> -> i64, !db.nullable<i64>
        %isnull = db.isnull %kv#1 : !db.nullable<i64>
        %thousand = arith.constant 1000 : i64
        %nulls = arith.select %isnull, %thousand, %zero : i64
        %a1 = arith.addi %a, %kv#0 : i64
        %a2 = arith.addi %a1, %nulls : i64
        scf.yield %a2 : i64
      }
      %lenI = arith.index_cast %len : index to i64
      %hundred = arith.constant 100 : i64
      %l100 = arith.muli %lenI, %hundred : i64
      %res = arith.addi %acc, %l100 : i64
      tuples.return %res : i64
    }
    %2 = relalg.sort %1 [(@outer::@x,asc)]
    %3 = relalg.materialize %2 [@m::@res] => ["res"] : !subop.local_table<[res$0 : i64], ["res"]>
    subop.set_result 0 %3 : !subop.local_table<[res$0 : i64], ["res"]>
    return
  }
}
