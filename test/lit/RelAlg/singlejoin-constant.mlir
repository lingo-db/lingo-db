// RUN: mlir-db-opt %s -mlir-print-local-scope --lower-relalg-to-subop | FileCheck %s
// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s --check-prefix=EXEC
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s --check-prefix=EXEC %}

// A "constant" single join (uncorrelated scalar subquery) scatters the right
// side's row into a simple_state and gathers it for every left row. If the
// right side has no row, the result is NULL: a "found" flag (initially false)
// decides, and the other members get placeholders (strings: ""). Before, the
// state was read uninitialized (random values instead of NULL, and garbage
// strings for refcounting).

// CHECK-LABEL: func.func @main
// CHECK: subop.create_simple_state <[found$0 : i1, member$0 : !db.string]> initial
// CHECK-NEXT: db.constant(false) : i1
// CHECK-NEXT: db.constant("") : !db.string
// CHECK-NEXT: tuples.return
// CHECK: subop.scatter {{.*}}@singlejoin::@found => found$0
// CHECK: subop.gather
// CHECK: subop.map
// CHECK: scf.if
// CHECK: db.as_nullable
// CHECK: } else {
// CHECK: db.null : <!db.string>

// EXEC: |                             x  |                             s  |
// EXEC: |                             1  |                          null  |
// EXEC: |                             2  |                          null  |
module {
  func.func @main() {
    %0 = relalg.const_relation columns : [@outer::@x({type = i64})] values : [[1], [2]]
    %1 = relalg.const_relation columns : [@inner::@k({type = i64}), @inner::@s({type = !db.string})] values : [[0, "a-string-not-inlined-0123456789"]]
    %2 = relalg.selection %1 (%t: !tuples.tuple) {
      %k = tuples.getcol %t @inner::@k : i64
      %c5 = db.constant(5 : i64) : i64
      %eq = db.compare eq %k : i64, %c5 : i64
      tuples.return %eq : i1
    }
    %3 = relalg.singlejoin %0, %2 (%t: !tuples.tuple) {
      tuples.return
    } mapping: {@sj::@s({type = !db.nullable<!db.string>})=[@inner::@s]} attributes {constantJoin}
    %4 = relalg.sort %3 [(@outer::@x,asc)]
    %5 = relalg.materialize %4 [@outer::@x, @sj::@s] => ["x", "s"] : !subop.local_table<[x$0 : i64, s$0 : !db.nullable<!db.string>], ["x", "s"]>
    subop.set_result 0 %5 : !subop.local_table<[x$0 : i64, s$0 : !db.nullable<!db.string>], ["x", "s"]>
    return
  }
}
