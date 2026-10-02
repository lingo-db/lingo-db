// RUN: mlir-db-opt %s -mlir-print-local-scope --relalg-extract-nested-operators --relalg-implicit-to-explicit-joins --lower-relalg-to-subop | FileCheck %s
// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s --check-prefix=EXEC
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s --check-prefix=EXEC %}

// relalg.getfirstrow with a nullable tuple as result type (hipy's
// sql.nullable(sql.row(...))): NULL instead of a runtime error if there is no
// row. The members of the simple_state are initialized with placeholders
// (strings: "", nullable: NULL) besides the "found" flag: reading the state
// back copies (and refcounts) the strings even if no row was found; before,
// they were uninitialized memory (random crashes in StringRuntime::addUse).

// CHECK-LABEL: func.func @main
// CHECK: subop.create_simple_state <[found$0 : i1, col$0 : i64, col$1 : !db.string, col$2 : !db.nullable<!db.string>]> initial
// CHECK-NEXT: db.constant(false) : i1
// CHECK-NEXT: util.undef : i64
// CHECK-NEXT: db.constant("") : !db.string
// CHECK-NEXT: db.null : <!db.string>
// CHECK-NEXT: tuples.return
// CHECK: subop.state_to_native {{.*}} -> tuple<i1, i64, !db.string, !db.nullable<!db.string>>
// CHECK: %[[ROW:[0-9]+]]:4 = util.unpack
// CHECK-NOT: raiseError
// CHECK: scf.if %[[ROW]]#0 -> (!db.nullable<tuple<i64, !db.string, !db.nullable<!db.string>>>)
// CHECK: util.pack %[[ROW]]#1, %[[ROW]]#2, %[[ROW]]#3
// CHECK: db.as_nullable
// CHECK: } else {
// CHECK: db.null : <tuple<i64, !db.string, !db.nullable<!db.string>>>

// x = 7: no row (NULL), x = 1: a row
// EXEC: |                           res  |
// EXEC: |                          null  |
// EXEC: |"b-string-not-inlined-0123456789"  |
module {
  func.func @main() {
    %0 = relalg.const_relation columns : [@outer::@x({type = i64})] values : [[7], [1]]
    %1 = relalg.map %0 computes : [@m::@res({type = !db.nullable<!db.string>})] (%t: !tuples.tuple) {
      %n = tuples.getcol %t @outer::@x : i64
      %rel = relalg.const_relation columns : [@inner::@k({type = i64}), @inner::@s({type = !db.string}), @inner::@t({type = !db.nullable<!db.string>})] values : [[0, "a-string-not-inlined-0123456789", "x"], [1, "b-string-not-inlined-0123456789", "y"]]
      %sel = relalg.selection %rel (%t2: !tuples.tuple) {
        %k = tuples.getcol %t2 @inner::@k : i64
        %eq = db.compare eq %k : i64, %n : i64
        tuples.return %eq : i1
      }
      %row = relalg.getfirstrow %sel [@inner::@k, @inner::@s, @inner::@t] : !db.nullable<tuple<i64, !db.string, !db.nullable<!db.string>>>
      %isnull = db.isnull %row : !db.nullable<tuple<i64, !db.string, !db.nullable<!db.string>>>
      %res = scf.if %isnull -> (!db.nullable<!db.string>) {
        %null = db.null : !db.nullable<!db.string>
        scf.yield %null : !db.nullable<!db.string>
      } else {
        %r = db.nullable_get_val %row : !db.nullable<tuple<i64, !db.string, !db.nullable<!db.string>>>
        %v:3 = util.unpack %r : tuple<i64, !db.string, !db.nullable<!db.string>> -> i64, !db.string, !db.nullable<!db.string>
        %s = db.as_nullable %v#1 : !db.string -> !db.nullable<!db.string>
        scf.yield %s : !db.nullable<!db.string>
      }
      tuples.return %res : !db.nullable<!db.string>
    }
    %2 = relalg.sort %1 [(@outer::@x,desc)]
    %3 = relalg.materialize %2 [@m::@res] => ["res"] : !subop.local_table<[res$0 : !db.nullable<!db.string>], ["res"]>
    subop.set_result 0 %3 : !subop.local_table<[res$0 : !db.nullable<!db.string>], ["res"]>
    return
  }
}
