//RUN: mlir-db-opt %s -subop-organize-execution-steps -subop-split-into-nested-steps -mlir-print-local-scope | FileCheck %s --check-prefix=SPLIT
//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

// subop.state_to_native from a buffer: a !db.list with one tuple per buffer
// entry (members in declaration order, entries in buffer order). Here in a
// nested query in a subop.map lambda (the shape of list-valued nested SQL,
// relalg.getrows): per outer n, the rows (i, i * 10) for i < n, with NULL
// instead of 10, are materialized into a fresh buffer and read back as a list.

//SPLIT: subop.map
//SPLIT: subop.nested_execution_group
//SPLIT: subop.create !subop.buffer<[a : i64, b : !db.nullable<i64>]>
//SPLIT: subop.materialize
//SPLIT: subop.state_to_native {{.*}} -> !db.list<tuple<i64, !db.nullable<i64>>>
//SPLIT: subop.nested_execution_group_return
//SPLIT: db.list_length

// per n: 100 * #rows + sum of (position + 1) * a (checks the order) + the
// non-NULL b's + 1000 * #NULLs. n = 0: 0 (an empty list); n = 3: rows
// (0, 0), (1, NULL), (2, 20): 300 + (1 * 0 + 2 * 1 + 3 * 2) + 20 + 1000 = 1328
// (reversed order: 1324)
//CHECK: |                         total  |
//CHECK: ----------------------------------
//CHECK-DAG: |                             0  |
//CHECK-DAG: |                          1328  |
module {
  func.func @main() {
    %result_table = subop.create !subop.result_table<[total : i64]>
    %generated, %streams:2 = subop.generate [@t::@n({type = i64})] {
      %c0 = arith.constant 0 : i64
      subop.generate_emit %c0 : i64
      %c3 = arith.constant 3 : i64
      subop.generate_emit %c3 : i64
      tuples.return
    }
    %mapped = subop.map %generated computes : [@m::@total({type = i64})] input : [@t::@n] (%n: i64) {
      %buffer = subop.create !subop.buffer<[a : i64, b : !db.nullable<i64>]>
      %rows, %row_streams = subop.generate [@g::@a({type = i64}), @g::@b({type = !db.nullable<i64>})] {
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        %c10 = arith.constant 10 : i64
        scf.for %i = %c0 to %n step %c1 : i64 {
          %v = arith.muli %i, %c10 : i64
          %nv = db.as_nullable %v : i64 -> !db.nullable<i64>
          %null = db.null : !db.nullable<i64>
          %isOne = arith.cmpi eq, %i, %c1 : i64
          %b = arith.select %isOne, %null, %nv : !db.nullable<i64>
          subop.generate_emit %i, %b : i64, !db.nullable<i64>
        }
        tuples.return
      }
      subop.materialize %rows {@g::@a => a, @g::@b => b}, %buffer : !subop.buffer<[a : i64, b : !db.nullable<i64>]>
      %list = subop.state_to_native %buffer : !subop.buffer<[a : i64, b : !db.nullable<i64>]> -> !db.list<tuple<i64, !db.nullable<i64>>>
      %len = db.list_length %list : !db.list<tuple<i64, !db.nullable<i64>>>
      %c0i = arith.constant 0 : index
      %c1i = arith.constant 1 : index
      %zero = arith.constant 0 : i64
      %acc = scf.for %i = %c0i to %len step %c1i iter_args(%s = %zero) -> (i64) {
        %row = db.list_get %list : !db.list<tuple<i64, !db.nullable<i64>>>[%i] : tuple<i64, !db.nullable<i64>>
        %ab:2 = util.unpack %row : tuple<i64, !db.nullable<i64>> -> i64, !db.nullable<i64>
        %isnull = db.isnull %ab#1 : !db.nullable<i64>
        %bv = db.nullable_get_val %ab#1 : !db.nullable<i64>
        %thousand = arith.constant 1000 : i64
        %bOr = arith.select %isnull, %thousand, %bv : i64
        %ii = arith.index_cast %i : index to i64
        %one = arith.constant 1 : i64
        %pos = arith.addi %ii, %one : i64
        %weighted = arith.muli %pos, %ab#0 : i64
        %s1 = arith.addi %s, %weighted : i64
        %s2 = arith.addi %s1, %bOr : i64
        scf.yield %s2 : i64
      }
      %lenI = arith.index_cast %len : index to i64
      %hundred = arith.constant 100 : i64
      %l100 = arith.muli %lenI, %hundred : i64
      %total = arith.addi %acc, %l100 : i64
      tuples.return %total : i64
    }
    subop.materialize %mapped {@m::@total => total}, %result_table : !subop.result_table<[total : i64]>
    %local_table = subop.create_from ["total"] %result_table : !subop.result_table<[total : i64]> -> !subop.local_table<[total : i64], ["total"]>
    subop.set_result 0 %local_table : !subop.local_table<[total : i64], ["total"]>
    return
  }
}
