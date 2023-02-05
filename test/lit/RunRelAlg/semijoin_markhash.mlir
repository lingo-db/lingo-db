//RUN: mlir-db-opt -lower-relalg-to-subop %s | run-mlir "-" %S/../../../resources/data/uni | FileCheck %s
//CHECK: |                        s.name  |
//CHECK: ----------------------------------
//CHECK: |                       "Jonas"  |
//CHECK: |                      "Fichte"  |
//CHECK: |                "Schopenhauer"  |
//CHECK: |                      "Carnap"  |
//CHECK: |                "Theophrastos"  |
//CHECK: |                   "Feuerbach"  |



module @querymodule {
  func.func @main() {
    %0 = relalg.basetable  {rows = 0.000000e+00 : f64, table_identifier = "hoeren"} columns: {matrnr => @hoeren::@matrnr({type = i64}), vorlnr => @hoeren::@vorlnr({type = i64})}
    %1 = relalg.basetable  {rows = 0.000000e+00 : f64, table_identifier = "studenten"} columns: {matrnr => @studenten::@matrnr({type = i64}), name => @studenten::@name({type = !db.string}), semester => @studenten::@semester({type = i64})}
    %2 = relalg.semijoin %1, %0 (%arg0: !tuples.tuple){
      %true = db.constant(1) : i1
      tuples.return %true : i1
    } attributes {cost = 2.100000e+00 : f64, impl = "hash", leftHash = [#tuples.columnref<@studenten::@matrnr>], rightHash = [#tuples.columnref<@hoeren::@matrnr>], nullsEqual = [0 : i8], rows = 1.000000e-01 : f64, useHashJoin,reverseSides }
    %15 = relalg.materialize %2 [@studenten::@name] => ["s.name"] : !subop.result_table<[sname: !db.string]>
    subop.set_result 0 %15 : !subop.result_table<[sname: !db.string]>
    return
  }
}
