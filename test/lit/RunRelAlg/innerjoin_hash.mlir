//RUN: mlir-db-opt -lower-relalg-to-subop %s | run-mlir "-" %S/../../../resources/data/uni | FileCheck %s
//CHECK: |                        s.name  |                       v.titel  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                   "Feuerbach"  |                  "Grundzuege"  |
//CHECK: |                "Theophrastos"  |                  "Grundzuege"  |
//CHECK: |                "Schopenhauer"  |                  "Grundzuege"  |
//CHECK: |                      "Fichte"  |                  "Grundzuege"  |
//CHECK: |                "Theophrastos"  |                       "Ethik"  |
//CHECK: |                      "Carnap"  |                       "Ethik"  |
//CHECK: |                "Theophrastos"  |                    "Maeeutik"  |
//CHECK: |                "Schopenhauer"  |                       "Logik"  |
//CHECK: |                      "Carnap"  |        "Wissenschaftstheorie"  |
//CHECK: |                      "Carnap"  |                    "Bioethik"  |
//CHECK: |                      "Carnap"  |            "Der Wiener Kreis"  |
//CHECK: |                   "Feuerbach"  |           "Glaube und Wissen"  |
//CHECK: |                       "Jonas"  |           "Glaube und Wissen"  |

module @querymodule {
  func.func @main() -> !dsa.table {
    %0 = relalg.basetable  {rows = 0.000000e+00 : f64, table_identifier = "hoeren"} columns: {matrnr => @hoeren::@matrnr({type = i64}), vorlnr => @hoeren::@vorlnr({type = i64})}
    %1 = relalg.basetable  {rows = 0.000000e+00 : f64, table_identifier = "studenten"} columns: {matrnr => @studenten::@matrnr({type = i64}), name => @studenten::@name({type = !db.string}), semester => @studenten::@semester({type = i64})}
    %2 = relalg.join %0, %1 (%arg0: !tuples.tuple){
      %6 = tuples.getcol %arg0 @hoeren::@matrnr : i64
      %7 = tuples.getcol %arg0 @studenten::@matrnr : i64
      %8 = db.compare eq %6 : i64, %7 : i64
      tuples.return %8 : i1
    } attributes {cost = 2.100000e+00 : f64, impl = "hash", leftHash = [#tuples.columnref<@hoeren::@matrnr>], rightHash = [#tuples.columnref<@studenten::@matrnr>], rows = 1.000000e-01 : f64, useHashJoin}
    %3 = relalg.basetable  {rows = 0.000000e+00 : f64, table_identifier = "vorlesungen"} columns: {gelesenvon => @vorlesungen::@gelesenvon({type = i64}), sws => @vorlesungen::@sws({type = i64}), titel => @vorlesungen::@titel({type = !db.string}), vorlnr => @vorlesungen::@vorlnr({type = i64})}
    %4 = relalg.join %2, %3 (%arg0: !tuples.tuple){
      %6 = tuples.getcol %arg0 @hoeren::@vorlnr : i64
      %7 = tuples.getcol %arg0 @vorlesungen::@vorlnr : i64
      %8 = db.compare eq %6 : i64, %7 : i64
      tuples.return %8 : i1
    } attributes {cost = 3.110000e+00 : f64, impl = "hash", leftHash = [#tuples.columnref<@hoeren::@vorlnr>], rightHash = [#tuples.columnref<@vorlesungen::@vorlnr>], rows = 0.010000000000000002 : f64, useHashJoin}
    %5 = relalg.materialize %4 [@studenten::@name,@vorlesungen::@titel] => ["s.name", "v.titel"] : !dsa.table
    return %5 : !dsa.table
  }
}