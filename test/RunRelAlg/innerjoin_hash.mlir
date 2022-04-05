//RUN: mlir-db-opt -relalg-to-db -canonicalize %s | db-run-query "-" %S/../../resources/data/uni | FileCheck %s
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

module @querymodule{
    func @main ()  -> !dsa.table{
        %1 = relalg.basetable @hoeren { table_identifier="hoeren" } columns: {matrnr => @matrnr({type=i64}),
            vorlnr => @vorlnr({type=i64})
        }
        %2 = relalg.basetable @studenten { table_identifier="studenten" } columns: {matrnr => @matrnr({type=i64}),
            name => @name({type=!db.string}),
            semester => @semester({type=i64})
        }
        %3 = relalg.join %1, %2 (%6: !relalg.tuple) {
                                                 %8 = relalg.getcol %6 @hoeren::@matrnr : i64
                                                 %9 = relalg.getcol %6 @studenten::@matrnr : i64
                                                 %10 = db.compare eq %8 : i64,%9 : i64
                                                 relalg.return %10 : i1
                                             } attributes { impl="hash" }
        %4 = relalg.basetable @vorlesungen { table_identifier="vorlesungen" } columns: {vorlnr => @vorlnr({type=i64}),
            titel => @titel({type=!db.string}),
            sws => @sws({type=i64}),
            gelesenvon => @gelesenvon({type=i64})
        }
        %5 = relalg.join %3, %4 (%6: !relalg.tuple) {
            %11 = relalg.getcol %6 @hoeren::@vorlnr : i64
            %12 = relalg.getcol %6 @vorlesungen::@vorlnr : i64
            %13 = db.compare eq %11 : i64,%12 : i64
            relalg.return %13 : i1
        } attributes { impl="hash" }
        %15 = relalg.materialize %5 [@studenten::@name,@vorlesungen::@titel] => ["s.name","v.titel"] : !dsa.table
        return %15 : !dsa.table
    }
}
