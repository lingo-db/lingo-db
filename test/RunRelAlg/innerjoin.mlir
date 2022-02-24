//RUN: mlir-db-opt -relalg-to-db -canonicalize %s | db-run "-" %S/../../resources/data/uni | FileCheck %s
//CHECK: |                        s.name  |                       v.titel  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                      "Fichte"  |                  "Grundzuege"  |
//CHECK: |                "Schopenhauer"  |                  "Grundzuege"  |
//CHECK: |                "Theophrastos"  |                  "Grundzuege"  |
//CHECK: |                   "Feuerbach"  |                  "Grundzuege"  |
//CHECK: |                      "Carnap"  |                       "Ethik"  |
//CHECK: |                "Theophrastos"  |                       "Ethik"  |
//CHECK: |                "Theophrastos"  |                    "Maeeutik"  |
//CHECK: |                "Schopenhauer"  |                       "Logik"  |
//CHECK: |                      "Carnap"  |        "Wissenschaftstheorie"  |
//CHECK: |                      "Carnap"  |                    "Bioethik"  |
//CHECK: |                      "Carnap"  |            "Der Wiener Kreis"  |
//CHECK: |                       "Jonas"  |           "Glaube und Wissen"  |
//CHECK: |                   "Feuerbach"  |           "Glaube und Wissen"  |

module @querymodule{
    func @main ()  -> !db.table{
        %1 = relalg.basetable @hoeren { table_identifier="hoeren" } columns: {matrnr => @matrnr({type=!db.int<64>}),
            vorlnr => @vorlnr({type=!db.int<64>})
        }
        %2 = relalg.basetable @studenten { table_identifier="studenten" } columns: {matrnr => @matrnr({type=!db.int<64>}),
            name => @name({type=!db.string}),
            semester => @semester({type=!db.int<64>})
        }
        %3 = relalg.join %1, %2 (%6: !relalg.tuple) {
                                                 %8 = relalg.getattr %6 @hoeren::@matrnr : !db.int<64>
                                                 %9 = relalg.getattr %6 @studenten::@matrnr : !db.int<64>
                                                 %10 = db.compare eq %8 : !db.int<64>,%9 : !db.int<64>
                                                 relalg.return %10 : i1
                                             }
        %4 = relalg.basetable @vorlesungen { table_identifier="vorlesungen" } columns: {vorlnr => @vorlnr({type=!db.int<64>}),
            titel => @titel({type=!db.string}),
            sws => @sws({type=!db.int<64>}),
            gelesenvon => @gelesenvon({type=!db.int<64>})
        }
        %5 = relalg.join %3, %4 (%6: !relalg.tuple) {
            %11 = relalg.getattr %6 @hoeren::@vorlnr : !db.int<64>
            %12 = relalg.getattr %6 @vorlesungen::@vorlnr : !db.int<64>
            %13 = db.compare eq %11 : !db.int<64>,%12 : !db.int<64>
            relalg.return %13 : i1
        }
        %15 = relalg.materialize %5 [@studenten::@name,@vorlesungen::@titel] => ["s.name","v.titel"] : !db.table
        return %15 : !db.table
    }
}
