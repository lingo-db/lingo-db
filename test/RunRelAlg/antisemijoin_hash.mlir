//RUN: mlir-db-opt -relalg-to-db -canonicalize %s | db-run "-" %S/../../resources/data/uni | FileCheck %s
//CHECK: |                        s.name  |
//CHECK: ----------------------------------
//CHECK: |                  "Xenokrates"  |
//CHECK: |                 "Aristoxenos"  |



module @querymodule{
    func @main ()  -> !db.table{
        %1 = relalg.basetable @hoeren { table_identifier="hoeren" } columns: {matrnr => @matrnr({type=!db.int<64>}),
            vorlnr => @vorlnr({type=!db.int<64>})
        }
        %2 = relalg.basetable @studenten { table_identifier="studenten" } columns: {matrnr => @matrnr({type=!db.int<64>}),
            name => @name({type=!db.string}),
            semester => @semester({type=!db.int<64>})
        }
        %3 = relalg.antisemijoin %2, %1 (%6: !relalg.tuple) {
                                                 %8 = relalg.getattr %6 @hoeren::@matrnr : !db.int<64>
                                                 %9 = relalg.getattr %6 @studenten::@matrnr : !db.int<64>
                                                 %10 = db.compare eq %8 : !db.int<64>,%9 : !db.int<64>
                                                 relalg.return %10 : !db.bool
                                             } attributes { impl="hash" }

        %15 = relalg.materialize %3 [@studenten::@name] => ["s.name"] : !db.table
        return %15 : !db.table
    }
}