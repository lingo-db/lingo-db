//RUN: mlir-db-opt -lower-relalg %s | run-mlir "-" %S/../../resources/data/uni | FileCheck %s
//CHECK: |                        s.name  |
//CHECK: ----------------------------------
//CHECK: |                  "Xenokrates"  |
//CHECK: |                 "Aristoxenos"  |



module @querymodule{
    func @main ()  -> !dsa.table{
        %1 = relalg.basetable { table_identifier="hoeren" } columns: {matrnr => @hoeren::@matrnr({type=i64}),
            vorlnr => @hoeren::@vorlnr({type=i64})
        }
        %2 = relalg.basetable { table_identifier="studenten" } columns: {matrnr => @studenten::@matrnr({type=i64}),
            name => @studenten::@name({type=!db.string}),
            semester => @studenten::@semester({type=i64})
        }
        %3 = relalg.antisemijoin %2, %1 (%6: !relalg.tuple) {
                                                 %8 = relalg.getcol %6 @hoeren::@matrnr : i64
                                                 %9 = relalg.getcol %6 @studenten::@matrnr : i64
                                                 %10 = db.compare eq %8 : i64,%9 : i64
                                                 relalg.return %10 : i1
                                             } attributes { impl="hash" }

        %15 = relalg.materialize %3 [@studenten::@name] => ["s.name"] : !dsa.table
        return %15 : !dsa.table
    }
}