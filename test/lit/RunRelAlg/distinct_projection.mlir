//RUN: mlir-db-opt -lower-relalg-to-subop %s | run-mlir "-" %S/../../../resources/data/uni | FileCheck %s
//CHECK: |                        matrnr  |
//CHECK: ----------------------------------
//CHECK: |                         25403  |
//CHECK: |                         26120  |
//CHECK: |                         27550  |
//CHECK: |                         28106  |
//CHECK: |                         29120  |
//CHECK: |                         29555  |


module @querymodule{
    func.func @main () {
        %1 = relalg.basetable { table_identifier="hoeren" } columns: {matrnr => @hoeren::@matrnr({type=i64}),
            vorlnr => @hoeren::@vorlnr({type=i64})
        }
        %2 = relalg.projection distinct  [@hoeren::@matrnr] %1
        %4 = relalg.sort %2 [(@hoeren::@matrnr,asc)]
        %3 = relalg.materialize %4 [@hoeren::@matrnr] => ["matrnr"]  : !subop.result_table<[hmatrnr : i64]>
        subop.set_result 0 %3 : !subop.result_table<[hmatrnr : i64]>
        return
    }
}
