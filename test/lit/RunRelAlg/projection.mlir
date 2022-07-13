//RUN: mlir-db-opt -lower-relalg-to-subop %s | run-mlir "-" %S/../../../resources/data/uni | FileCheck %s
//CHECK: |                        matrnr  |
//CHECK: ----------------------------------
//CHECK: |                         26120  |
//CHECK: |                         27550  |
//CHECK: |                         27550  |
//CHECK: |                         28106  |
//CHECK: |                         28106  |
//CHECK: |                         28106  |
//CHECK: |                         28106  |
//CHECK: |                         29120  |
//CHECK: |                         29120  |
//CHECK: |                         29120  |
//CHECK: |                         29555  |
//CHECK: |                         25403  |
//CHECK: |                         29555  |

module @querymodule{
    func.func @main ()  -> !dsa.table{
        %1 = relalg.basetable { table_identifier="hoeren" } columns: {matrnr => @hoeren::@matrnr({type=i64}),
            vorlnr => @hoeren::@vorlnr({type=i64})
        }
        %2 = relalg.projection all  [@hoeren::@matrnr] %1
        %3 = relalg.materialize %2 [@hoeren::@matrnr] => ["matrnr"] : !dsa.table
        return %3 : !dsa.table
    }
}
