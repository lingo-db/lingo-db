//RUN: mlir-db-opt -lower-relalg %s | run-mlir "-" %S/../../resources/data/uni | FileCheck %s
//CHECK: |                        matrnr  |                          name  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                         24002  |                  "Xenokrates"  |
//CHECK: |                         25403  |                       "Jonas"  |
//CHECK: |                         26120  |                      "Fichte"  |
//CHECK: |                         26830  |                 "Aristoxenos"  |
//CHECK: |                         27550  |                "Schopenhauer"  |
//CHECK: |                         28106  |                      "Carnap"  |
//CHECK: |                         29120  |                "Theophrastos"  |
//CHECK: |                         29555  |                   "Feuerbach"  |

module @querymodule{
    func @main ()  -> !dsa.table{
        %1 = relalg.basetable { table_identifier="studenten" } columns: {matrnr => @studenten::@matrnr({type=i64}),
            name => @studenten::@name({type=!db.string}),
            semester => @studenten::@semester({type=i64})
        }
        %2 = relalg.renaming %1  renamed: [@renamed::@matrnr({type = i64})=[@studenten::@matrnr]]
        %3 = relalg.materialize %2 [@renamed::@matrnr,@studenten::@name] => ["matrnr","name"] : !dsa.table
        return %3 : !dsa.table
    }
}
