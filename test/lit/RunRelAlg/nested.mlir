//RUN: run-mlir %s %S/../../../resources/data/uni | FileCheck %s
//CHECK: |                        matrnr  |                          name  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                         24002  |                  "Xenokrates"  |
//CHECK: |                         25403  |                       "Jonas"  |
//CHECK: |                         26120  |                      "Fichte"  |
//CHECK: |                         26830  |                 "Aristoxenos"  |

module @querymodule{
    func.func @main () {
        %1 = relalg.basetable { table_identifier="studenten" } columns: {matrnr => @studenten::@matrnr({type=i64}),
            name => @studenten::@name({type=!db.string}),
            semester => @studenten::@semester({type=i64})
        }
        %2 = relalg.nested %1 [@studenten::@matrnr,@studenten::@name] -> [@studenten::@matrnr, @studenten::@name] (%stream){
            %mapped = subop.map %stream computes : [@studenten::@matlt({type=i1})] (%tpl: !tuples.tuple){
                %matrnr = tuples.getcol %tpl @studenten::@matrnr : i64
                %c27550 = arith.constant 27550 : i64
                %lt = arith.cmpi slt, %matrnr, %c27550 : i64
                tuples.return %lt : i1
            }
            %filtered = subop.filter %mapped all_true [@studenten::@matlt]
            tuples.return %filtered : !tuples.tuplestream
        }
        %3 = relalg.materialize %2 [@studenten::@matrnr,@studenten::@name] => ["matrnr","name"] : !subop.result_table<[hmatrnr : i64,sname: !db.string]>
        subop.set_result 0 %3 : !subop.result_table<[hmatrnr : i64,sname: !db.string]>
        return
    }
}
