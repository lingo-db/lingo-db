//RUN: run-mlir %s %S/../../../resources/data/uni | FileCheck %s
//CHECK: |                            id  |                      original  |                      splitted  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                             1  |                  "first word"  |                       "first"  |
//CHECK: |                             1  |                  "first word"  |                        "word"  |
//CHECK: |                             2  |                 "second word"  |                      "second"  |
//CHECK: |                             2  |                 "second word"  |                        "word"  |

module @querymodule{
    func.func @main () {
        %1 = relalg.const_relation columns : [@constrel::@attr1({type = i64}),@constrel::@attr2({type = !db.string})] values : [[1,"first word"], [2,"second word"]]
        %2 = relalg.nested %1 [@constrel::@attr1,@constrel::@attr2] -> [@constrel::@attr1,@constrel::@attr2,@nested::@splitted] (%stream){
            %splitted = subop.nested_map %stream [@constrel::@attr2](%t, %str){
                %generated = subop.generate [@nested::@splitted({type=!db.string})] {
                        %npos = arith.constant 0x8000000000000000 : i64
                        %c0 = arith.constant 0 : i64
                        %c1 = arith.constant 1 : i64
                        %split_pattern=db.constant(" ") : !db.string

                        %res = scf.while (%currPos = %c0) : (i64) -> i64 {
                          %valid = arith.cmpi ult, %currPos, %npos : i64
                          scf.condition(%valid) %currPos : i64
                        } do {
                        ^bb0(%currPos : i64):
                          %next = db.runtime_call "StringFind" (%str,%split_pattern,%currPos) : (!db.string,!db.string,i64) -> i64
                          %currPos1 = arith.addi %currPos, %c1 : i64
                          %substr = db.runtime_call "Substring" (%str, %currPos1, %next) : (!db.string, i64,i64) -> !db.string
                          subop.generate_emit %substr : !db.string
                          %nextStartPos = arith.addi %next, %c1 : i64
                          scf.yield %nextStartPos : i64
                        }
                        tuples.return
                    }
                tuples.return %generated : !tuples.tuplestream
            }
            tuples.return %splitted : !tuples.tuplestream
        }
        %3 = relalg.materialize %2 [@constrel::@attr1,@constrel::@attr2,@nested::@splitted] => ["id","original","splitted"] : !subop.result_table<[id : i64,original: !db.string,splitted: !db.string]>
        subop.set_result 0 %3 : !subop.result_table<[id : i64,original: !db.string,splitted: !db.string]>
        return
    }
}
