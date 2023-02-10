//RUN: run-mlir %s %S/../../../resources/data/uni | FileCheck %s
//CHECK: |                            id  |                      original  |                         ngram  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                             1  |                  "first word"  |                         "fir"  |
//CHECK: |                             1  |                  "first word"  |                         "irs"  |
//CHECK: |                             1  |                  "first word"  |                         "rst"  |
//CHECK: |                             1  |                  "first word"  |                         "st "  |
//CHECK: |                             1  |                  "first word"  |                         "t w"  |
//CHECK: |                             1  |                  "first word"  |                         " wo"  |
//CHECK: |                             1  |                  "first word"  |                         "wor"  |
//CHECK: |                             1  |                  "first word"  |                         "ord"  |
//CHECK: |                             2  |                 "second word"  |                         "sec"  |
//CHECK: |                             2  |                 "second word"  |                         "eco"  |
//CHECK: |                             2  |                 "second word"  |                         "con"  |
//CHECK: |                             2  |                 "second word"  |                         "ond"  |
//CHECK: |                             2  |                 "second word"  |                         "nd "  |
//CHECK: |                             2  |                 "second word"  |                         "d w"  |
//CHECK: |                             2  |                 "second word"  |                         " wo"  |
//CHECK: |                             2  |                 "second word"  |                         "wor"  |
//CHECK: |                             2  |                 "second word"  |                         "ord"  |

module @querymodule{
    func.func @main () {
        %1 = relalg.const_relation columns : [@constrel::@attr1({type = i64}),@constrel::@attr2({type = !db.string})] values : [[1,"first word"], [2,"second word"]]
        %2 = relalg.nested %1 [@constrel::@attr1,@constrel::@attr2] -> [@constrel::@attr1,@constrel::@attr2,@nested::@ngram] (%stream){
            %splitted = subop.nested_map %stream [@constrel::@attr2](%t, %str){
                %generated = subop.generate [@nested::@ngram({type=!db.string})] {
                        %c0 = arith.constant 0 : index
                        %c1 = arith.constant 1 : index
                        %n = arith.constant 3 : index // n as in n-grams
                        %len64 = db.runtime_call "StringLength" (%str) : (!db.string) -> i64
                        %len = arith.index_castui %len64 : i64 to index
                        %lenLtN = arith.cmpi ult, %len, %n : index
                        scf.if %lenLtN {
                            subop.generate_emit %str : !db.string
                        } else {
                            %lenMn = arith.subi %len, %n : index
                            %lenMnP1 = arith.addi %lenMn,%c1 : index
                            scf.for %i = %c0 to %lenMnP1 step %c1 {
                              %iP1 = arith.addi %i, %c1 : index
                              %iPn = arith.addi %i, %n : index
                              %substr = db.runtime_call "Substring" (%str, %iP1, %iPn) : (!db.string, index,index) -> !db.string
                              subop.generate_emit %substr : !db.string
                            }

                        }
                        tuples.return
                    }
                tuples.return %generated : !tuples.tuplestream
            }
            tuples.return %splitted : !tuples.tuplestream
        }
        %3 = relalg.materialize %2 [@constrel::@attr1,@constrel::@attr2,@nested::@ngram] => ["id","original","ngram"] : !subop.result_table<[id : i64,original: !db.string,ngram: !db.string]>
        subop.set_result 0 %3 : !subop.result_table<[id : i64,original: !db.string,ngram: !db.string]>
        return
    }
}
