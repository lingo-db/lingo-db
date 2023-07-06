//RUN: run-mlir %s %S/../../../resources/data/uni | FileCheck %s

//CHECK: |                            id  |                      original  |                      matching  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                             1  |                 "  jhonson  "  |                             1  |
//CHECK: |                             2  |   "this is another long text"  |                             3  |
//CHECK: |                             2  |   "this is another long text"  |                             2  |
//CHECK: |                             3  |        "why not testing this"  |                             3  |
//CHECK: |                             3  |        "why not testing this"  |                             2  |
module @querymodule{
    func.func @main () {
        %rel1 = relalg.const_relation columns : [@rel1::@attr1({type = i64}),@rel1::@attr2({type = !db.string})] values : [[1,"  johnson  "],[2,"this is a very long text"],[3,"this is a test"]]
        %rel2 = relalg.const_relation columns : [@rel2::@attr1({type = i64}),@rel2::@attr2({type = !db.string})] values : [[1,"  jhonson  "],[2,"this is another long text"],[3,"why not testing this"]]

        %nested = relalg.nested %rel1,%rel2 [@rel1::@attr1,@rel1::@attr2,@rel2::@attr1,@rel2::@attr2] -> [@rel2::@attr1,@rel2::@attr1] (%stream1, %stream2){
            %ngram_index = subop.create !subop.multimap<[ ngram_ht_1 : !db.string],[ id_ht_1 : i64]>
            %rel1_splitted = subop.nested_map %stream1 [@rel1::@attr2](%t, %str){
                %splitted = subop.generate [@rel1::@ngram({type=!db.string})] {
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
                tuples.return %splitted : !tuples.tuplestream
            }
            subop.insert %rel1_splitted %ngram_index :   !subop.multimap<[ ngram_ht_1 : !db.string],[ id_ht_1 : i64]> {@rel1::@attr1 => id_ht_1, @rel1::@ngram => ngram_ht_1} eq: ([%l],[%r]){
                %matches = db.compare eq %l : !db.string, %r : !db.string
                tuples.return %matches : i1
            }
            %rel2_splitted = subop.nested_map %stream2 [@rel2::@attr2](%t, %str){
                %cnt_map = subop.create !subop.hashmap<[ id_ht2 : i64],[ cnt : i64]>
                %splitted = subop.generate [@rel2::@ngram({type=!db.string})] {
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
                %rel2_lookup = subop.lookup %splitted %ngram_index [@rel2::@ngram] :  !subop.multimap<[ ngram_ht_1 : !db.string],[ id_ht_1 : i64]>  @ngramlist::@lookup_ref_list({type=!subop.list<!subop.lookup_entry_ref<!subop.multimap<[ ngram_ht_1 : !db.string],[ id_ht_1 : i64]>>>}) eq: ([%l],[%r]){
                    %matches = db.compare eq %l : !db.string, %r : !db.string
                    tuples.return %matches : i1
                }
                %rel2_matches = subop.nested_map %rel2_lookup [@ngramlist::@lookup_ref_list](%t2, %list){
                    %scan_matches = subop.scan_list %list : !subop.list<!subop.lookup_entry_ref<!subop.multimap<[ ngram_ht_1 : !db.string],[ id_ht_1 : i64]>>> @ngramlist::@lookup_ref({type=!subop.lookup_entry_ref<!subop.multimap<[ ngram_ht_1 : !db.string],[ id_ht_1 : i64]>>})
                    %combined = subop.combine_tuple %scan_matches, %t2
                    %gathered = subop.gather %combined @ngramlist::@lookup_ref { ngram_ht_1 => @ngramlist::@ngram({type= !db.string}),  id_ht_1 => @ngramlist::@id({type=i64}) }
                    %lookedUp =subop.lookup_or_insert %gathered %cnt_map[@ngramlist::@id] : !subop.hashmap<[ id_ht2 : i64],[ cnt : i64]> @cntmap::@ref({type=!subop.lookup_entry_ref<!subop.hashmap<[ id_ht2 : i64],[ cnt : i64]>>})
                        eq: ([%l], [%r]){
                            %eq = arith.cmpi eq, %l, %r :i64
                            tuples.return %eq : i1
                        }
                        initial: {
                            %zero = arith.constant 0 : i64
                            tuples.return %zero : i64
                        }
                    %gatheredCnt = subop.gather %lookedUp @cntmap::@ref { cnt => @cntmap::@cnt({type=i64}) }
                    %incCnt = subop.map %gatheredCnt computes : [@cntmap::@newcnt({type=i64}), @cntmap::@emit({type=i1})] (%tpl: !tuples.tuple){
                        %currCount = tuples.getcol %tpl @cntmap::@cnt : i64
                        %c1 = arith.constant 1 : i64
                        %nextCount = arith.addi %currCount, %c1 : i64
                        %emitCount = arith.constant 5 : i64 //todo: compute according to n and string length
                        %emit = arith.cmpi eq, %nextCount, %emitCount : i64
                        tuples.return %nextCount, %emit : i64, i1
                    }
                    subop.scatter %incCnt @cntmap::@ref { @cntmap::@newcnt => cnt }
                    %filteredWithCnt = subop.filter %incCnt all_true [@cntmap::@emit]
                    tuples.return %filteredWithCnt : !tuples.tuplestream
                }
                tuples.return %rel2_matches : !tuples.tuplestream
            }


            tuples.return %rel2_splitted: !tuples.tuplestream
        }


        %3 = relalg.materialize %nested [@rel2::@attr1,@rel2::@attr2,@ngramlist::@id] => ["id","original","matching"] : !subop.result_table<[id : i64,original: !db.string,matching_id: i64]>
        subop.set_result 0 %3 :  !subop.result_table<[id : i64,original: !db.string,matching_id: i64]>
        return
    }
}
