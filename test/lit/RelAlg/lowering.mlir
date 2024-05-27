// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --lower-relalg-to-subop | FileCheck %s
//CHECK: %{{.*}} = subop.generate[@t::@col1({type = !db.string})]{
//CHECK:   %{{.*}} = db.constant("A") : !db.string
//CHECK:   subop.generate_emit %{{.*}} : !db.string
//CHECK:   %{{.*}} = db.constant("B") : !db.string
//CHECK:   subop.generate_emit %{{.*}} : !db.string
//CHECK:   tuples.return
//CHECK: }
%0 = relalg.const_relation columns : [@t::@col1({type = !db.string})] values : [["A"],["B"]]
// -----
//CHECK:   %{{.*}} = subop.rename %{{.*}} renamed : [@renamed::@col2({type = !db.string})=[@t::@col1]]
%0 = relalg.const_relation columns : [@t::@col1({type = !db.string})] values : [["A"],["B"]]
%1 = relalg.renaming %0  renamed: [@renamed::@col2({type = !db.string})=[@t::@col1]]
// -----
//CHECK: module {
//CHECK:   %{{.*}}, %{{.*}} = subop.generate[@t::@col1({type = !db.string})]{
//CHECK:     %{{.*}} = db.constant("A") : !db.string
//CHECK:     subop.generate_emit %{{.*}} : !db.string
//CHECK:     %{{.*}} = db.constant("B") : !db.string
//CHECK:     subop.generate_emit %{{.*}} : !db.string
//CHECK:     tuples.return
//CHECK:   }
//CHECK: }
%0 = relalg.const_relation columns : [@t::@col1({type = !db.string})] values : [["A"],["B"]]
%1 = relalg.projection all  [@t::@col1] %0
// -----
//CHECK: [[INITIAL:%.*]], %{{.*}}:2 = subop.generate
//CHECK: [[MAP:%.*]]  = subop.create !subop.map<[keyval$0 : !db.string], []>
//CHECK: %{{.*}} = subop.lookup_or_insert [[INITIAL]][[MAP]] [@t::@col1] : !subop.map<[keyval$0 : !db.string], []> @lookup::@ref({type = !subop.lookup_entry_ref<!subop.map<[keyval$0 : !db.string], []>>}) eq: ([%arg0],[%arg1]) {
//CHECK:   %true = arith.constant true
//CHECK:   %{{.*}} = db.compare isa %arg0 : !db.string, %arg1 : !db.string
//CHECK:   %{{.*}} = arith.andi %true, %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }initial: {
//CHECK:   tuples.return
//CHECK: }
//CHECK: %{{.*}} = subop.scan [[MAP]] : !subop.map<[keyval$0 : !db.string], []> {keyval$0 => @t::@col1({type = !db.string})}

%0 = relalg.const_relation columns : [@t::@col1({type = !db.string})] values : [["A"],["B"]]
%1 = relalg.projection distinct  [@t::@col1] %0
// -----
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK: %{{.*}} = subop.create !subop.buffer
//CHECK: subop.materialize [[LEFT]] {@t::@col1 => member$0}, %{{.*}} : !subop.buffer
//CHECK: %{{.*}} = subop.nested_map [[RIGHT]] [] (%arg0) {
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$0 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.scan %{{.*}} : !subop.buffer
//CHECK:   %{{.*}} = subop.combine_tuple %{{.*}}, %arg0
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map::@pred({type = i1})] (%arg1: !tuples.tuple){
//CHECK:     %{{.*}} = tuples.getcol %arg1 @t::@col1 : i64
//CHECK:     %{{.*}} = tuples.getcol %arg1 @t2::@col1 : i64
//CHECK:     %{{.*}} = db.compare eq %{{.*}} : i64, %{{.*}} : i64
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@map::@pred]
//CHECK:   tuples.return %{{.*}} : !tuples.tuplestream
//CHECK: }

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.join %0, %1 (%6: !tuples.tuple) {
	 %8 = tuples.getcol %6 @t::@col1 : i64
	 %9 = tuples.getcol %6 @t2::@col1 : i64
	 %10 = db.compare eq %8 : i64,%9 : i64
	 tuples.return %10 : i1
 }
 // -----
//CHECK: module
 //CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
 //CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
 //CHECK: [[LEFTMAP:%.*]] = subop.map [[LEFT]]
 //CHECK: [[RIGHTMAP:%.*]] = subop.map [[RIGHT]]
 //CHECK: %{{.*}} = subop.union [[LEFTMAP]], [[RIGHTMAP]]
 %0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
 %1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
 %2 = relalg.union all %0, %1  mapping: {@setop::@col1({type = i64})=[@t::@col1,@t2::@col1]}
// -----
//CHECK: module
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK-DAG: [[MAP:%.*]] = subop.create !subop.map<[keyval$0 : i64], []>
//CHECK-DAG: [[LEFTMAP:%.*]] = subop.map [[LEFT]]
//CHECK-DAG: [[RIGHTMAP:%.*]] = subop.map [[RIGHT]]
//CHECK: %{{.*}} = subop.lookup_or_insert [[LEFTMAP]][[MAP]] [@setop::@col1]
//CHECK: %{{.*}} = subop.lookup_or_insert [[RIGHTMAP]][[MAP]] [@setop::@col1]
//CHECK: %{{.*}} = subop.scan [[MAP]]
%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.union distinct %0, %1  mapping: {@setop::@col1({type = i64})=[@t::@col1,@t2::@col1]}
// -----
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK: %{{.*}} = subop.create !subop.buffer<[member$0 : i64]>
//CHECK: subop.materialize [[RIGHT]] {@t2::@col1 => member$0}, %{{.*}} : !subop.buffer<[member$0 : i64]>
//CHECK: %{{.*}} = subop.nested_map [[LEFT]] [] (%arg0) {
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$0 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.scan %{{.*}} : !subop.buffer<[member$0 : i64]> {member$0 => @t2::@col1({type = i64})}
//CHECK:   %{{.*}} = subop.combine_tuple %{{.*}}, %arg0
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map::@pred({type = i1})] (%arg1: !tuples.tuple){
//CHECK:     %{{.*}} = tuples.getcol %arg1 @t::@col1 : i64
//CHECK:     %{{.*}} = tuples.getcol %arg1 @t2::@col1 : i64
//CHECK:     %{{.*}} = db.compare eq %{{.*}} : i64, %{{.*}} : i64
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@map::@pred]
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$1 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map1::@boolval({type = i1})] (){
//CHECK:     %{{.*}} = db.constant(true) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.lookup %{{.*}}%{{.*}} [] : !subop.simple_state<[marker$1 : i1]> @lookup::@ref({type = !subop.lookup_entry_ref<!subop.simple_state<[marker$1 : i1]>>})
//CHECK:   subop.scatter %{{.*}} @lookup::@ref {@map1::@boolval => marker$1}
//CHECK:   %{{.*}} = subop.scan %{{.*}} : !subop.simple_state<[marker$1 : i1]> {marker$1 => @marker::@marker({type = i1})}
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@marker::@marker]
//CHECK:   tuples.return %{{.*}} : !tuples.tuplestream
//CHECK: }
%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.semijoin %0, %1 (%6: !tuples.tuple) {
	 %8 = tuples.getcol %6 @t::@col1 : i64
	 %9 = tuples.getcol %6 @t2::@col1 : i64
	 %10 = db.compare eq %8 : i64,%9 : i64
	 tuples.return %10 : i1
 }
// -----
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK: %{{.*}} = subop.create !subop.buffer<[member$0 : i64]>
//CHECK: subop.materialize [[RIGHT]] {@t2::@col1 => member$0}, %{{.*}} : !subop.buffer<[member$0 : i64]>
//CHECK: %{{.*}} = subop.nested_map [[LEFT]] [] (%arg0) {
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$0 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.scan %{{.*}} : !subop.buffer<[member$0 : i64]> {member$0 => @t2::@col1({type = i64})}
//CHECK:   %{{.*}} = subop.combine_tuple %{{.*}}, %arg0
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map::@pred({type = i1})] (%arg1: !tuples.tuple){
//CHECK:     %{{.*}} = tuples.getcol %arg1 @t::@col1 : i64
//CHECK:     %{{.*}} = tuples.getcol %arg1 @t2::@col1 : i64
//CHECK:     %{{.*}} = db.compare eq %{{.*}} : i64, %{{.*}} : i64
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@map::@pred]
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$1 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map1::@boolval({type = i1})] (){
//CHECK:     %{{.*}} = db.constant(true) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.lookup %{{.*}}%{{.*}} [] : !subop.simple_state<[marker$1 : i1]> @lookup::@ref({type = !subop.lookup_entry_ref<!subop.simple_state<[marker$1 : i1]>>})
//CHECK:   subop.scatter %{{.*}} @lookup::@ref {@map1::@boolval => marker$1}
//CHECK:   %{{.*}} = subop.scan %{{.*}} : !subop.simple_state<[marker$1 : i1]> {marker$1 => @marker::@marker({type = i1})}
//CHECK:   %{{.*}} = subop.filter %{{.*}} none_true [@marker::@marker]
//CHECK:   tuples.return %{{.*}} : !tuples.tuplestream
//CHECK: }

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.antisemijoin %0, %1 (%6: !tuples.tuple) {
	 %8 = tuples.getcol %6 @t::@col1 : i64
	 %9 = tuples.getcol %6 @t2::@col1 : i64
	 %10 = db.compare eq %8 : i64,%9 : i64
	 tuples.return %10 : i1
 }
// -----
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK: %{{.*}} = subop.create !subop.multimap<[member$0 : i64], []>
//CHECK: subop.insert [[LEFT]]%{{.*}}  : !subop.multimap<[member$0 : i64], []> {@t::@col1 => member$0} eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.lookup [[RIGHT]]%{{.*}} [@t2::@col1] : !subop.multimap<[member$0 : i64], []> @lookup::@list({type = !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], []>>>})eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.nested_map %{{.*}} [@lookup::@list] (%arg0, %arg1) {
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$0 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.scan_list %arg1 : !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], []>>> @lookup1::@entryref({type = !subop.multi_map_entry_ref<<[member$0 : i64], []>>})
//CHECK:   %{{.*}} = subop.gather %{{.*}} @lookup1::@entryref {member$0 => @t::@col1({type = i64})}
//CHECK:   %{{.*}} = subop.combine_tuple %{{.*}}, %arg0
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map::@pred({type = i1})] (%arg2: !tuples.tuple){
//CHECK:     %{{.*}} = db.constant(1 : i64) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@map::@pred]
//CHECK:   tuples.return %{{.*}} : !tuples.tuplestream
//CHECK: }

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.join %0, %1 (%arg: !tuples.tuple) {
      %true = db.constant(1) : i1
      tuples.return %true : i1
 } attributes {useHashJoin, leftHash = [#tuples.columnref<@t::@col1>], rightHash = [#tuples.columnref<@t2::@col1>], nullsEqual = [0 : i8]}

// -----
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK: %{{.*}} = subop.create !subop.multimap<[member$0 : i64], []>
//CHECK: subop.insert [[RIGHT]]%{{.*}}  : !subop.multimap<[member$0 : i64], []> {@t2::@col1 => member$0} eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.lookup [[LEFT]]%{{.*}} [@t::@col1] : !subop.multimap<[member$0 : i64], []> @lookup::@list({type = !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], []>>>})eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.nested_map %{{.*}} [@lookup::@list] (%arg0, %arg1) {
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$0 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.scan_list %arg1 : !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], []>>> @lookup1::@entryref({type = !subop.multi_map_entry_ref<<[member$0 : i64], []>>})
//CHECK:   %{{.*}} = subop.gather %{{.*}} @lookup1::@entryref {member$0 => @t2::@col1({type = i64})}
//CHECK:   %{{.*}} = subop.combine_tuple %{{.*}}, %arg0
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map::@pred({type = i1})] (%arg2: !tuples.tuple){
//CHECK:     %{{.*}} = db.constant(1 : i64) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@map::@pred]
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$1 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map1::@boolval({type = i1})] (){
//CHECK:     %{{.*}} = db.constant(true) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.lookup %{{.*}}%{{.*}} [] : !subop.simple_state<[marker$1 : i1]> @lookup2::@ref({type = !subop.lookup_entry_ref<!subop.simple_state<[marker$1 : i1]>>})
//CHECK:   subop.scatter %{{.*}} @lookup2::@ref {@map1::@boolval => marker$1}
//CHECK:   %{{.*}} = subop.scan %{{.*}} : !subop.simple_state<[marker$1 : i1]> {marker$1 => @marker::@marker({type = i1})}
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@marker::@marker]
//CHECK:   tuples.return %{{.*}} : !tuples.tuplestream
//CHECK: }

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.semijoin %0, %1 (%arg: !tuples.tuple) {
      %true = db.constant(1) : i1
      tuples.return %true : i1
 } attributes {useHashJoin, leftHash = [#tuples.columnref<@t::@col1>], rightHash = [#tuples.columnref<@t2::@col1>], nullsEqual = [0 : i8]}

// -----
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK: %{{.*}} = subop.create !subop.multimap<[member$0 : i64], []>
//CHECK: subop.insert [[RIGHT]]%{{.*}}  : !subop.multimap<[member$0 : i64], []> {@t2::@col1 => member$0} eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.lookup [[LEFT]]%{{.*}} [@t::@col1] : !subop.multimap<[member$0 : i64], []> @lookup::@list({type = !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], []>>>})eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.nested_map %{{.*}} [@lookup::@list] (%arg0, %arg1) {
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$0 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.scan_list %arg1 : !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], []>>> @lookup1::@entryref({type = !subop.multi_map_entry_ref<<[member$0 : i64], []>>})
//CHECK:   %{{.*}} = subop.gather %{{.*}} @lookup1::@entryref {member$0 => @t2::@col1({type = i64})}
//CHECK:   %{{.*}} = subop.combine_tuple %{{.*}}, %arg0
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map::@pred({type = i1})] (%arg2: !tuples.tuple){
//CHECK:     %{{.*}} = db.constant(1 : i64) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@map::@pred]
//CHECK:   %{{.*}} = subop.create_simple_state <[marker$1 : i1]> initial : {
//CHECK:     %{{.*}} = db.constant(false) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map1::@boolval({type = i1})] (){
//CHECK:     %{{.*}} = db.constant(true) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.lookup %{{.*}}%{{.*}} [] : !subop.simple_state<[marker$1 : i1]> @lookup2::@ref({type = !subop.lookup_entry_ref<!subop.simple_state<[marker$1 : i1]>>})
//CHECK:   subop.scatter %{{.*}} @lookup2::@ref {@map1::@boolval => marker$1}
//CHECK:   %{{.*}} = subop.scan %{{.*}} : !subop.simple_state<[marker$1 : i1]> {marker$1 => @marker::@marker({type = i1})}
//CHECK:   %{{.*}} = subop.filter %{{.*}} none_true [@marker::@marker]
//CHECK:   tuples.return %{{.*}} : !tuples.tuplestream
//CHECK: }

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.antisemijoin %0, %1 (%arg: !tuples.tuple) {
      %true = db.constant(1) : i1
      tuples.return %true : i1
 } attributes {useHashJoin, leftHash = [#tuples.columnref<@t::@col1>], rightHash = [#tuples.columnref<@t2::@col1>], nullsEqual = [0 : i8]}

// -----
//CHECK: [[RIGHT:%.*]], %{{.*}} = subop.generate[@t::@col1({type = i64})]
//CHECK: [[LEFT:%.*]], %{{.*}} = subop.generate[@t2::@col1({type = i64})]
//CHECK: %{{.*}} = subop.create !subop.multimap<[member$0 : i64], [member$1 : i64, flag$0 : i1]>
//CHECK: %{{.*}} = subop.map [[LEFT]] computes : [@materialized::@marker({type = i1})] (){
//CHECK:   %{{.*}} = db.constant(false) : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: subop.insert %{{.*}}%{{.*}}  : !subop.multimap<[member$0 : i64], [member$1 : i64, flag$0 : i1]> {@materialized::@marker => flag$0, @t::@col1 => member$0, @t2::@col1 => member$1} eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.lookup [[RIGHT]]%{{.*}} [@t2::@col1] : !subop.multimap<[member$0 : i64], [member$1 : i64, flag$0 : i1]> @lookup::@list({type = !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], [member$1 : i64, flag$0 : i1]>>>})eq: ([%arg0],[%arg1]) {
//CHECK:   %{{.*}} = db.compare eq %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.and %{{.*}} : i1
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.nested_map %{{.*}} [@lookup::@list] (%arg0, %arg1) {
//CHECK:   %{{.*}} = subop.scan_list %arg1 : !subop.list<!subop.multi_map_entry_ref<<[member$0 : i64], [member$1 : i64, flag$0 : i1]>>> @lookup1::@entryref({type = !subop.multi_map_entry_ref<<[member$0 : i64], [member$1 : i64, flag$0 : i1]>>})
//CHECK:   %{{.*}} = subop.gather %{{.*}} @lookup1::@entryref {member$0 => @t::@col1({type = i64}), member$1 => @t2::@col1({type = i64})}
//CHECK:   %{{.*}} = subop.combine_tuple %{{.*}}, %arg0
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@map::@pred({type = i1})] (%arg2: !tuples.tuple){
//CHECK:     %{{.*}} = db.constant(1 : i64) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   %{{.*}} = subop.filter %{{.*}} all_true [@map::@pred]
//CHECK:   %{{.*}} = subop.map %{{.*}} computes : [@marker::@marker({type = i1})] (){
//CHECK:     %{{.*}} = db.constant(true) : i1
//CHECK:     tuples.return %{{.*}} : i1
//CHECK:   }
//CHECK:   subop.scatter %{{.*}} @lookup1::@entryref {@marker::@marker => flag$0}
//CHECK:   tuples.return
//CHECK: }
//CHECK: %{{.*}} = subop.scan %{{.*}} : !subop.multimap<[member$0 : i64], [member$1 : i64, flag$0 : i1]> {flag$0 => @materialized::@marker({type = i1}), member$0 => @t::@col1({type = i64}), member$1 => @t2::@col1({type = i64})}
//CHECK: %{{.*}} = subop.filter %{{.*}} all_true [@materialized::@marker]

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.const_relation columns : [@t2::@col1({type = i64})] values : [[0],[1]]
%2 = relalg.semijoin %1, %0 (%arg: !tuples.tuple) {
      %true = db.constant(1) : i1
      tuples.return %true : i1
 } attributes {useHashJoin,reverseSides, leftHash = [#tuples.columnref<@t::@col1>], rightHash = [#tuples.columnref<@t2::@col1>], nullsEqual = [0 : i8]}
// -----
//CHECK: module
//CHECK:%{{.*}} = subop.create_heap[] -> !subop.heap<5, []> ([],[]){
//CHECK:   %false = arith.constant false
//CHECK:   tuples.return %false : i1
//CHECK: }
//CHECK: subop.materialize %{{.*}} {}, %{{.*}} : !subop.heap<5, []>
//CHECK: %{{.*}} = subop.scan %{{.*}} : !subop.heap<5, []> {}
%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.limit 5 %0
// -----
//CHECK: %{{.*}} = subop.create_heap["member$0"] -> !subop.heap<5, [member$0 : i64]> ([%arg0],[%arg1]){
//CHECK:   %{{.*}} = db.constant(0 : i8) : i8
//CHECK:   %{{.*}} = db.sort_compare %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.compare lt %{{.*}} : i8, %{{.*}} : i8
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: subop.materialize %{{.*}} {@t::@col1 => member$0}, %{{.*}} : !subop.heap<5, [member$0 : i64]>
//CHECK: %{{.*}} = subop.scan %{{.*}} : !subop.heap<5, [member$0 : i64]> {member$0 => @t::@col1({type = i64})} {sequential}

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.topk 5 %0 [(@t::@col1,asc)]
// -----
//CHECK: %{{.*}} = subop.create !subop.buffer<[member$0 : i64]>
//CHECK: subop.materialize %{{.*}} {@t::@col1 => member$0}, %{{.*}} : !subop.buffer<[member$0 : i64]>
//CHECK: %{{.*}} = subop.create_sorted_view %{{.*}} : !subop.buffer<[member$0 : i64]> ["member$0"] ([%arg0],[%arg1]){
//CHECK:   %{{.*}} = db.sort_compare %arg0 : i64, %arg1 : i64
//CHECK:   %{{.*}} = db.compare lt %{{.*}} : i8, %{{.*}} : i8
//CHECK:   tuples.return %{{.*}} : i1
//CHECK: }
//CHECK: %{{.*}} = subop.scan %{{.*}} : !subop.sorted_view<!subop.buffer<[member$0 : i64]>> {member$0 => @t::@col1({type = i64})} {sequential}

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.sort %0 [(@t::@col1,asc)]

// -----
//CHECK: %{{.*}} = subop.create !subop.buffer<[]>
//CHECK: subop.materialize %{{.*}} {}, %{{.*}} : !subop.buffer<[]>
//CHECK: %{{.*}} = subop.scan %{{.*}} : !subop.buffer<[]> {}

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.tmp %0 [@t::@col1] -> !tuples.tuplestream

// -----
//CHECK: %{{.*}} = subop.get_external "{ \22table\22: \22test\22, \22mapping\22: { \22t$0\22 :\22t\22} }" : !subop.table<[t$0 : i64]>
//CHECK: %{{.*}} = subop.scan %{{.*}} : !subop.table<[t$0 : i64]> {}

%1 = relalg.basetable { table_identifier="test" } columns: {t => @t1::@col1({type=i64})}
// -----
//CHECK:  %{{.*}} = subop.map %{{.*}} computes : [@m::@col2({type = i64})] (%arg0: !tuples.tuple){
//CHECK:    %{{.*}} = tuples.getcol %arg0 @t::@col1 : i64
//CHECK:    %c42_i64 = arith.constant 42 : i64
//CHECK:    %{{.*}} = arith.addi %{{.*}}, %c42_i64 : i64
//CHECK:    tuples.return %{{.*}} : i64
//CHECK:  }

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.map %0 computes : [@m::@col2({type=i64})] (%tpl: !tuples.tuple){
   %curr = tuples.getcol %tpl @t::@col1 : i64
   %c42 = arith.constant 42 : i64
   %res = arith.addi %curr,%c42 : i64
   tuples.return %res : i64
}

// -----
//CHECK: %{{.*}} = subop.create !subop.result_table<[col1 : i64]>
//CHECK: subop.materialize %{{.*}} {@t::@col1 => col1}, %{{.*}} : !subop.result_table<[col1 : i64]>

%0 = relalg.const_relation columns : [@t::@col1({type = i64})] values : [[0],[1]]
%1 = relalg.materialize %0 [@t::@col1] => ["col1"] : !subop.local_table<[col1 : i64],["col1"]>
