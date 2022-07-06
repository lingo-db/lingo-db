// RUN: mlir-db-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s
module{
// CHECK: %0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})]
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
}
// -----

module{
// CHECK: %0 = relalg.basetable {table_identifier = "table1"} columns: {col1 => @table1::@col1({type = !db.string}), col2 => @table1::@col2({type = !db.string})}
%0 = relalg.basetable {table_identifier = "table1"} columns:{ col1 => @table1::@col1({type = !db.string}),col2 => @table1::@col2({type = !db.string})}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = i1})] values : ["true", "false"]
//CHECK: %1 = relalg.selection %0 (%arg0: !tuples.tuple)
%1 = relalg.selection %0 (%arg0: !tuples.tuple) {
    //CHECK:    %2 = tuples.getcol %arg0 @constrel::@attr1 : i1
	%2 = tuples.getcol %arg0 @constrel::@attr1 : i1
	//CHECK:    tuples.return %2 : i1
	tuples.return %2 : i1
}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string})] values : ["A", "B"]
//CHECK: %1 = relalg.map %0 computes : [@map::@attr2({type = i1})] (%arg0: !tuples.tuple)
%1 = relalg.map %0 computes : [@map::@attr2({type = i1})] (%arg0: !tuples.tuple) {
    //CHECK:    %2 = db.constant("true") : i1
	%2 = db.constant("true") : i1
	//CHECK: tuples.return %2 : i1
	tuples.return %2 : i1
}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string})] values : ["A", "B"]
//CHECK: %1 = relalg.aggregation %0 [@constrel::@attr1] computes : [] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple)
%1 = relalg.aggregation %0 [@constrel::@attr1] computes : [] (%arg0: !tuples.tuplestream, %arg1: !tuples.tuple) {
	tuples.return
}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string})] values : ["A", "B"]
%1 = relalg.materialize %0 [@constrel::@attr1] => ["col1"] : !dsa.table
}

//renaming:
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string})] values : ["A", "B"]
//CHECK: %1 = relalg.renaming %0 renamed : [@renamed::@attr1({type = !db.string})=[@constrel::@attr1]]
%1 = relalg.renaming %0 renamed : [@renamed::@attr1({type = !db.string})=[@constrel::@attr1]]
}




//joins
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.crossproduct %0, %1
%2 = relalg.crossproduct %0, %1
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.join %0, %1  (%arg0: !tuples.tuple)
%2 = relalg.join %0, %1  (%arg0: !tuples.tuple) {
    //CHECK:    %3 = db.constant("true") : i1
	%3 = db.constant("true") : i1
	//CHECK:    tuples.return %3 : i1
	tuples.return %3 : i1
}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.semijoin %0, %1  (%arg0: !tuples.tuple)
%2 = relalg.semijoin %0, %1  (%arg0: !tuples.tuple) {
    //CHECK:    %3 = db.constant("true") : i1
	%3 = db.constant("true") : i1
	//CHECK:    tuples.return %3 : i1
	tuples.return %3 : i1
}
}

// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.antisemijoin %0, %1  (%arg0: !tuples.tuple)
%2 = relalg.antisemijoin %0, %1  (%arg0: !tuples.tuple) {
    //CHECK:    %3 = db.constant("true") : i1
	%3 = db.constant("true") : i1
	//CHECK:    tuples.return %3 : i1
	tuples.return %3 : i1
}
}

// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.outerjoin %0, %1  (%arg0: !tuples.tuple)
%2 = relalg.outerjoin %0, %1  (%arg0: !tuples.tuple) {
    //CHECK:    %3 = db.constant("true") : i1
	%3 = db.constant("true") : i1
	//CHECK:    tuples.return %3 : i1
	tuples.return %3 : i1
} mapping: {@outerjoin::@attr1({type = !db.string})=[@constrel2::@attr1]}
}

// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.fullouterjoin %0, %1  (%arg0: !tuples.tuple)
%2 = relalg.fullouterjoin %0, %1  (%arg0: !tuples.tuple) {
    //CHECK:    %3 = db.constant("true") : i1
	%3 = db.constant("true") : i1
	//CHECK:    tuples.return %3 : i1
	tuples.return %3 : i1
}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.markjoin @markjoin::@markattr({type = !db.string}) %0, %1  (%arg0: !tuples.tuple)
%2 = relalg.markjoin @markjoin::@markattr({type = !db.string}) %0, %1  (%arg0: !tuples.tuple) {
    //CHECK:    %3 = db.constant("true") : i1
	%3 = db.constant("true") : i1
	//CHECK:    tuples.return %3 : i1
	tuples.return %3 : i1
}
}


//sorting+limit
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.sort %0 [(@constrel::@attr1,asc),(@constrel::@attr2,desc)]
%1 = relalg.sort %0 [(@constrel::@attr1,asc),(@constrel::@attr2,desc)]
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.limit 100 %0
%1 = relalg.limit 100 %0
}

//aggregation functions
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = i32}),@constrel::@attr2({type = i32})] values : [[1,1], [0,0]]
//CHECK: %1 = relalg.aggrfn min @constrel::@attr1 %0 : !db.nullable<i32>
%1 = relalg.aggrfn min @constrel::@attr1 %0 : !db.nullable<i32>
//CHECK: %2 = relalg.aggrfn max @constrel::@attr1 %0 : !db.nullable<i32>
%2 = relalg.aggrfn max @constrel::@attr1 %0 : !db.nullable<i32>
//CHECK: %3 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.nullable<i32>
%3 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.nullable<i32>
//CHECK: %4 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.nullable<i32>
%4 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.nullable<i32>
//CHECK: %5 = relalg.aggrfn avg @constrel::@attr1 %0 : !db.nullable<i32>
%5 = relalg.aggrfn avg @constrel::@attr1 %0 : !db.nullable<i32>
//CHECK: %6 = relalg.aggrfn count @constrel::@attr1 %0 : i64
%6 = relalg.aggrfn count @constrel::@attr1 %0 : i64
}

//scalar operations
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string})] values : ["A", "B"]
//CHECK: %1 = relalg.getscalar @constrel::@attr1 %0 : !db.string
%1 = relalg.getscalar @constrel::@attr1 %0 : !db.string
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string})] values : ["A", "B"]
%1 = db.constant("A") : !db.string
//CHECK: %2 = relalg.in %1 : !db.string, %0
%2 = relalg.in %1 : !db.string, %0
}

//set operations
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.projection all [@constrel::@attr1] %0
%1 = relalg.projection all [@constrel::@attr1] %0
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.projection distinct [@constrel::@attr1] %0
%1 = relalg.projection distinct [@constrel::@attr1] %0
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.union distinct  %0, %1  mapping: {@union1::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @union1::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.union distinct  %0, %1  mapping: {@union1::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @union1::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.union all  %0, %1  mapping: {@union1::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @union1::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.union all  %0, %1  mapping: {@union1::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @union1::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.intersect distinct  %0, %1  mapping: {@intersect::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @intersect::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.intersect distinct  %0, %1  mapping: {@intersect::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @intersect::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.intersect all  %0, %1  mapping: {@intersect::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @intersect::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.intersect all  %0, %1  mapping: {@intersect::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @intersect::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.except distinct  %0, %1  mapping: {@except::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @except::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.except distinct  %0, %1  mapping: {@except::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @except::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation columns : [@constrel::@attr1({type = !db.string}),@constrel::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation columns : [@constrel2::@attr1({type = !db.string}),@constrel2::@attr2({type = !db.string})] values : [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.except all  %0, %1  mapping: {@except::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @except::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.except all  %0, %1  mapping: {@except::@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @except::@attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}