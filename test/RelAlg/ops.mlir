// RUN: mlir-db-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s
module{
// CHECK: %0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})]
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A","B"],["C","D"]]
}
// -----

module{
// CHECK: %0 = relalg.basetable @table1 {table_identifier = "table1"} columns: {col1 => @col1({type = !db.string}), col2 => @col2({type = !db.string})}
%0 = relalg.basetable @table1 {table_identifier = "table1"} columns:{ col1 => @col1({type = !db.string}),col2 => @col2({type = !db.string})}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.bool})] values: ["true", "false"]
//CHECK: %1 = relalg.selection %0 (%arg0: !relalg.tuple) {
%1 = relalg.selection %0 (%arg0: !relalg.tuple) {
    //CHECK:    %2 = relalg.getattr %arg0 @constrel::@attr1 : !db.bool
	%2 = relalg.getattr %arg0 @constrel::@attr1 : !db.bool
	//CHECK:    relalg.return %2 : !db.bool
	relalg.return %2 : !db.bool
}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["A", "B"]
//CHECK: %1 = relalg.map @map %0 (%arg0: !relalg.tuple) {
%1 = relalg.map @map %0 (%arg0: !relalg.tuple) {
    //CHECK:    %2 = db.constant( "true" ) : !db.bool
	%2 = db.constant( "true" ) : !db.bool
	//CHECK:    %3 = relalg.addattr %arg0, @attr2({type = !db.bool}) %2
	%tpl=relalg.addattr %arg0, @attr2({type = !db.bool}) %2
	relalg.return %tpl : !relalg.tuple
}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["A", "B"]
//CHECK: %1 = relalg.aggregation @aggr %0 [@constrel::@attr1] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple) {
%1 = relalg.aggregation @aggr %0 [@constrel::@attr1] (%arg0: !relalg.tuplestream, %arg1: !relalg.tuple) {
	relalg.return
}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["A", "B"]
%1 = relalg.materialize %0 [@constrel::@attr1] => ["col1"] : !db.table
}

//renaming:
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["A", "B"]
//CHECK: %1 = relalg.renaming @renamed %0 renamed: [@attr1({type = !db.string})=[@constrel::@attr1]]
%1 = relalg.renaming @renamed %0 renamed: [@attr1({type = !db.string})=[@constrel::@attr1]]
}




//joins
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.crossproduct %0, %1
%2 = relalg.crossproduct %0, %1
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.join %0, %1  (%arg0: !relalg.tuple) {
%2 = relalg.join %0, %1  (%arg0: !relalg.tuple) {
    //CHECK:    %3 = db.constant( "true" ) : !db.bool
	%3 = db.constant( "true" ) : !db.bool
	//CHECK:    relalg.return %3 : !db.bool
	relalg.return %3 : !db.bool
}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.semijoin %0, %1  (%arg0: !relalg.tuple) {
%2 = relalg.semijoin %0, %1  (%arg0: !relalg.tuple) {
    //CHECK:    %3 = db.constant( "true" ) : !db.bool
	%3 = db.constant( "true" ) : !db.bool
	//CHECK:    relalg.return %3 : !db.bool
	relalg.return %3 : !db.bool
}
}

// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.antisemijoin %0, %1  (%arg0: !relalg.tuple) {
%2 = relalg.antisemijoin %0, %1  (%arg0: !relalg.tuple) {
    //CHECK:    %3 = db.constant( "true" ) : !db.bool
	%3 = db.constant( "true" ) : !db.bool
	//CHECK:    relalg.return %3 : !db.bool
	relalg.return %3 : !db.bool
}
}

// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.outerjoin @outerjoin %0, %1  (%arg0: !relalg.tuple) {
%2 = relalg.outerjoin @outerjoin %0, %1  (%arg0: !relalg.tuple) {
    //CHECK:    %3 = db.constant( "true" ) : !db.bool
	%3 = db.constant( "true" ) : !db.bool
	//CHECK:    relalg.return %3 : !db.bool
	relalg.return %3 : !db.bool
} mapping: {@attr1({type = !db.string})=[@constrel2::@attr1]}
}

// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.fullouterjoin %0, %1  (%arg0: !relalg.tuple) {
%2 = relalg.fullouterjoin %0, %1  (%arg0: !relalg.tuple) {
    //CHECK:    %3 = db.constant( "true" ) : !db.bool
	%3 = db.constant( "true" ) : !db.bool
	//CHECK:    relalg.return %3 : !db.bool
	relalg.return %3 : !db.bool
}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.markjoin @markjoin @markattr({type = !db.string}) %0, %1  (%arg0: !relalg.tuple) {
%2 = relalg.markjoin @markjoin @markattr({type = !db.string}) %0, %1  (%arg0: !relalg.tuple) {
    //CHECK:    %3 = db.constant( "true" ) : !db.bool
	%3 = db.constant( "true" ) : !db.bool
	//CHECK:    relalg.return %3 : !db.bool
	relalg.return %3 : !db.bool
}
}


//sorting+limit
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.sort %0 [(@constrel::@attr1,asc),(@constrel::@attr2,desc)]
%1 = relalg.sort %0 [(@constrel::@attr1,asc),(@constrel::@attr2,desc)]
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.limit 100 %0
%1 = relalg.limit 100 %0
}

//aggregation functions
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.aggrfn min @constrel::@attr1 %0 : !db.int<32>
%1 = relalg.aggrfn min @constrel::@attr1 %0 : !db.int<32>
//CHECK: %2 = relalg.aggrfn max @constrel::@attr1 %0 : !db.int<32>
%2 = relalg.aggrfn max @constrel::@attr1 %0 : !db.int<32>
//CHECK: %3 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.int<32>
%3 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.int<32>
//CHECK: %4 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.int<32>
%4 = relalg.aggrfn sum @constrel::@attr1 %0 : !db.int<32>
//CHECK: %5 = relalg.aggrfn avg @constrel::@attr1 %0 : !db.int<32>
%5 = relalg.aggrfn avg @constrel::@attr1 %0 : !db.int<32>
//CHECK: %6 = relalg.aggrfn count @constrel::@attr1 %0 : !db.int<32>
%6 = relalg.aggrfn count @constrel::@attr1 %0 : !db.int<32>
}

//scalar operations
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["A"]
//CHECK: %1 = relalg.getscalar @constrel::@attr1 %0 : !db.string
%1 = relalg.getscalar @constrel::@attr1 %0 : !db.string
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["A", "B"]
%1 = db.constant( "A" ) : !db.string
//CHECK: %2 = relalg.in %1 : !db.string, %0
%2 = relalg.in %1 : !db.string, %0
}

//set operations
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.projection all [@constrel::@attr1] %0
%1 = relalg.projection all [@constrel::@attr1] %0
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %1 = relalg.projection distinct [@constrel::@attr1] %0
%1 = relalg.projection distinct [@constrel::@attr1] %0
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.union @union1 distinct  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.union @union1 distinct  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.union @union1 all  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.union @union1 all  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.intersect @intersect distinct  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.intersect @intersect distinct  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.intersect @intersect all  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.intersect @intersect all  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.except @except distinct  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.except @except distinct  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}
// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.string}),@attr2({type = !db.string})] values: [["A1","B1"], ["A2","B2"]]
//CHECK: %2 = relalg.except @except all  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
%2 = relalg.except @except all  %0, %1  mapping: {@attr1({type = !db.string})=[@constrel::@attr1,@constrel2::@attr1], @attr2({type = !db.string})=[@constrel::@attr2,@constrel2::@attr2]}
}