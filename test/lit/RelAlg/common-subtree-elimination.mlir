// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope --relalg-cse --canonicalize --relalg-introduce-tmp --canonicalize | FileCheck %s
module @querymodule  {
  func.func @query() {
    //CHECK: %0 = relalg.basetable  {table_identifier = "table1"} columns: {col1 => @table1::@col1({type = !db.string}), col2 => @table1::@col2({type = !db.string})}
  	%0 = relalg.basetable {table_identifier = "table1"} columns:{ col1 => @table1::@col1({type = !db.string}),col2 => @table1::@col2({type = !db.string})}
    //CHECK: %1 = relalg.selection %0 (%arg0: !tuples.tuple){
  	%1 = relalg.selection %0 (%arg0: !tuples.tuple) {
        %4 = tuples.getcol %arg0 @table1::@col1 : i32
    	%5 = tuples.getcol %arg0 @table1::@col2 : i32
    	%6 = db.compare eq %4 : i32, %5 : i32
        tuples.return %6 : i1
    }

    //CHECK: %2 = relalg.basetable  {table_identifier = "table2"} columns: {col1 => @table2::@col1({type = !db.string}), col2 => @table2::@col2({type = !db.string})}
  	%2 = relalg.basetable {table_identifier = "table2"} columns:{ col1 => @table2::@col1({type = !db.string}),col2 => @table2::@col2({type = !db.string})}
	//CHECK: %3 = relalg.selection %2 (%arg0: !tuples.tuple){
	%3 = relalg.selection %2 (%arg0: !tuples.tuple) {
		%4 = tuples.getcol %arg0 @table2::@col1 : i32
		%5 = tuples.getcol %arg0 @table2::@col2 : i32
		%6 = db.compare eq %4 : i32, %5 : i32
		tuples.return %6 : i1
	}

	//CHECK: %4 = relalg.crossproduct %1, %3
  	%4 = relalg.crossproduct %1, %3
  	//CHECK: %5:2 = relalg.tmp %4 [@table1::@col1,@table2::@col1,@table1::@col2,@table2::@col2] -> !tuples.tuplestream, !tuples.tuplestream

  	%5 = relalg.basetable {table_identifier = "table1"} columns:{ col1 => @table1_u_1::@col1({type = !db.string}),col2 => @table1_u_1::@col2({type = !db.string})}
  	%6 = relalg.selection %5 (%arg0: !tuples.tuple) {
        %C4 = tuples.getcol %arg0 @table1_u_1::@col1 : i32
    	%C5 = tuples.getcol %arg0 @table1_u_1::@col2 : i32
    	%C6 = db.compare eq %C4 : i32, %C5 : i32
        tuples.return %C6 : i1
    }

  	%7 = relalg.basetable {table_identifier = "table2"} columns:{ col1 => @table2_u_1::@col1({type = !db.string}),col2 => @table2_u_1::@col2({type = !db.string})}
	%8 = relalg.selection %7 (%arg0: !tuples.tuple) {
		%C4 = tuples.getcol %arg0 @table2_u_1::@col1 : i32
		%C5 = tuples.getcol %arg0 @table2_u_1::@col2 : i32
		%C6 = db.compare eq %C4 : i32, %C5 : i32
		tuples.return %C6 : i1
	}
  	%9 = relalg.crossproduct %6, %8
    //CHECK: %6 = relalg.renaming %5#0 renamed : [@table1_u_1::@col1({type = !db.string})=[@table1::@col1],@table2_u_1::@col1({type = !db.string})=[@table2::@col1],@table1_u_1::@col2({type = !db.string})=[@table1::@col2],@table2_u_1::@col2({type = !db.string})=[@table2::@col2]]

    //CHECK: %7 = relalg.crossproduct %5#1, %6
  	%10 = relalg.crossproduct %4, %9
  	//CHECK: %8 = relalg.materialize %7 [] => [] : !subop.local_table<[], []>
    %res_table = relalg.materialize %10 [] => [] : !subop.local_table<[],[]>
    subop.set_result 0 %res_table : !subop.local_table<[],[]>
    return
  }
}