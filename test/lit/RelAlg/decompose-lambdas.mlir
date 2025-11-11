// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-decompose-lambdas | FileCheck %s
module @querymodule  {
  func.func @query() {
    //CHECK: %{{.*}} = relalg.const_relation columns : [@constrel::@attr1({type = i32}),@constrel::@attr2({type = i32})]
    //CHECK: %{{.*}} = relalg.const_relation columns : [@constrel2::@attr1({type = i32}),@constrel2::@attr2({type = i32})]
    //CHECK: %{{.*}} = relalg.crossproduct
	//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg0: !tuples.tuple)
	//CHECK: %{{.*}} = tuples.getcol %arg0 @constrel::@attr1 : i32
	//CHECK: %{{.*}} = tuples.getcol %arg0 @constrel2::@attr1 : i32
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
	//CHECK: tuples.return %{{.*}} : i1
	//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg0: !tuples.tuple)
	//CHECK: %{{.*}} = tuples.getcol %arg0 @constrel::@attr2 : i32
	//CHECK: %{{.*}} = tuples.getcol %arg0 @constrel2::@attr2 : i32
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
	//CHECK: tuples.return %{{.*}} : i1
  	%0 = relalg.const_relation  columns: [@constrel::@attr1({type = i32}),@constrel::@attr2({type = i32})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation  columns: [@constrel2::@attr1({type = i32}),@constrel2::@attr2({type = i32})] values: [[1, 1], [2, 2]]
  	%2 = relalg.crossproduct %0, %1
  	%3 = relalg.selection %2 (%arg0: !tuples.tuple) {
		%4 = tuples.getcol %arg0 @constrel::@attr1 : i32
		%5 = tuples.getcol %arg0 @constrel2::@attr1 : i32
		%6 = db.compare eq %4 : i32, %5 : i32
		%7 = tuples.getcol %arg0 @constrel::@attr2 : i32
		%8 = tuples.getcol %arg0 @constrel2::@attr2 : i32
		%9 = db.compare eq %7 : i32, %8 : i32
		%10 = db.and %6,%9:i1,i1
        tuples.return %10 : i1
  	}
    %res_table = relalg.materialize %3 [] => [] : !subop.local_table<[],[]>
    subop.set_result 0 %res_table : !subop.local_table<[],[]>
    return
  }
}