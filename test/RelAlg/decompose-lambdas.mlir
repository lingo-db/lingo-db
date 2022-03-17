// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-decompose-lambdas | FileCheck %s
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel
    //CHECK: %{{.*}} = relalg.const_relation @constrel2
    //CHECK: %{{.*}} = relalg.crossproduct
	//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
	//CHECK: relalg.return %{{.*}} : i1
	//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr2 : i32
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr2 : i32
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
	//CHECK: relalg.return %{{.*}} : i1
  	%0 = relalg.const_relation @constrel  columns: [@attr1({type = i32}),@attr2({type = i32})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  columns: [@attr1({type = i32}),@attr2({type = i32})] values: [[1, 1], [2, 2]]
  	%2 = relalg.crossproduct %0, %1
  	%3 = relalg.selection %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getcol %arg0 @constrel::@attr1 : i32
		%5 = relalg.getcol %arg0 @constrel2::@attr1 : i32
		%6 = db.compare eq %4 : i32, %5 : i32
		%7 = relalg.getcol %arg0 @constrel::@attr2 : i32
		%8 = relalg.getcol %arg0 @constrel2::@attr2 : i32
		%9 = db.compare eq %7 : i32, %8 : i32
		%10 = db.and %6,%9:i1,i1
        relalg.return %10 : i1
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel
    //CHECK: %{{.*}} = relalg.const_relation @constrel2
    //CHECK: %{{.*}} = relalg.crossproduct
  	%0 = relalg.const_relation @constrel  columns: [@attr1({type = i32}),@attr2({type = i32})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  columns: [@attr1({type = i32}),@attr2({type = i32})] values: [[1, 1], [2, 2]]
  	%2 = relalg.crossproduct %0, %1
    //CHECK: %{{.*}} = relalg.map @map %{{.*}} (%arg0: !relalg.tuple)
    //CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr2 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr2 : i32
    //CHECK: %{{.*}} = db.add %{{.*}} : i32, %{{.*}} : i32
    //CHECK: relalg.addcol %{{.*}}, @attr4({type = i32}) %{{.*}}
    //CHECK: relalg.return %{{.*}} : !relalg.tuple
	//CHECK: %{{.*}} = relalg.map @map %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
	//CHECK: %{{.*}} = db.add %5 : i32, %6 : i32
	//CHECK: %{{.*}} = relalg.addcol %{{.*}}, @attr3({type = i32}) %{{.*}}
	//CHECK: relalg.return %{{.*}} : !relalg.tuple
  	%3 = relalg.map @map %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getcol %arg0 @constrel::@attr1 : i32
		%5 = relalg.getcol %arg0 @constrel2::@attr1 : i32
		%6 = db.add %4 : i32, %5 : i32
		%tpl = relalg.addcol %arg0, @attr3({type = i32}) %6
		%7 = relalg.getcol %arg0 @constrel::@attr2 : i32
		%8 = relalg.getcol %arg0 @constrel2::@attr2 : i32
		%9 = db.add %7 : i32, %8 : i32
		%tpl2 =relalg.addcol %tpl, @attr4({type = i32}) %9
        relalg.return %tpl2 : !relalg.tuple
  	}
    return
  }
}