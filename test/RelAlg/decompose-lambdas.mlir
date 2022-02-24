// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-decompose-lambdas | FileCheck %s
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel
    //CHECK: %{{.*}} = relalg.const_relation @constrel2
    //CHECK: %{{.*}} = relalg.crossproduct
	//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>
	//CHECK: relalg.return %{{.*}} : i1
	//CHECK: %{{.*}} = relalg.selection %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>
	//CHECK: relalg.return %{{.*}} : i1
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%2 = relalg.crossproduct %0, %1
  	%3 = relalg.selection %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
		%7 = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
		%8 = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
		%9 = db.compare eq %7 : !db.int<32>, %8 : !db.int<32>
		%10 = db.and %6:i1,%9:i1
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
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%2 = relalg.crossproduct %0, %1
    //CHECK: %{{.*}} = relalg.map @map %{{.*}} (%arg0: !relalg.tuple)
    //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
    //CHECK: %{{.*}} = db.add %{{.*}}:!db.int<32>,%{{.*}}:!db.int<32>
    //CHECK: relalg.addattr %{{.*}}, @attr4({type = !db.int<32>}) %{{.*}}
    //CHECK: relalg.return %{{.*}} : !relalg.tuple
	//CHECK: %{{.*}} = relalg.map @map %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = db.add %5:!db.int<32>,%6:!db.int<32>
	//CHECK: %{{.*}} = relalg.addattr %{{.*}}, @attr3({type = !db.int<32>}) %{{.*}}
	//CHECK: relalg.return %{{.*}} : !relalg.tuple
  	%3 = relalg.map @map %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%6 = db.add %4 : !db.int<32>, %5 : !db.int<32>
		%tpl = relalg.addattr %arg0, @attr3({type = !db.int<32>}) %6
		%7 = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
		%8 = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
		%9 = db.add %7 : !db.int<32>, %8 : !db.int<32>
		%tpl2 =relalg.addattr %tpl, @attr4({type = !db.int<32>}) %9
        relalg.return %tpl2 : !relalg.tuple
  	}
    return
  }
}