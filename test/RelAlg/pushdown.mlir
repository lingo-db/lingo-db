// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-pushdown| FileCheck %s
module @querymodule  {
  func @query() {
    //CHECK: %0 = relalg.const_relation @constrel
    //CHECK: %1 = relalg.const_relation @constrel2
    //CHECK: %2 = relalg.selection %0
    //CHECK: %3 = relalg.selection %1
    //CHECK: %4 = relalg.crossproduct %2, %3
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%2 = relalg.crossproduct %0, %1
  	%3 = relalg.selection %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
        relalg.return %6 : !db.bool
  	}
	%4 = relalg.selection %3 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
		relalg.return %6 : !db.bool
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %0 = relalg.const_relation @constrel
    //CHECK: %1 = relalg.const_relation @constrel2
    //CHECK: %2 = relalg.selection %0
    //CHECK: %3 = relalg.selection %1
    //CHECK: %4 = relalg.crossproduct %2, %3
    //CHECK: %5 = relalg.selection %4
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%2 = relalg.crossproduct %0, %1
	%3 = relalg.selection %2 (%arg0: !relalg.tuple) {
		%10 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%11 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%12 = db.compare eq %10 : !db.int<32>, %11 : !db.int<32>
		relalg.return %12 : !db.bool
	}
  	%4 = relalg.selection %3 (%arg0: !relalg.tuple) {
		%10 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%11 = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
		%12 = db.compare eq %10 : !db.int<32>, %11 : !db.int<32>
		relalg.return %12 : !db.bool
  	}
	%5 = relalg.selection %4 (%arg0: !relalg.tuple) {
		%10 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%11 = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
		%12 = db.compare eq %10 : !db.int<32>, %11 : !db.int<32>
		relalg.return %12 : !db.bool
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %0 = relalg.const_relation @constrel
    //CHECK: %1 = relalg.const_relation @constrel2
    //CHECK: %2 = relalg.selection %0
    //CHECK: %3 = relalg.selection %1
    //CHECK: %4 = relalg.join %2, %3
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%2 = relalg.join %0, %1 (%arg0: !relalg.tuple) {
  		relalg.return
  	}
  	%3 = relalg.selection %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
        relalg.return %6 : !db.bool
  	}
	%4 = relalg.selection %3 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
		relalg.return %6 : !db.bool
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %0 = relalg.const_relation @constrel
    //CHECK: %1 = relalg.const_relation @constrel2
    //CHECK: %2 = relalg.selection %0
    //CHECK: %3 = relalg.outerjoin %2, %1
    //CHECK: %4 = relalg.selection %3
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%2 = relalg.outerjoin %0, %1 (%arg0: !relalg.tuple) {
  		relalg.return
  	}
  	%3 = relalg.selection %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
        relalg.return %6 : !db.bool
  	}
	%4 = relalg.selection %3 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
		relalg.return %6 : !db.bool
	}
    return
  }
}

// -----
module @querymodule  {
  func @query() {
    //CHECK: %0 = relalg.const_relation @constrel
    //CHECK: %1 = relalg.const_relation @constrel2
    //CHECK: %2 = relalg.fullouterjoin %0, %1
    //CHECK: %3 = relalg.selection %2
    //CHECK: %4 = relalg.selection %3
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>}),@attr2({type = !db.int<32>})] values: [[1, 1], [2, 2]]
  	%2 = relalg.fullouterjoin %0, %1 (%arg0: !relalg.tuple) {
  		relalg.return
  	}
  	%3 = relalg.selection %2 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
        relalg.return %6 : !db.bool
  	}
	%4 = relalg.selection %3 (%arg0: !relalg.tuple) {
		%4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%5 = relalg.getattr %arg0 @constrel2::@attr2 : !db.int<32>
		%6 = db.compare eq %4 : !db.int<32>, %5 : !db.int<32>
		relalg.return %6 : !db.bool
	}
    return
  }
}