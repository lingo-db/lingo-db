// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-implicit-to-explicit-joins | FileCheck %s
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	//CHECK: %{{.*}} =  relalg.semijoin left %{{.*}}, %{{.*}} (%arg0: !relalg.tuple) {
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  	      %4 = relalg.exists %1
          relalg.return %4 : !db.bool
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	//CHECK: %{{.*}} =  relalg.antisemijoin left %{{.*}}, %{{.*}} (%arg0: !relalg.tuple) {
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  	      %4 = relalg.exists %1
  	      %5 = db.not %4 : !db.bool
          relalg.return %5 : !db.bool
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	//CHECK: %{{.*}} =  relalg.markjoin left @markjoin @markattr({type = !db.bool}) %{{.*}}, %{{.*}} (%arg0: !relalg.tuple) {
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  //CHECK: %{{.*}} = relalg.getattr %arg0 @markjoin::@markattr
  	      %3 = relalg.exists %1
  	      %4 = db.constant( "true" ) : !db.bool
  	      %5 = db.and %3:!db.bool,%4:!db.bool
          relalg.return %5 : !db.bool
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
  	//CHECK: %{{.*}} =  relalg.singlejoin left %{{.*}}, %{{.*}} (%arg0: !relalg.tuple) {
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1
  	      %3 = relalg.getscalar @constrel2::@attr1 %1 : !db.int<32>
  	      %4 = db.constant( 1 ) : !db.int<32>
  	      %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
          relalg.return %5 : !db.bool
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	//CHECK: %{{.*}} = relalg.semijoin left %0, %1 (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
  	      %4 = relalg.in %3 : !db.int<32>, %1
          relalg.return %4 : !db.bool
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	//CHECK: %{{.*}} = relalg.antisemijoin left %0, %1 (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
  	      %4 = relalg.in %3 : !db.int<32>, %1
  	      %5 = db.not %4 : !db.bool
          relalg.return %5 : !db.bool
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	//CHECK: %2 = relalg.markjoin left @markjoin @markattr({type = !db.bool}) %0, %1 (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>
	//CHECK: relalg.return %6 : !db.bool
	//CHECK: %{{.*}} = relalg.selection %2 (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	//CHECK: %{{.*}} = relalg.getattr %arg0 @markjoin::@markattr : !db.bool
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		%3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
  	    %4 = relalg.in %3 : !db.int<32>, %1
        %5 = db.constant( "true" ) : !db.bool
		%6 = db.and %4:!db.bool,%5:!db.bool
		relalg.return %6 : !db.bool
  	}
    return
  }
}
