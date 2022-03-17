// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-implicit-to-explicit-joins | FileCheck %s
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1, 2]
  	%0 = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
  	%1 = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1, 2]
  	//CHECK: %{{.*}} =  relalg.semijoin %{{.*}}, %{{.*}} (%arg0: !relalg.tuple)
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  	      %4 = relalg.exists %1
          relalg.return %4 : i1
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1, 2]
  	%0 = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
  	%1 = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1, 2]
  	//CHECK: %{{.*}} =  relalg.antisemijoin %{{.*}}, %{{.*}} (%arg0: !relalg.tuple)
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  	      %4 = relalg.exists %1
  	      %5 = db.not %4 : i1
          relalg.return %5 : i1
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1, 2]
  	%0 = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
  	%1 = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1, 2]
  	//CHECK: %{{.*}} =  relalg.markjoin @markjoin @markattr({type = i1}) %{{.*}}, %{{.*}} (%arg0: !relalg.tuple)
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  //CHECK: %{{.*}} = relalg.getcol %arg0 @markjoin::@markattr
  	      %3 = relalg.exists %1
  	      %4 = db.constant( "true" ) : i1
  	      %5 = db.and %3,%4:i1,i1
          relalg.return %5 : i1
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1]
  	%0 = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
  	%1 = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1]
  	//CHECK: %{{.*}} =  relalg.singlejoin @singlejoin %{{.*}}, %{{.*}} (%arg0: !relalg.tuple)
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  //CHECK: %{{.*}} = relalg.getcol %arg0 @singlejoin::@sjattr
  	      %3 = relalg.getscalar @constrel2::@attr1 %1 : !db.nullable<i32>
  	      %4 = db.constant( 1 ) : i32
  	      %5 = db.compare eq %3 : !db.nullable<i32>, %4 : i32
          relalg.return %5 : !db.nullable<i1>
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
  	%1 = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1]
	//CHECK: %{{.*}} = relalg.semijoin %0, %1 (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
	//CHECK: %{{.*}} = db.compare eq %3 : i32, %4 : i32
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
  	      %4 = relalg.in %3 : i32, %1
          relalg.return %4 : i1
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
  	%1 = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1]
	//CHECK: %{{.*}} = relalg.antisemijoin %0, %1 (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
	//CHECK: %{{.*}} = db.compare eq %3 : i32, %4 : i32
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		  %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
  	      %4 = relalg.in %3 : i32, %1
  	      %5 = db.not %4 : i1
          relalg.return %5 : i1
  	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  columns : [@attr1({type = i32})] values : [1, 2]
  	%1 = relalg.const_relation @constrel2  columns : [@attr1({type = i32})] values : [1]
	//CHECK: %2 = relalg.markjoin @markjoin @markattr({type = i1}) %0, %1 (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
	//CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
	//CHECK: relalg.return %6 : i1
	//CHECK: %{{.*}} = relalg.selection %2 (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
	//CHECK: %{{.*}} = relalg.getcol %arg0 @markjoin::@markattr : i1
  	%2 = relalg.selection %0 (%arg0: !relalg.tuple) {
  		%3 = relalg.getcol %arg0 @constrel::@attr1 : i32
  	    %4 = relalg.in %3 : i32, %1
        %5 = db.constant( "true" ) : i1
		%6 = db.and %4,%5:i1,i1
		relalg.return %6 : i1
  	}
    return
  }
}
