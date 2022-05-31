// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-unnesting | FileCheck %s


// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
	%10 = relalg.const_relation columns: [ @constrel3 ::@attr1({type = i32})] values: [1, 2]
	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
		%3 = relalg.getcol %arg0 @constrel::@attr1 : i32
		%4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
		%5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
	}
	%110 = relalg.fullouterjoin %2, %10 (%arg0: !relalg.tuple) {
    	relalg.return
    }
  	//CHECK: %{{.*}} = relalg.projection distinct [@constrel::@attr1] %0
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.selection %{{.*}}
  	//CHECK: relalg.return
  	//CHECK: %{{.*}} = relalg.fullouterjoin %{{.*}}, %{{.*}}
	//CHECK: %{{.*}} = relalg.renaming {{.*}} renamed : [@renaming{{.*}}::@renamed0({type = i32})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @renaming{{.*}}::@renamed0 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32


  	%3 = relalg.join %0, %110 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
  	//CHECK: %{{.*}} = relalg.projection distinct [@constrel::@attr1] %0
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.selection %{{.*}}
  	//CHECK: relalg.return
  	//CHECK: %{{.*}} = relalg.aggregation %{{.*}}  [{{.*}},{{.*}}] computes : []
	//CHECK: %{{.*}} = relalg.renaming %{{.*}}  renamed : [@renaming{{.*}}::@renamed0({type = i32})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @renaming{{.*}}::@renamed0 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
	    %4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
	    %5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
  	}
  	%20 = relalg.aggregation %2 [@constrel2::@attr1] computes:[] (%arg0: !relalg.tuplestream) {
		relalg.return
	}
  	%3 = relalg.join %0, %20 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}

// -----

module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
  	//CHECK: %{{.*}} = relalg.projection distinct [@constrel::@attr1] %0
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.selection %{{.*}}
  	//CHECK: relalg.return
  	//CHECK: %{{.*}} = relalg.projection all [{{.*}},{{.*}}] %{{.*}}
	//CHECK: %{{.*}} = relalg.renaming %{{.*}}  renamed : [@renaming{{.*}}::@renamed0({type = i32})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @renaming{{.*}}::@renamed0 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
	    %4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
	    %5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
  	}
  	%20 = relalg.projection all [@constrel2::@attr1] %2
  	%3 = relalg.join %0, %20 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
	    %4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
	    %5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
	//CHECK: %{{.*}} = relalg.join %{{.*}}, %0 (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
	    %4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
	    %5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
  	}
  	%3 = relalg.join %2, %0 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
	%10 = relalg.const_relation columns: [ @constrel3 ::@attr1({type = i32})] values: [1, 2]
	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
		%3 = relalg.getcol %arg0 @constrel::@attr1 : i32
		%4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
		%5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
	}
	%110 = relalg.semijoin %10, %2 (%arg0: !relalg.tuple) {
    	relalg.return
    }
  	//CHECK: %{{.*}} = relalg.projection distinct [@constrel::@attr1] %0
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.selection %{{.*}}
  	//CHECK: relalg.return
  	//CHECK: %{{.*}} = relalg.semijoin %{{.*}}, %{{.*}}
	//CHECK: %{{.*}} = relalg.renaming %{{.*}}  renamed : [@renaming{{.*}}::@renamed0({type = i32})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @renaming{{.*}}::@renamed0 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32


  	%3 = relalg.join %0, %110 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
	%10 = relalg.const_relation columns: [ @constrel3 ::@attr1({type = i32})] values: [1, 2]
	%110 = relalg.crossproduct %1, %10
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %{{.*}}
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32

  	%2 = relalg.selection %110 (%arg0: !relalg.tuple) {
	    %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
	    %4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
	    %5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
	%10 = relalg.const_relation columns: [ @constrel3 ::@attr1({type = i32})] values: [1, 2]
	%110 = relalg.join %1, %10 (%arg0: !relalg.tuple) {
    	relalg.return
    }
    //CHECK: %{{.*}} = relalg.join %{{.*}}, %{{.*}}

	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32

  	%2 = relalg.selection %110 (%arg0: !relalg.tuple) {
	    %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
	    %4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
	    %5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func.func @query() {
  	%0 = relalg.const_relation columns: [ @constrel ::@attr1({type = i32})] values: [1, 2]
  	%1 = relalg.const_relation columns: [ @constrel2 ::@attr1({type = i32})] values: [1]
	%10 = relalg.const_relation columns: [ @constrel3 ::@attr1({type = i32})] values: [1, 2]
	%110 = relalg.semijoin %1, %10 (%arg0: !relalg.tuple) {
    	relalg.return
    }
    //CHECK: %{{.*}} = relalg.semijoin %{{.*}}, %{{.*}}

	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple)
	//CHECK: %{{.*}} = relalg.getcol %arg0 @constrel::@attr1 : i32
    //CHECK: %{{.*}} = relalg.getcol %arg0 @constrel2::@attr1 : i32
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : i32, %{{.*}} : i32

  	%2 = relalg.selection %110 (%arg0: !relalg.tuple) {
	    %3 = relalg.getcol %arg0 @constrel::@attr1 : i32
	    %4 = relalg.getcol %arg0 @constrel2::@attr1 : i32
	    %5 = db.compare eq %3 : i32, %4 : i32
		relalg.return %5 : i1
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}