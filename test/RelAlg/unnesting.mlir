// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-unnesting | FileCheck %s


// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	%10 = relalg.const_relation @constrel3  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
		%3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
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
	//CHECK: %{{.*}} = relalg.renaming @renaming{{.*}} %{{.*}}  renamed: [@renamed0({type = !db.int<32>})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @renaming{{.*}}::@renamed0 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>


  	%3 = relalg.join %0, %110 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
  	//CHECK: %{{.*}} = relalg.projection distinct [@constrel::@attr1] %0
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.selection %{{.*}}
  	//CHECK: relalg.return
  	//CHECK: %{{.*}} = relalg.aggregation @aggr %{{.*}} [{{.*}},{{.*}}]
	//CHECK: %{{.*}} = relalg.renaming @renaming %{{.*}}  renamed: [@renamed0({type = !db.int<32>})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @renaming::@renamed0 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	    %4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	    %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
  	}
  	%20 = relalg.aggregation @aggr %2 [@constrel2::@attr1] (%arg0: !relalg.relation) {
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
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
  	//CHECK: %{{.*}} = relalg.projection distinct [@constrel::@attr1] %0
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %1
  	//CHECK: %{{.*}} = relalg.selection %{{.*}}
  	//CHECK: relalg.return
  	//CHECK: %{{.*}} = relalg.projection all [{{.*}},{{.*}}] %{{.*}}
	//CHECK: %{{.*}} = relalg.renaming @renaming %{{.*}}  renamed: [@renamed0({type = !db.int<32>})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @renaming::@renamed0 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	    %4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	    %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
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
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	    %4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	    %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	//CHECK: %{{.*}} = relalg.join %{{.*}}, %0 (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>
  	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
	    %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	    %4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	    %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
  	}
  	%3 = relalg.join %2, %0 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	%10 = relalg.const_relation @constrel3  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
	%2 = relalg.selection %1 (%arg0: !relalg.tuple) {
		%3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
		%4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
		%5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
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
	//CHECK: %{{.*}} = relalg.renaming @renaming{{.*}} %{{.*}}  renamed: [@renamed0({type = !db.int<32>})=[@constrel::@attr1]]
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @renaming{{.*}}::@renamed0 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>


  	%3 = relalg.join %0, %110 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	%10 = relalg.const_relation @constrel3  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
	%110 = relalg.crossproduct %1, %10
  	//CHECK: %{{.*}} = relalg.crossproduct %{{.*}}, %{{.*}}
	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>

  	%2 = relalg.selection %110 (%arg0: !relalg.tuple) {
	    %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	    %4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	    %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	%10 = relalg.const_relation @constrel3  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
	%110 = relalg.join %1, %10 (%arg0: !relalg.tuple) {
    	relalg.return
    }
    //CHECK: %{{.*}} = relalg.join %{{.*}}, %{{.*}}

	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>

  	%2 = relalg.selection %110 (%arg0: !relalg.tuple) {
	    %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	    %4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	    %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}
// -----
module @querymodule  {
  func @query() {
  	%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
  	%1 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1]
	%10 = relalg.const_relation @constrel3  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
	%110 = relalg.semijoin %1, %10 (%arg0: !relalg.tuple) {
    	relalg.return
    }
    //CHECK: %{{.*}} = relalg.semijoin %{{.*}}, %{{.*}}

	//CHECK: %{{.*}} = relalg.join %0, %{{.*}} (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
    //CHECK: %{{.*}} = db.compare eq %{{.*}} : !db.int<32>, %{{.*}} : !db.int<32>

  	%2 = relalg.selection %110 (%arg0: !relalg.tuple) {
	    %3 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
	    %4 = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
	    %5 = db.compare eq %3 : !db.int<32>, %4 : !db.int<32>
		relalg.return %5 : !db.bool
  	}
  	%3 = relalg.join %0, %2 (%arg0: !relalg.tuple) {
		relalg.return
	}
    return
  }
}