// RUN: db-run %s | FileCheck %s

 module {
	func @test (%arg0: !db.nullable<i1>) {
	 	db.dump %arg0 : !db.nullable<i1>
		%1 = db.isnull %arg0 : !db.nullable<i1>
		db.dump %1 : i1
		return
	}
	func @main () {
 		%const = db.constant ( 1 ) : i1
 		%null = db.null : !db.nullable<i1>
 		%not_null = db.cast %const  : i1 -> !db.nullable<i1>
 		//CHECK: bool(NULL)
 		//CHECK: bool(true)
 		call  @test(%null) : (!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		//CHECK: bool(false)
 		call  @test(%not_null) : (!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		%not_null_value =db.cast %not_null :  !db.nullable<i1> -> i1
 		db.dump %not_null_value : i1
		//CHECK: bool(NULL)
		%const_null = db.combine_null %const : i1,%const : !db.nullable<i1>
		db.dump %const_null : !db.nullable<i1>
		return
	}
 }