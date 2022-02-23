// RUN: db-run %s | FileCheck %s

 module {
	func @test (%arg0: !db.nullable<!db.bool>) {
	 	db.dump %arg0 : !db.nullable<!db.bool>
		%1 = db.isnull %arg0 : !db.nullable<!db.bool>
		db.dump %1 : !db.bool
		return
	}
	func @main () {
 		%const = db.constant ( 1 ) : !db.bool
 		%null = db.null : !db.nullable<!db.bool>
 		%not_null = db.cast %const  : !db.bool -> !db.nullable<!db.bool>
 		//CHECK: bool(NULL)
 		//CHECK: bool(true)
 		call  @test(%null) : (!db.nullable<!db.bool>) -> ()
 		//CHECK: bool(true)
 		//CHECK: bool(false)
 		call  @test(%not_null) : (!db.nullable<!db.bool>) -> ()
 		//CHECK: bool(true)
 		%not_null_value =db.cast %not_null :  !db.nullable<!db.bool> -> !db.bool
 		db.dump %not_null_value : !db.bool
		//CHECK: bool(NULL)
		%const_null = db.combine_null %const : !db.bool,%const : !db.nullable<!db.bool>
		db.dump %const_null : !db.nullable<!db.bool>
		return
	}
 }