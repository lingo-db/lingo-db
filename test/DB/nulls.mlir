// RUN: db-run %s | FileCheck %s

 module {
	func @test (%arg0: !db.bool<nullable>) {
	 	db.dump %arg0 : !db.bool<nullable>
		%1 = db.isnull %arg0 : !db.bool<nullable>
		db.dump %1 : !db.bool
		return
	}
	func @main () {
 		%const = db.constant ( 1 ) : !db.bool
 		%null = db.null : !db.bool<nullable>
 		%not_null = db.cast %const  : !db.bool -> !db.bool<nullable>
 		//CHECK: bool(NULL)
 		//CHECK: bool(true)
 		call  @test(%null) : (!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		//CHECK: bool(false)
 		call  @test(%not_null) : (!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		%not_null_value =db.cast %not_null :  !db.bool<nullable> -> !db.bool
 		db.dump %not_null_value : !db.bool


		return
	}
 }