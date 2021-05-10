 // RUN: db-run %s | FileCheck %s
 module {
 	func private @bitcode_test (!db.int<32>,!db.int<32>) ->!db.int<32>

	func @main () {
	 		%int32_const = db.constant ( 10 ) : !db.int<32>
	 		%1 = call @bitcode_test(%int32_const, %int32_const) : (!db.int<32>,!db.int<32>) -> !db.int<32>
	 		//CHECK: int(20)
			db.dump %1 : !db.int<32>

		return
	}
 }