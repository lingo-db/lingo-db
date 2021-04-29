// RUN: db-run %s | FileCheck %s

 module {
	func @main () {
 		%const_int32_1 = db.constant ( 1 ) : !db.int<32>
 		%const_float32_1 = db.constant ( 1.101 ) : !db.float<32>
 		%const_decimal_10_3 = db.constant ( "1.101" ) : !db.decimal<10,3>
 		%const_decimal_10_5 = db.constant ( "1.10101" ) : !db.decimal<10,5>

 		//CHECK: float(1)
		%0 = db.cast %const_int32_1 : !db.int<32> -> !db.float<32>
		db.dump %0 : !db.float<32>
		//CHECK: int(1)
		%1 = db.cast %const_float32_1 : !db.float<32> -> !db.int<32>
		db.dump %1 : !db.int<32>
		//CHECK: decimal(1.10100)
		%2 = db.cast %const_decimal_10_3 : !db.decimal<10,3> -> !db.decimal<10,5>
		db.dump %2 : !db.decimal<10,5>
		//CHECK: decimal(1.101)
		%3 = db.cast %const_decimal_10_5 : !db.decimal<10,5> -> !db.decimal<10,3>
		db.dump %3 : !db.decimal<10,3>
		//CHECK: decimal(1.10099)
		%4 = db.cast %const_float32_1 : !db.float<32> -> !db.decimal<10,5>
		db.dump %4 : !db.decimal<10,5>
		//CHECK: float(1.10101)
		%5 = db.cast %const_decimal_10_5: !db.decimal<10,5> -> !db.float<32>
		db.dump %5 : !db.float<32>
		//CHECK: int(1)
		%6 = db.cast %const_decimal_10_5: !db.decimal<10,5> -> !db.int<32>
		db.dump %6 : !db.int<32>
		//CHECK: decimal(1.00000)
		%7 = db.cast %const_int32_1: !db.int<32> -> !db.decimal<10,5>
		db.dump %7 : !db.decimal<10,5>

		return
	}
 }