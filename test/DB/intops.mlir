// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
 	 	%constm10 = db.constant ( -10 ) : !db.int<32>
 	  	%constm1 = db.constant ( -1 ) : !db.int<32>
  	 	%const0 = db.constant ( 0 ) : !db.int<32>
    	%const1 = db.constant ( 1 ) : !db.int<32>
 		%const10 = db.constant ( 10 ) : !db.int<32>
 		%const4 = db.constant ( 4 ) : !db.int<32>

 		//CHECK: int(-10)
 		db.dump %constm10 : !db.int<32>
 		//CHECK: int(-1)
 		db.dump %constm1 : !db.int<32>
  		//CHECK: int(0)
  		db.dump %const0 : !db.int<32>
  		//CHECK: int(1)
  		db.dump %const1 : !db.int<32>
  		//CHECK: int(10)
  		db.dump %const10 : !db.int<32>

  		%0 = db.add %constm10 : !db.int<32>,%constm1 : !db.int<32>
 		//CHECK: int(-11)
 		db.dump %0 : !db.int<32>
		%1 = db.add %const10 : !db.int<32>,%const1 : !db.int<32>
		//CHECK: int(11)
		db.dump %1 : !db.int<32>

  		%2 = db.sub %constm10 : !db.int<32>,%constm1 : !db.int<32>
 		//CHECK: int(-9)
 		db.dump %2 : !db.int<32>
		%3 = db.sub %const10 : !db.int<32>,%const1 : !db.int<32>
		//CHECK: int(9)
		db.dump %3 : !db.int<32>

  		%4 = db.mul %constm10 : !db.int<32>,%constm10 : !db.int<32>
 		//CHECK: int(100)
 		db.dump %4 : !db.int<32>
		%5 = db.mul %const10 : !db.int<32>,%const10 : !db.int<32>
		//CHECK: int(100)
		db.dump %5 : !db.int<32>

  		%6 = db.div %constm10 : !db.int<32>,%constm10 : !db.int<32>
 		//CHECK: int(1)
 		db.dump %6 : !db.int<32>
		%7 = db.div %const10 : !db.int<32>,%const10 : !db.int<32>
		//CHECK: int(1)
		db.dump %7 : !db.int<32>

  		%8 = db.mod %constm10 : !db.int<32>,%const4 : !db.int<32>
 		//CHECK: int(-2)
 		db.dump %8 : !db.int<32>
		%9 = db.mod %const10 : !db.int<32>,%const4 : !db.int<32>
		//CHECK: int(2)
		db.dump %9 : !db.int<32>

 		return
  }
 }
