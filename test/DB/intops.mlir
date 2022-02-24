// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
 	 	%constm10 = db.constant ( -10 ) : i32
 	  	%constm1 = db.constant ( -1 ) : i32
  	 	%const0 = db.constant ( 0 ) : i32
    	%const1 = db.constant ( 1 ) : i32
 		%const10 = db.constant ( 10 ) : i32
 		%const4 = db.constant ( 4 ) : i32

 		//CHECK: int(-10)
 		db.dump %constm10 : i32
 		//CHECK: int(-1)
 		db.dump %constm1 : i32
  		//CHECK: int(0)
  		db.dump %const0 : i32
  		//CHECK: int(1)
  		db.dump %const1 : i32
  		//CHECK: int(10)
  		db.dump %const10 : i32

  		%0 = db.add %constm10 : i32,%constm1 : i32
 		//CHECK: int(-11)
 		db.dump %0 : i32
		%1 = db.add %const10 : i32,%const1 : i32
		//CHECK: int(11)
		db.dump %1 : i32

  		%2 = db.sub %constm10 : i32,%constm1 : i32
 		//CHECK: int(-9)
 		db.dump %2 : i32
		%3 = db.sub %const10 : i32,%const1 : i32
		//CHECK: int(9)
		db.dump %3 : i32

  		%4 = db.mul %constm10 : i32,%constm10 : i32
 		//CHECK: int(100)
 		db.dump %4 : i32
		%5 = db.mul %const10 : i32,%const10 : i32
		//CHECK: int(100)
		db.dump %5 : i32

  		%6 = db.div %constm10 : i32,%constm10 : i32
 		//CHECK: int(1)
 		db.dump %6 : i32
		%7 = db.div %const10 : i32,%const10 : i32
		//CHECK: int(1)
		db.dump %7 : i32

  		%8 = db.mod %constm10 : i32,%const4 : i32
 		//CHECK: int(-2)
 		db.dump %8 : i32
		%9 = db.mod %const10 : i32,%const4 : i32
		//CHECK: int(2)
		db.dump %9 : i32

 		return
  }
 }
