// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
 	 	%constm10 = db.constant ( -10.1 ) : f32
 	  	%constm1 = db.constant ( -1.2 ) : f32
  	 	%const0 = db.constant ( 0.3 ) : f32
    	%const1 = db.constant ( 1.4 ) : f32
 		%const10 = db.constant ( 10.5 ) : f32
 		%const4 = db.constant ( 4.6 ) : f32

 		//CHECK: float(-10.1)
 		db.dump %constm10 : f32
 		//CHECK: float(-1.2)
 		db.dump %constm1 : f32
  		//CHECK: float(0.3)
  		db.dump %const0 : f32
  		//CHECK: float(1.4)
  		db.dump %const1 : f32
  		//CHECK: float(10.5)
  		db.dump %const10 : f32

  		%0 = db.add %constm10 : f32,%constm1 : f32
 		//CHECK: float(-11.3)
 		db.dump %0 : f32
		%1 = db.add %const10 : f32,%const1 : f32
		//CHECK: float(11.9)
		db.dump %1 : f32

  		%2 = db.sub %constm10 : f32,%constm1 : f32
 		//CHECK: float(-8.9)
 		db.dump %2 : f32
		%3 = db.sub %const10 : f32,%const1 : f32
		//CHECK: float(9.1)
		db.dump %3 : f32

  		%4 = db.mul %constm10 : f32,%constm10 : f32
 		//CHECK: float(102.01)
 		db.dump %4 : f32
		%5 = db.mul %const10 : f32,%const10 : f32
		//CHECK: float(110.25)
		db.dump %5 : f32

  		%6 = db.div %constm10 : f32,%constm10 : f32
 		//CHECK: float(1)
 		db.dump %6 : f32
		%7 = db.div %const10 : f32,%const10 : f32
		//CHECK: float(1)
		db.dump %7 : f32

  		%8 = db.mod %constm10 : f32,%const4 : f32
 		//CHECK: float(-0.900001)
 		db.dump %8 : f32
		%9 = db.mod %const10 : f32,%const4 : f32
		//CHECK: float(1.3)
		db.dump %9 : f32

 		return
  }
 }
