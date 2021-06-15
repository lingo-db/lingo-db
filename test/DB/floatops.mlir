// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
 	 	%constm10 = db.constant ( -10.1 ) : !db.float<32>
 	  	%constm1 = db.constant ( -1.2 ) : !db.float<32>
  	 	%const0 = db.constant ( 0.3 ) : !db.float<32>
    	%const1 = db.constant ( 1.4 ) : !db.float<32>
 		%const10 = db.constant ( 10.5 ) : !db.float<32>
 		%const4 = db.constant ( 4.6 ) : !db.float<32>

 		//CHECK: float(-10.1)
 		db.dump %constm10 : !db.float<32>
 		//CHECK: float(-1.2)
 		db.dump %constm1 : !db.float<32>
  		//CHECK: float(0.3)
  		db.dump %const0 : !db.float<32>
  		//CHECK: float(1.4)
  		db.dump %const1 : !db.float<32>
  		//CHECK: float(10.5)
  		db.dump %const10 : !db.float<32>

  		%0 = db.add %constm10 : !db.float<32>,%constm1 : !db.float<32>
 		//CHECK: float(-11.3)
 		db.dump %0 : !db.float<32>
		%1 = db.add %const10 : !db.float<32>,%const1 : !db.float<32>
		//CHECK: float(11.9)
		db.dump %1 : !db.float<32>

  		%2 = db.sub %constm10 : !db.float<32>,%constm1 : !db.float<32>
 		//CHECK: float(-8.9)
 		db.dump %2 : !db.float<32>
		%3 = db.sub %const10 : !db.float<32>,%const1 : !db.float<32>
		//CHECK: float(9.1)
		db.dump %3 : !db.float<32>

  		%4 = db.mul %constm10 : !db.float<32>,%constm10 : !db.float<32>
 		//CHECK: float(102.01)
 		db.dump %4 : !db.float<32>
		%5 = db.mul %const10 : !db.float<32>,%const10 : !db.float<32>
		//CHECK: float(110.25)
		db.dump %5 : !db.float<32>

  		%6 = db.div %constm10 : !db.float<32>,%constm10 : !db.float<32>
 		//CHECK: float(1)
 		db.dump %6 : !db.float<32>
		%7 = db.div %const10 : !db.float<32>,%const10 : !db.float<32>
		//CHECK: float(1)
		db.dump %7 : !db.float<32>

  		%8 = db.mod %constm10 : !db.float<32>,%const4 : !db.float<32>
 		//CHECK: float(-0.900001)
 		db.dump %8 : !db.float<32>
		%9 = db.mod %const10 : !db.float<32>,%const4 : !db.float<32>
		//CHECK: float(1.3)
		db.dump %9 : !db.float<32>

 		return
  }
 }
