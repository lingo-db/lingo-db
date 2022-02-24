// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
  	 	%const0 = db.constant ( 0 ) : ui32
    	%const1 = db.constant ( 1 ) : ui32
 		%const10 = db.constant ( 10 ) : ui32
 		%const4 = db.constant ( 4 ) : ui32
 	 	%constmax = db.constant ( 4294967295 ) : ui32

  		//CHECK: uint(0)
  		db.dump %const0 : ui32
  		//CHECK: uint(1)
  		db.dump %const1 : ui32
  		//CHECK: uint(10)
  		db.dump %const10 : ui32
 		//CHECK: uint(4294967295)
 		db.dump %constmax : ui32

		%1 = db.add %const10 : ui32,%const1 : ui32
		//CHECK: uint(11)
		db.dump %1 : ui32


		%3 = db.sub %const10 : ui32,%const1 : ui32
		//CHECK: uint(9)
		db.dump %3 : ui32


		%5 = db.mul %const10 : ui32,%const10 : ui32
		//CHECK: uint(100)
		db.dump %5 : ui32


		%7 = db.div %const10 : ui32,%const10 : ui32
		//CHECK: uint(1)
		db.dump %7 : ui32

		%9 = db.mod %const10 : ui32,%const4 : ui32
		//CHECK: uint(2)
		db.dump %9 : ui32
 		return
  }
 }
