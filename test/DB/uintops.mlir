// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
  	 	%const0 = db.constant ( 0 ) : !db.uint<32>
    	%const1 = db.constant ( 1 ) : !db.uint<32>
 		%const10 = db.constant ( 10 ) : !db.uint<32>
 		%const4 = db.constant ( 4 ) : !db.uint<32>
 	 	%constmax = db.constant ( 4294967295 ) : !db.uint<32>

  		//CHECK: uint(0)
  		db.dump %const0 : !db.uint<32>
  		//CHECK: uint(1)
  		db.dump %const1 : !db.uint<32>
  		//CHECK: uint(10)
  		db.dump %const10 : !db.uint<32>
 		//CHECK: uint(4294967295)
 		db.dump %constmax : !db.uint<32>

		%1 = db.add %const10 : !db.uint<32>,%const1 : !db.uint<32>
		//CHECK: uint(11)
		db.dump %1 : !db.uint<32>


		%3 = db.sub %const10 : !db.uint<32>,%const1 : !db.uint<32>
		//CHECK: uint(9)
		db.dump %3 : !db.uint<32>


		%5 = db.mul %const10 : !db.uint<32>,%const10 : !db.uint<32>
		//CHECK: uint(100)
		db.dump %5 : !db.uint<32>


		%7 = db.div %const10 : !db.uint<32>,%const10 : !db.uint<32>
		//CHECK: uint(1)
		db.dump %7 : !db.uint<32>

		%9 = db.mod %const10 : !db.uint<32>,%const4 : !db.uint<32>
		//CHECK: uint(2)
		db.dump %9 : !db.uint<32>
 		return
  }
 }
