// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
 	 	%constm10 = db.constant ( "-10.00000001" ) : !db.decimal<12,8>
 	  	%constm1 = db.constant ( "-1.00000002" ) : !db.decimal<12,8>
  	 	%const0 = db.constant ( "0.00000003" ) : !db.decimal<12,8>
    	%const1 = db.constant ( "1.00000004" ) : !db.decimal<12,8>
 		%const10 = db.constant ( "10.00000005" ) : !db.decimal<12,8>
 		%const4 = db.constant ( "4.00000006" ) : !db.decimal<12,8>

 		//CHECK: decimal(-10.00000001)
 		db.dump %constm10 : !db.decimal<12,8>
 		//CHECK: decimal(-1.00000002)
 		db.dump %constm1 : !db.decimal<12,8>
  		//CHECK: decimal(3.E-8)
  		db.dump %const0 : !db.decimal<12,8>
  		//CHECK: decimal(1.00000004)
  		db.dump %const1 : !db.decimal<12,8>
  		//CHECK: decimal(10.00000005)
  		db.dump %const10 : !db.decimal<12,8>

  		%0 = db.add %constm10 : !db.decimal<12,8>,%constm1 : !db.decimal<12,8>
 		//CHECK: decimal(-11.00000003)
 		db.dump %0 : !db.decimal<12,8>
		%1 = db.add %const10 : !db.decimal<12,8>,%const1 : !db.decimal<12,8>
		//CHECK: decimal(11.00000009)
		db.dump %1 : !db.decimal<12,8>

  		%2 = db.sub %constm10 : !db.decimal<12,8>,%constm1 : !db.decimal<12,8>
 		//CHECK: decimal(-8.99999999)
 		db.dump %2 : !db.decimal<12,8>
		%3 = db.sub %const10 : !db.decimal<12,8>,%const1 : !db.decimal<12,8>
		//CHECK: decimal(9.00000001)
		db.dump %3 : !db.decimal<12,8>

  		%4 = db.mul %constm10 : !db.decimal<12,8>,%constm10 : !db.decimal<12,8>
 		//CHECK: decimal(10000000020.00000001)
 		db.dump %4 : !db.decimal<12,8>
		%5 = db.mul %const10 : !db.decimal<12,8>,%const10 : !db.decimal<12,8>
		//CHECK: decimal(10000000100.00000025)
		db.dump %5 : !db.decimal<12,8>

  		%6 = db.div %constm10 : !db.decimal<12,8>,%constm10 : !db.decimal<12,8>
 		//CHECK: decimal(1.00000000)
 		db.dump %6 : !db.decimal<12,8>
		%7 = db.div %const10 : !db.decimal<12,8>,%const10 : !db.decimal<12,8>
		//CHECK: decimal(1.00000000)
		db.dump %7 : !db.decimal<12,8>

  		%8 = db.mod %constm10 : !db.decimal<12,8>,%const4 : !db.decimal<12,8>
 		//CHECK: decimal(-2.00000024)
 		db.dump %8 : !db.decimal<12,8>
		%9 = db.mod %const10 : !db.decimal<12,8>,%const4 : !db.decimal<12,8>
		//CHECK: decimal(2.00000018)
		db.dump %9 : !db.decimal<12,8>

 		return
  }
 }
