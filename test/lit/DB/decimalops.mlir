// RUN: run-mlir %s | FileCheck %s


 module {
 	func.func @main () {
 	 	%constm10 = db.constant ( "-10.00000001" ) : !db.decimal<12,8>
 	  	%constm1 = db.constant ( "-1.00000002" ) : !db.decimal<12,8>
  	 	%const0 = db.constant ( "0.00000003" ) : !db.decimal<12,8>
    	%const1 = db.constant ( "1.00000004" ) : !db.decimal<12,8>
 		%const10 = db.constant ( "10.00000005" ) : !db.decimal<12,8>
 		%const4 = db.constant ( "4.00000006" ) : !db.decimal<12,8>

 		//CHECK: decimal(-10.00000001)
 		db.runtime_call "DumpValue" (%constm10) : (!db.decimal<12,8>) -> ()
 		//CHECK: decimal(-1.00000002)
 		db.runtime_call "DumpValue" (%constm1) : (!db.decimal<12,8>) -> ()
  		//CHECK: decimal(3E-8)
  		db.runtime_call "DumpValue" (%const0) : (!db.decimal<12,8>) -> ()
  		//CHECK: decimal(1.00000004)
  		db.runtime_call "DumpValue" (%const1) : (!db.decimal<12,8>) -> ()
  		//CHECK: decimal(10.00000005)
  		db.runtime_call "DumpValue" (%const10) : (!db.decimal<12,8>) -> ()

  		%0 = db.add %constm10 : !db.decimal<12,8>,%constm1 : !db.decimal<12,8>
 		//CHECK: decimal(-11.00000003)
 		db.runtime_call "DumpValue" (%0) : (!db.decimal<12,8>) -> ()
		%1 = db.add %const10 : !db.decimal<12,8>,%const1 : !db.decimal<12,8>
		//CHECK: decimal(11.00000009)
		db.runtime_call "DumpValue" (%1) : (!db.decimal<12,8>) -> ()

  		%2 = db.sub %constm10 : !db.decimal<12,8>,%constm1 : !db.decimal<12,8>
 		//CHECK: decimal(-8.99999999)
 		db.runtime_call "DumpValue" (%2) : (!db.decimal<12,8>) -> ()
		%3 = db.sub %const10 : !db.decimal<12,8>,%const1 : !db.decimal<12,8>
		//CHECK: decimal(9.00000001)
		db.runtime_call "DumpValue" (%3) : (!db.decimal<12,8>) -> ()

  		%4 = db.mul %constm10 : !db.decimal<12,8>,%constm10 : !db.decimal<12,8>
 		//CHECK: decimal(100.0000002000000001)
 		db.runtime_call "DumpValue" (%4) : (!db.decimal<24,16>) -> ()
		%5 = db.mul %const10 : !db.decimal<12,8>,%const10 : !db.decimal<12,8>
		//CHECK: decimal(100.0000010000000025)
		db.runtime_call "DumpValue" (%5) : (!db.decimal<24,16>) -> ()

  		%6 = db.div %constm10 : !db.decimal<12,8>,%constm10 : !db.decimal<12,8>
 		//CHECK: decimal(1.00000000000000000000)
 		db.runtime_call "DumpValue" (%6) : (!db.decimal<32, 20>) -> ()
		%7 = db.div %const10 : !db.decimal<12,8>,%const10 : !db.decimal<12,8>
		//CHECK: decimal(1.00000000000000000000)
		db.runtime_call "DumpValue" (%7) : (!db.decimal<32, 20>) -> ()

  		%8 = db.mod %constm10 : !db.decimal<12,8>,%const4 : !db.decimal<12,8>
 		//CHECK: decimal(-2.00000024)
 		db.runtime_call "DumpValue" (%8) : (!db.decimal<12, 8>) -> ()
		%9 = db.mod %const10 : !db.decimal<12,8>,%const4 : !db.decimal<12,8>
		//CHECK: decimal(2.00000018)
		db.runtime_call "DumpValue" (%9) : (!db.decimal<12, 8>) -> ()

 		return
  }
 }
