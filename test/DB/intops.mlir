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
 		db.runtime_call "DumpValue" (%constm10) : (i32) -> ()
 		//CHECK: int(-1)
 		db.runtime_call "DumpValue" (%constm1) : (i32) -> ()
  		//CHECK: int(0)
  		db.runtime_call "DumpValue" (%const0) : (i32) -> ()
  		//CHECK: int(1)
  		db.runtime_call "DumpValue" (%const1) : (i32) -> ()
  		//CHECK: int(10)
  		db.runtime_call "DumpValue" (%const10) : (i32) -> ()

  		%0 = db.add %constm10 : i32,%constm1 : i32
 		//CHECK: int(-11)
 		db.runtime_call "DumpValue" (%0) : (i32) -> ()
		%1 = db.add %const10 : i32,%const1 : i32
		//CHECK: int(11)
		db.runtime_call "DumpValue" (%1) : (i32) -> ()

  		%2 = db.sub %constm10 : i32,%constm1 : i32
 		//CHECK: int(-9)
 		db.runtime_call "DumpValue" (%2) : (i32) -> ()
		%3 = db.sub %const10 : i32,%const1 : i32
		//CHECK: int(9)
		db.runtime_call "DumpValue" (%3) : (i32) -> ()

  		%4 = db.mul %constm10 : i32,%constm10 : i32
 		//CHECK: int(100)
 		db.runtime_call "DumpValue" (%4) : (i32) -> ()
		%5 = db.mul %const10 : i32,%const10 : i32
		//CHECK: int(100)
		db.runtime_call "DumpValue" (%5) : (i32) -> ()

  		%6 = db.div %constm10 : i32,%constm10 : i32
 		//CHECK: int(1)
 		db.runtime_call "DumpValue" (%6) : (i32) -> ()
		%7 = db.div %const10 : i32,%const10 : i32
		//CHECK: int(1)
		db.runtime_call "DumpValue" (%7) : (i32) -> ()

  		%8 = db.mod %constm10 : i32,%const4 : i32
 		//CHECK: int(-2)
 		db.runtime_call "DumpValue" (%8) : (i32) -> ()
		%9 = db.mod %const10 : i32,%const4 : i32
		//CHECK: int(2)
		db.runtime_call "DumpValue" (%9) : (i32) -> ()

 		return
  }
 }
