// RUN: run-mlir %s | FileCheck %s


 module {
 	func.func @main () {
 	 	%constm10 = db.constant ( -10.1 ) : f32
 	  	%constm1 = db.constant ( -1.2 ) : f32
  	 	%const0 = db.constant ( 0.3 ) : f32
    	%const1 = db.constant ( 1.4 ) : f32
 		%const10 = db.constant ( 10.5 ) : f32
 		%const4 = db.constant ( 4.6 ) : f32

 		//CHECK: float(-10.1)
 		db.runtime_call "DumpValue" (%constm10) : (f32) -> ()
 		//CHECK: float(-1.2)
 		db.runtime_call "DumpValue" (%constm1) : (f32) -> ()
  		//CHECK: float(0.3)
  		db.runtime_call "DumpValue" (%const0) : (f32) -> ()
  		//CHECK: float(1.4)
  		db.runtime_call "DumpValue" (%const1) : (f32) -> ()
  		//CHECK: float(10.5)
  		db.runtime_call "DumpValue" (%const10) : (f32) -> ()

  		%0 = db.add %constm10 : f32,%constm1 : f32
 		//CHECK: float(-11.3)
 		db.runtime_call "DumpValue" (%0) : (f32) -> ()
		%1 = db.add %const10 : f32,%const1 : f32
		//CHECK: float(11.9)
		db.runtime_call "DumpValue" (%1) : (f32) -> ()

  		%2 = db.sub %constm10 : f32,%constm1 : f32
 		//CHECK: float(-8.9)
 		db.runtime_call "DumpValue" (%2) : (f32) -> ()
		%3 = db.sub %const10 : f32,%const1 : f32
		//CHECK: float(9.1)
		db.runtime_call "DumpValue" (%3) : (f32) -> ()

  		%4 = db.mul %constm10 : f32,%constm10 : f32
 		//CHECK: float(102.01)
 		db.runtime_call "DumpValue" (%4) : (f32) -> ()
		%5 = db.mul %const10 : f32,%const10 : f32
		//CHECK: float(110.25)
		db.runtime_call "DumpValue" (%5) : (f32) -> ()

  		%6 = db.div %constm10 : f32,%constm10 : f32
 		//CHECK: float(1)
 		db.runtime_call "DumpValue" (%6) : (f32) -> ()
		%7 = db.div %const10 : f32,%const10 : f32
		//CHECK: float(1)
		db.runtime_call "DumpValue" (%7) : (f32) -> ()

  		%8 = db.mod %constm10 : f32,%const4 : f32
 		//CHECK: float(-0.900001)
 		db.runtime_call "DumpValue" (%8) : (f32) -> ()
		%9 = db.mod %const10 : f32,%const4 : f32
		//CHECK: float(1.3)
		db.runtime_call "DumpValue" (%9) : (f32) -> ()

 		return
  }
 }
