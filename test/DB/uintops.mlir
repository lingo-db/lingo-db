// RUN: run-mlir %s | FileCheck %s


 module {
 	func @main () {
  	 	%const0 = db.constant ( 0 ) : ui32
    	%const1 = db.constant ( 1 ) : ui32
 		%const10 = db.constant ( 10 ) : ui32
 		%const4 = db.constant ( 4 ) : ui32
 	 	%constmax = db.constant ( 4294967295 ) : ui32

  		//CHECK: uint(0)
  		db.runtime_call "DumpValue" (%const0) : (ui32) -> ()
  		//CHECK: uint(1)
  		db.runtime_call "DumpValue" (%const1) : (ui32) -> ()
  		//CHECK: uint(10)
  		db.runtime_call "DumpValue" (%const10) : (ui32) -> ()
 		//CHECK: uint(4294967295)
 		db.runtime_call "DumpValue" (%constmax) : (ui32) -> ()

		%1 = db.add %const10 : ui32,%const1 : ui32
		//CHECK: uint(11)
		db.runtime_call "DumpValue" (%1) : (ui32) -> ()


		%3 = db.sub %const10 : ui32,%const1 : ui32
		//CHECK: uint(9)
		db.runtime_call "DumpValue" (%3) : (ui32) -> ()


		%5 = db.mul %const10 : ui32,%const10 : ui32
		//CHECK: uint(100)
		db.runtime_call "DumpValue" (%5) : (ui32) -> ()


		%7 = db.div %const10 : ui32,%const10 : ui32
		//CHECK: uint(1)
		db.runtime_call "DumpValue" (%7) : (ui32) -> ()

		%9 = db.mod %const10 : ui32,%const4 : ui32
		//CHECK: uint(2)
		db.runtime_call "DumpValue" (%9) : (ui32) -> ()
 		return
  }
 }
