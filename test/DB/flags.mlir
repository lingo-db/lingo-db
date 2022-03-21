// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
 	 	 %true = db.constant ( 1 ) : i1
 		 %flag1 = db.createflag
 		 %flag2 = db.createflag
         db.setflag %flag1,%true
         %bool1 = db.getflag %flag1
         %bool2 = db.getflag %flag2
         //CHECK: bool(true)
 		 db.runtime_call "DumpValue" (%bool1) : (i1) -> ()
 		 //CHECK: bool(false)
 		 db.runtime_call "DumpValue" (%bool2) : (i1) -> ()

 		return
  }
 }
