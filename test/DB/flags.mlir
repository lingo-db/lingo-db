// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
 		 %flag1 = db.createflag
 		 %flag2 = db.createflag
         db.setflag %flag1
         %bool1 = db.getflag %flag1
         %bool2 = db.getflag %flag2
         //CHECK: bool(true)
 		 db.dump %bool1 : !db.bool
 		 //CHECK: bool(false)
 		 db.dump %bool2 : !db.bool

 		return
  }
 }
