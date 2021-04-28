// RUN: db-run %s | FileCheck %s

 module {
  	func @test (%arg0: !db.bool) {
  		%1 = db.if %arg0  : !db.bool  -> !db.string {
         			%2 = db.constant ( "true" ) :!db.string
         			db.yield %2 :!db.string
         		} else {
          			%2 = db.constant ( "false" ) :!db.string
          			db.yield %2 :!db.string
         		}
        db.dump %1 : !db.string
  		return
  	}
 	func @main () {
 		%false = db.constant ( 0 ) : !db.bool
 		%true = db.constant ( 1 ) : !db.bool
 		//CHECK: string("false")
 		call  @test(%false) : (!db.bool) -> ()
 		 //CHECK: string("true")
 		call  @test(%true) : (!db.bool) -> ()
 		return
 	}
 }
