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
   	func @test2 (%arg0: !db.bool) {
   		db.if %arg0  : !db.bool{
          			%2 = db.constant ( "true" ) :!db.string
          			db.dump %2 : !db.string
          			db.yield
          			   		} else {
           			%2 = db.constant ( "false" ) :!db.string
          			db.dump %2 : !db.string
          			db.yield
          		}
   		return
   	}
  	func @test3 (%arg0: !db.nullable<!db.bool>) {
  		%1 = db.if %arg0  : !db.nullable<!db.bool>  -> !db.string {
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
 		%false_nullable = db.cast %false : !db.bool -> !db.nullable<!db.bool>
        %true_nullable = db.cast %true : !db.bool -> !db.nullable<!db.bool>
 		//CHECK: string("false")
 		call  @test(%false) : (!db.bool) -> ()
 		 //CHECK: string("true")
 		call  @test(%true) : (!db.bool) -> ()
  		//CHECK: string("false")
  		call  @test2(%false) : (!db.bool) -> ()
  		 //CHECK: string("true")
  		call  @test2(%true) : (!db.bool) -> ()
        //CHECK: string("false")
        call  @test3(%false_nullable) : (!db.nullable<!db.bool>) -> ()
         //CHECK: string("true")
        call  @test3(%true_nullable) : (!db.nullable<!db.bool>) -> ()
 		return
 	}
 }
