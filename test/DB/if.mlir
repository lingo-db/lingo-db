// RUN: db-run %s | FileCheck %s

 module {
  	func @test (%arg0: i1) {
  		%1 = db.if %arg0  : i1  -> !db.string {
         			%2 = db.constant ( "true" ) :!db.string
         			db.yield %2 :!db.string
         		} else {
          			%2 = db.constant ( "false" ) :!db.string
          			db.yield %2 :!db.string
         		}
        db.dump %1 : !db.string
  		return
  	}
   	func @test2 (%arg0: i1) {
   		db.if %arg0  : i1{
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
  	func @test3 (%arg0: !db.nullable<i1>) {
  		%1 = db.if %arg0  : !db.nullable<i1>  -> !db.string {
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
 		%false = db.constant ( 0 ) : i1
 		%true = db.constant ( 1 ) : i1
 		%false_nullable = db.cast %false : i1 -> !db.nullable<i1>
        %true_nullable = db.cast %true : i1 -> !db.nullable<i1>
 		//CHECK: string("false")
 		call  @test(%false) : (i1) -> ()
 		 //CHECK: string("true")
 		call  @test(%true) : (i1) -> ()
  		//CHECK: string("false")
  		call  @test2(%false) : (i1) -> ()
  		 //CHECK: string("true")
  		call  @test2(%true) : (i1) -> ()
        //CHECK: string("false")
        call  @test3(%false_nullable) : (!db.nullable<i1>) -> ()
         //CHECK: string("true")
        call  @test3(%true_nullable) : (!db.nullable<i1>) -> ()
 		return
 	}
 }
