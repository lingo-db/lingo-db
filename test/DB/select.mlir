// RUN: db-run %s | FileCheck %s

 module {
  	func @test (%arg0: i1) {
  	    %t = db.constant ( "true" ) :!db.string
  	    %f = db.constant ( "false" ) :!db.string
  		%1 = db.select %arg0 : i1, %t, %f : !db.string
        db.dump %1 : !db.string
  		return
  	}
   	func @test2 (%arg0: !db.nullable<i1>) {
  	    %t = db.constant ( "true" ) :!db.string
  	    %f = db.constant ( "false" ) :!db.string
  		%1 = db.select %arg0 : !db.nullable<i1>, %t, %f : !db.string
        db.dump %1 : !db.string
  		return
   	}

 	func @main () {
 		%false = db.constant ( 0 ) : i1
 		%true = db.constant ( 1 ) : i1
 		%null = db.null : !db.nullable<i1>
 		%false_nullable = db.cast %false : i1 -> !db.nullable<i1>
        %true_nullable = db.cast %true : i1 -> !db.nullable<i1>
 		//CHECK: string("false")
 		call  @test(%false) : (i1) -> ()
 		 //CHECK: string("true")
 		call  @test(%true) : (i1) -> ()
        //CHECK: string("false")
        call  @test2(%false_nullable) : (!db.nullable<i1>) -> ()
         //CHECK: string("true")
        call  @test2(%true_nullable) : (!db.nullable<i1>) -> ()
        //CHECK: string("false")
        call  @test2(%null) : (!db.nullable<i1>) -> ()
 		return
 	}
 }
