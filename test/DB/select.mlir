// RUN: db-run %s | FileCheck %s

 module {
  	func @test (%arg0: !db.bool) {
  	    %t = db.constant ( "true" ) :!db.string
  	    %f = db.constant ( "false" ) :!db.string
  		%1 = db.select %arg0 : !db.bool, %t, %f : !db.string
        db.dump %1 : !db.string
  		return
  	}
   	func @test2 (%arg0: !db.nullable<!db.bool>) {
  	    %t = db.constant ( "true" ) :!db.string
  	    %f = db.constant ( "false" ) :!db.string
  		%1 = db.select %arg0 : !db.nullable<!db.bool>, %t, %f : !db.string
        db.dump %1 : !db.string
  		return
   	}

 	func @main () {
 		%false = db.constant ( 0 ) : !db.bool
 		%true = db.constant ( 1 ) : !db.bool
 		%null = db.null : !db.nullable<!db.bool>
 		%false_nullable = db.cast %false : !db.bool -> !db.nullable<!db.bool>
        %true_nullable = db.cast %true : !db.bool -> !db.nullable<!db.bool>
 		//CHECK: string("false")
 		call  @test(%false) : (!db.bool) -> ()
 		 //CHECK: string("true")
 		call  @test(%true) : (!db.bool) -> ()
        //CHECK: string("false")
        call  @test2(%false_nullable) : (!db.nullable<!db.bool>) -> ()
         //CHECK: string("true")
        call  @test2(%true_nullable) : (!db.nullable<!db.bool>) -> ()
        //CHECK: string("false")
        call  @test2(%null) : (!db.nullable<!db.bool>) -> ()
 		return
 	}
 }
