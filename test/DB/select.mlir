// RUN: db-run %s | FileCheck %s

 module {
  	func @test (%arg0: !db.bool) {
  	    %t = db.constant ( "true" ) :!db.string
  	    %f = db.constant ( "false" ) :!db.string
  		%1 = db.select %arg0 : !db.bool, %t, %f : !db.string
        db.dump %1 : !db.string
  		return
  	}
   	func @test2 (%arg0: !db.bool<nullable>) {
  	    %t = db.constant ( "true" ) :!db.string
  	    %f = db.constant ( "false" ) :!db.string
  		%1 = db.select %arg0 : !db.bool<nullable>, %t, %f : !db.string
        db.dump %1 : !db.string
  		return
   	}

 	func @main () {
 		%false = db.constant ( 0 ) : !db.bool
 		%true = db.constant ( 1 ) : !db.bool
 		%null = db.null : !db.bool<nullable>
 		%false_nullable = db.cast %false : !db.bool -> !db.bool<nullable>
        %true_nullable = db.cast %true : !db.bool -> !db.bool<nullable>
 		//CHECK: string("false")
 		call  @test(%false) : (!db.bool) -> ()
 		 //CHECK: string("true")
 		call  @test(%true) : (!db.bool) -> ()
        //CHECK: string("false")
        call  @test2(%false_nullable) : (!db.bool<nullable>) -> ()
         //CHECK: string("true")
        call  @test2(%true_nullable) : (!db.bool<nullable>) -> ()
        //CHECK: string("false")
        call  @test2(%null) : (!db.bool<nullable>) -> ()
 		return
 	}
 }
