// RUN: db-run %s | FileCheck %s

 module {
 	func @test (%arg0: !db.int<32,nullable>,%arg1: !db.int<32,nullable>) -> !db.int<32,nullable> {
 		%0 = db.add  %arg0 : !db.int<32,nullable>, %arg1 : !db.int<32,nullable>
 		return %0 : !db.int<32,nullable>
 	}
 	func @main () {
 		%0 = db.null : !db.int<32,nullable>
 		%1 = call @test(%0, %0) : (!db.int<32,nullable>,!db.int<32,nullable>) -> !db.int<32,nullable>
 		//CHECK: int(NULL)
        db.dump %1 : !db.int<32,nullable>
 		return
 	}
 }
