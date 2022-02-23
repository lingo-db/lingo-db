// RUN: db-run %s | FileCheck %s

 module {
 	func @test (%arg0: !db.nullable<!db.int<32>>,%arg1: !db.nullable<!db.int<32>>) -> !db.nullable<!db.int<32>> {
 		%0 = db.add  %arg0 : !db.nullable<!db.int<32>>, %arg1 : !db.nullable<!db.int<32>>
 		return %0 : !db.nullable<!db.int<32>>
 	}
 	func @main () {
 		%0 = db.null : !db.nullable<!db.int<32>>
 		%1 = call @test(%0, %0) : (!db.nullable<!db.int<32>>,!db.nullable<!db.int<32>>) -> !db.nullable<!db.int<32>>
 		//CHECK: int(NULL)
        db.dump %1 : !db.nullable<!db.int<32>>
 		return
 	}
 }
