// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: if [ "$(uname)" = "Linux" ]; then env LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s; fi

 module {
 	func.func @test (%arg0: !db.nullable<i32>,%arg1: !db.nullable<i32>) -> !db.nullable<i32> {
 		%0 = db.add  %arg0 : !db.nullable<i32>, %arg1 : !db.nullable<i32>
 		return %0 : !db.nullable<i32>
 	}
 	func.func @main () {
 		%0 = db.null : !db.nullable<i32>
 		%1 = call @test(%0, %0) : (!db.nullable<i32>,!db.nullable<i32>) -> !db.nullable<i32>
 		//CHECK: int(NULL)
        db.runtime_call "DumpValue" (%1) : (!db.nullable<i32>) -> ()
 		return
 	}
 }
