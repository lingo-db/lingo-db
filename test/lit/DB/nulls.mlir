// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

 module {
	func.func @test (%arg0: !db.nullable<i1>) {
	 	db.runtime_call "DumpValue" (%arg0) : (!db.nullable<i1>) -> ()
		%1 = db.isnull %arg0 : !db.nullable<i1>
		db.runtime_call "DumpValue" (%1) : (i1) -> ()
		return
	}
	func.func @main () {
 		%const = db.constant ( 1 ) : i1
 		%null = db.null : !db.nullable<i1>
 		%not_null = db.as_nullable %const  : i1 -> !db.nullable<i1>
 		//CHECK: bool(NULL)
 		//CHECK: bool(true)
 		call  @test(%null) : (!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		//CHECK: bool(false)
 		call  @test(%not_null) : (!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		%not_null_value =db.nullable_get_val %not_null :  !db.nullable<i1>
 		db.runtime_call "DumpValue" (%not_null_value) : (i1) -> ()
		//CHECK: bool(NULL)
		%const_null = db.as_nullable %const : i1, %const -> !db.nullable<i1>
		db.runtime_call "DumpValue" (%const_null) : (!db.nullable<i1>) -> ()
		return
	}
 }
