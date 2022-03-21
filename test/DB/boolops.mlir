// RUN: db-run %s | FileCheck %s

 module {
	func @test_or (%arg0: i1,%arg1: i1,%arg2: i1) {
		%1 = db.or %arg0, %arg1, %arg2 : i1,i1,i1
		db.runtime_call "DumpValue" (%1) : (i1) -> ()
		return
	}
	func @test_or_nullable (%arg0: !db.nullable<i1>,%arg1: !db.nullable<i1>,%arg2: !db.nullable<i1>) {
		%1 = db.or %arg0, %arg1, %arg2 : !db.nullable<i1>, !db.nullable<i1>, !db.nullable<i1>
		db.runtime_call "DumpValue" (%1) : (!db.nullable<i1>) -> ()
		return
	}
	func @test_and (%arg0: i1,%arg1: i1,%arg2: i1) {
		%1 = db.and %arg0, %arg1, %arg2 : i1, i1, i1
		db.runtime_call "DumpValue" (%1) : (i1) -> ()
		return
	}
	func @test_and_nullable (%arg0: !db.nullable<i1>,%arg1: !db.nullable<i1>,%arg2: !db.nullable<i1>) {
		%1 = db.and %arg0, %arg1, %arg2 : !db.nullable<i1>, !db.nullable<i1>, !db.nullable<i1>
		db.runtime_call "DumpValue" (%1) : (!db.nullable<i1>) -> ()
		return
	}	
	func @test_ands () {
 		%false = db.constant ( 0 ) : i1
 		%true = db.constant ( 1 ) : i1
 		//CHECK: bool(false)
 		call @test_and(%false,%false,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(false)
 		call @test_and(%false,%false,%true) : (i1,i1,i1) -> ()
 		//CHECK: bool(false)
 		call @test_and(%false,%true,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(false)
 		call @test_and(%false,%true,%true) : (i1,i1,i1) -> ()
 		//CHECK: bool(false)
 		call @test_and(%true,%false,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(false)
 		call @test_and(%true,%false,%true) : (i1,i1,i1) -> ()
 		//CHECK: bool(false)
 		call @test_and(%true,%true,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_and(%true,%true,%true) : (i1,i1,i1) -> ()
		%sep =db.constant ( "and-------------" ) : !db.string
		db.runtime_call "DumpValue" (%sep) : (!db.string) -> ()
 		%false_nullable = db.as_nullable %false : i1 -> !db.nullable<i1>
 		%true_nullable = db.as_nullable %true : i1 -> !db.nullable<i1>
 		%null_nullable = db.null : !db.nullable<i1>
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%false_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%false_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%false_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%true_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%true_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(false)
		call @test_and_nullable (%false_nullable,%true_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%null_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%null_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(false)
		call @test_and_nullable (%false_nullable,%null_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%false_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%false_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%false_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%true_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_and_nullable (%true_nullable,%true_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%true_nullable,%true_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%null_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_and_nullable (%true_nullable,%null_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%true_nullable,%null_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%false_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%false_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%false_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%true_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_and_nullable (%null_nullable,%true_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%null_nullable,%true_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%null_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_and_nullable (%null_nullable,%null_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%null_nullable,%null_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		return
	}
	func @test_ors () {
 		%false = db.constant ( 0 ) : i1
 		%true = db.constant ( 1 ) : i1
 		//CHECK: bool(false)
 		call @test_or(%false,%false,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_or(%false,%false,%true) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_or(%false,%true,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_or(%false,%true,%true) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%false,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%false,%true) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%true,%false) : (i1,i1,i1) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%true,%true) : (i1,i1,i1) -> ()
		%sep =db.constant ( "or-------------" ) : !db.string
		db.runtime_call "DumpValue" (%sep) : (!db.string) -> ()
 		%false_nullable = db.as_nullable %false : i1 -> !db.nullable<i1>
 		%true_nullable = db.as_nullable %true : i1 -> !db.nullable<i1>
 		%null_nullable = db.null : !db.nullable<i1>
 		//CHECK: bool(false)
 		call @test_or_nullable (%false_nullable,%false_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%false_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%false_nullable,%false_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%true_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%true_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%false_nullable,%true_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%false_nullable,%null_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%null_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(NULL)
		call @test_or_nullable (%false_nullable,%null_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%false_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%false_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%false_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%true_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%true_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%true_nullable,%true_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%null_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%null_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%true_nullable,%null_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%null_nullable,%false_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%false_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%null_nullable,%false_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%true_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%true_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%null_nullable,%true_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%null_nullable,%null_nullable,%false_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%null_nullable,%true_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
		//CHECK: bool(NULL)
		call @test_or_nullable (%null_nullable,%null_nullable,%null_nullable) : (!db.nullable<i1>,!db.nullable<i1>,!db.nullable<i1>) -> ()
 		return
	}
	func @test_not () {
 		%false = db.constant ( 0 ) : i1
 		%true = db.constant ( 1 ) : i1
		%false_nullable = db.as_nullable %false : i1 -> !db.nullable<i1>
		%true_nullable = db.as_nullable %true : i1 -> !db.nullable<i1>
		%null_nullable = db.null : !db.nullable<i1>
 		%not_true = db.not %true : i1
 		%not_false = db.not %false : i1
		%not_true_nullable = db.not %true_nullable : !db.nullable<i1>
		%not_false_nullable = db.not %false_nullable : !db.nullable<i1>
		%not_null_nullable = db.not %null_nullable : !db.nullable<i1>
 		//CHECK: bool(false)
 		db.runtime_call "DumpValue" (%not_true) : (i1) -> ()
  		//CHECK: bool(true)
  		db.runtime_call "DumpValue" (%not_false) : (i1) -> ()
		//CHECK: bool(false)
		db.runtime_call "DumpValue" (%not_true_nullable) : (!db.nullable<i1>) -> ()
		//CHECK: bool(true)
		db.runtime_call "DumpValue" (%not_false_nullable) : (!db.nullable<i1>) -> ()
		//CHECK: bool(NULL)
		db.runtime_call "DumpValue" (%not_null_nullable) : (!db.nullable<i1>) -> ()
 		return
 	}
	func @main () {
		call @test_ands () : () -> ()
		call @test_ors () : () -> ()
		call @test_not () : () -> ()
		return
	}

 }