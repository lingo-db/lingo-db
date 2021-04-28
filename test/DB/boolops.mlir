// RUN: db-run %s | FileCheck %s

 module {
	func @test_or (%arg0: !db.bool,%arg1: !db.bool,%arg2: !db.bool) {
		%1 = db.or %arg0 : !db.bool, %arg1 : !db.bool, %arg2 : !db.bool
		db.dump %1 : !db.bool
		return
	}
	func @test_or_nullable (%arg0: !db.bool<nullable>,%arg1: !db.bool<nullable>,%arg2: !db.bool<nullable>) {
		%1 = db.or %arg0 : !db.bool<nullable>, %arg1 : !db.bool<nullable>, %arg2 : !db.bool<nullable>
		db.dump %1 : !db.bool<nullable>
		return
	}
	func @test_and (%arg0: !db.bool,%arg1: !db.bool,%arg2: !db.bool) {
		%1 = db.and %arg0 : !db.bool, %arg1 : !db.bool, %arg2 : !db.bool
		db.dump %1 : !db.bool
		return
	}
	func @test_and_nullable (%arg0: !db.bool<nullable>,%arg1: !db.bool<nullable>,%arg2: !db.bool<nullable>) {
		%1 = db.and %arg0 : !db.bool<nullable>, %arg1 : !db.bool<nullable>, %arg2 : !db.bool<nullable>
		db.dump %1 : !db.bool<nullable>
		return
	}	
	func @test_ands () {
 		%false = db.constant ( 0 ) : !db.bool
 		%true = db.constant ( 1 ) : !db.bool
 		//CHECK: bool(false)
 		call @test_and(%false,%false,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(false)
 		call @test_and(%false,%false,%true) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(false)
 		call @test_and(%false,%true,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(false)
 		call @test_and(%false,%true,%true) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(false)
 		call @test_and(%true,%false,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(false)
 		call @test_and(%true,%false,%true) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(false)
 		call @test_and(%true,%true,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_and(%true,%true,%true) : (!db.bool,!db.bool,!db.bool) -> ()
		%sep =db.constant ( "and-------------" ) : !db.string
		db.dump %sep : !db.string
 		%false_nullable = db.cast %false : !db.bool -> !db.bool<nullable>
 		%true_nullable = db.cast %true : !db.bool -> !db.bool<nullable>
 		%null_nullable = db.null : !db.bool<nullable>
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%false_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%false_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%false_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%true_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%true_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(false)
		call @test_and_nullable (%false_nullable,%true_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%null_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%false_nullable,%null_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(false)
		call @test_and_nullable (%false_nullable,%null_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%false_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%false_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%false_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%true_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_and_nullable (%true_nullable,%true_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%true_nullable,%true_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%true_nullable,%null_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_and_nullable (%true_nullable,%null_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%true_nullable,%null_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%false_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%false_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%false_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%true_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_and_nullable (%null_nullable,%true_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%null_nullable,%true_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(false)
 		call @test_and_nullable (%null_nullable,%null_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_and_nullable (%null_nullable,%null_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(NULL)
		call @test_and_nullable (%null_nullable,%null_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		return
	}
	func @test_ors () {
 		%false = db.constant ( 0 ) : !db.bool
 		%true = db.constant ( 1 ) : !db.bool
 		//CHECK: bool(false)
 		call @test_or(%false,%false,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_or(%false,%false,%true) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_or(%false,%true,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_or(%false,%true,%true) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%false,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%false,%true) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%true,%false) : (!db.bool,!db.bool,!db.bool) -> ()
 		//CHECK: bool(true)
 		call @test_or(%true,%true,%true) : (!db.bool,!db.bool,!db.bool) -> ()
		%sep =db.constant ( "or-------------" ) : !db.string
		db.dump %sep : !db.string
 		%false_nullable = db.cast %false : !db.bool -> !db.bool<nullable>
 		%true_nullable = db.cast %true : !db.bool -> !db.bool<nullable>
 		%null_nullable = db.null : !db.bool<nullable>
 		//CHECK: bool(false)
 		call @test_or_nullable (%false_nullable,%false_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%false_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%false_nullable,%false_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%true_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%true_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%false_nullable,%true_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%false_nullable,%null_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%false_nullable,%null_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(NULL)
		call @test_or_nullable (%false_nullable,%null_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%false_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%false_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%false_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%true_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%true_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%true_nullable,%true_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%null_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%true_nullable,%null_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%true_nullable,%null_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%null_nullable,%false_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%false_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%null_nullable,%false_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%true_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%true_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(true)
		call @test_or_nullable (%null_nullable,%true_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(NULL)
 		call @test_or_nullable (%null_nullable,%null_nullable,%false_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		//CHECK: bool(true)
 		call @test_or_nullable (%null_nullable,%null_nullable,%true_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
		//CHECK: bool(NULL)
		call @test_or_nullable (%null_nullable,%null_nullable,%null_nullable) : (!db.bool<nullable>,!db.bool<nullable>,!db.bool<nullable>) -> ()
 		return
	}
	func @test_not () {
 		%false = db.constant ( 0 ) : !db.bool
 		%true = db.constant ( 1 ) : !db.bool
		%false_nullable = db.cast %false : !db.bool -> !db.bool<nullable>
		%true_nullable = db.cast %true : !db.bool -> !db.bool<nullable>
		%null_nullable = db.null : !db.bool<nullable>
 		%not_true = db.not %true : !db.bool
 		%not_false = db.not %false : !db.bool
		%not_true_nullable = db.not %true_nullable : !db.bool<nullable>
		%not_false_nullable = db.not %false_nullable : !db.bool<nullable>
		%not_null_nullable = db.not %null_nullable : !db.bool<nullable>
 		//CHECK: bool(false)
 		db.dump %not_true : !db.bool
  		//CHECK: bool(true)
  		db.dump %not_false : !db.bool
		//CHECK: bool(false)
		db.dump %not_true_nullable : !db.bool<nullable>
		//CHECK: bool(true)
		db.dump %not_false_nullable : !db.bool<nullable>
		//CHECK: bool(NULL)
		db.dump %not_null_nullable : !db.bool<nullable>
 		return
 	}
	func @main () {
		call @test_ands () : () -> ()
		call @test_ors () : () -> ()
		call @test_not () : () -> ()
		return
	}

 }