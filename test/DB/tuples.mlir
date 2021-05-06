 // RUN: db-run %s | FileCheck %s
 module {
	func @main () {
		%false = db.constant ( 0 ) : !db.bool
		%true = db.constant ( 1 ) : !db.bool
		%0 = util.undef_tuple : tuple<!db.bool, !db.bool>
		%1 = util.set_tuple %0[0]= %true : (tuple<!db.bool, !db.bool>,!db.bool) -> tuple<!db.bool, !db.bool>
		%2 = util.set_tuple %1[1]= %false : (tuple<!db.bool, !db.bool>,!db.bool) -> tuple<!db.bool, !db.bool>
		%3,%4 = util.unpack %2 : tuple<!db.bool, !db.bool> -> !db.bool, !db.bool
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.dump %3 : !db.bool
		db.dump %4 : !db.bool

		%5 = util.pack %true, %false : !db.bool, !db.bool -> tuple<!db.bool, !db.bool>
		%6,%7 = util.unpack %5 : tuple<!db.bool, !db.bool> -> !db.bool, !db.bool
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.dump %6 : !db.bool
		db.dump %7 : !db.bool

		%8 = util.pack %true, %false : !db.bool, !db.bool -> tuple<!db.bool, !db.bool>
		%9 = util.pack %8, %8 : tuple<!db.bool, !db.bool>,tuple<!db.bool, !db.bool> -> tuple<tuple<!db.bool, !db.bool>,tuple<!db.bool, !db.bool>>
		%10,%11 = util.unpack %9 : tuple<tuple<!db.bool, !db.bool>,tuple<!db.bool, !db.bool>> -> tuple<!db.bool, !db.bool>,tuple<!db.bool, !db.bool>
		%12,%13 = util.unpack %10 : tuple<!db.bool, !db.bool> -> !db.bool, !db.bool
		%14,%15 = util.unpack %10 : tuple<!db.bool, !db.bool> -> !db.bool, !db.bool
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.dump %12 : !db.bool
		db.dump %13 : !db.bool
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.dump %14 : !db.bool
		db.dump %15 : !db.bool
		return
	}
 }