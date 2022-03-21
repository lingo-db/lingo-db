 // RUN: db-run %s | FileCheck %s
 module {
	func @main () {
		%false = db.constant ( 0 ) : i1
		%true = db.constant ( 1 ) : i1
		%0 = util.undef_tuple : tuple<i1, i1>
		%1 = util.set_tuple %0[0]= %true : (tuple<i1, i1>,i1) -> tuple<i1, i1>
		%2 = util.set_tuple %1[1]= %false : (tuple<i1, i1>,i1) -> tuple<i1, i1>
		%3,%4 = util.unpack %2 : tuple<i1, i1> -> i1, i1
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.runtime_call "DumpValue" (%3) : (i1) -> ()
		db.runtime_call "DumpValue" (%4) : (i1) -> ()

		%5 = util.pack %true, %false : i1, i1 -> tuple<i1, i1>
		%6,%7 = util.unpack %5 : tuple<i1, i1> -> i1, i1
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.runtime_call "DumpValue" (%6) : (i1) -> ()
		db.runtime_call "DumpValue" (%7) : (i1) -> ()

		%8 = util.pack %true, %false : i1, i1 -> tuple<i1, i1>
		%9 = util.pack %8, %8 : tuple<i1, i1>,tuple<i1, i1> -> tuple<tuple<i1, i1>,tuple<i1, i1>>
		%10,%11 = util.unpack %9 : tuple<tuple<i1, i1>,tuple<i1, i1>> -> tuple<i1, i1>,tuple<i1, i1>
		%12,%13 = util.unpack %10 : tuple<i1, i1> -> i1, i1
		%14,%15 = util.unpack %10 : tuple<i1, i1> -> i1, i1
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.runtime_call "DumpValue" (%12) : (i1) -> ()
		db.runtime_call "DumpValue" (%13) : (i1) -> ()
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.runtime_call "DumpValue" (%14) : (i1) -> ()
		db.runtime_call "DumpValue" (%15) : (i1) -> ()
		return
	}
 }