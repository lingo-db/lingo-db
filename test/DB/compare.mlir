// RUN: db-run %s | FileCheck %s


 module {
 	func @main () {
    	%const1 = db.constant ( 1 ) : !db.int<32>
    	%const2 = db.constant ( 2 ) : !db.int<32>
    	%constf1 = db.constant ( 1. ) : !db.float<32>
        %constf2 = db.constant ( 2. ) : !db.float<32>


		%const1_nullable = db.cast %const1 : !db.int<32> -> !db.int<32,nullable>
		%const2_nullable = db.cast %const2 : !db.int<32> -> !db.int<32,nullable>
		%null_nullable = db.null : !db.int<32,nullable>
		//CHECK: bool(NULL)
		%n_eq_n = db.compare eq %null_nullable:!db.int<32,nullable>, %null_nullable:!db.int<32,nullable>
		db.dump %n_eq_n : !db.bool<nullable>
		//CHECK: bool(NULL)
		%n_eq_one = db.compare eq %null_nullable:!db.int<32,nullable>, %const1_nullable:!db.int<32,nullable>
		db.dump %n_eq_one : !db.bool<nullable>
		//CHECK: bool(NULL)
		%one_eq_n = db.compare eq %const1_nullable:!db.int<32,nullable>, %null_nullable:!db.int<32,nullable>
		db.dump %one_eq_n : !db.bool<nullable>
		//CHECK: bool(true)
		%one_eq_one = db.compare eq %const1_nullable:!db.int<32,nullable>, %const1_nullable:!db.int<32,nullable>
		db.dump %one_eq_one : !db.bool<nullable>
		//CHECK: bool(false)
		%one_eq_two = db.compare eq %const1_nullable:!db.int<32,nullable>, %const2_nullable:!db.int<32,nullable>
		db.dump %one_eq_two : !db.bool<nullable>

		//CHECK: bool(true)
		%10 = db.compare eq %const1 : !db.int<32>, %const1 : !db.int<32>
		db.dump %10 : !db.bool
		//CHECK: bool(false)
		%11 = db.compare eq %const1 : !db.int<32>, %const2 : !db.int<32>
		db.dump %11 : !db.bool
		//CHECK: bool(false)
		%12 = db.compare neq %const1 : !db.int<32>, %const1 : !db.int<32>
		db.dump %12 : !db.bool
		//CHECK: bool(false)
		%13 = db.compare neq %const1 : !db.int<32>, %const1 : !db.int<32>
		db.dump %13 : !db.bool
		//CHECK: bool(true)
		%14 = db.compare neq %const1 : !db.int<32>, %const2 : !db.int<32>
		db.dump %14 : !db.bool
		//CHECK: bool(false)
		%15 = db.compare lt %const1 : !db.int<32>, %const1 : !db.int<32>
		db.dump %15 : !db.bool
		//CHECK: bool(true)
		%16 = db.compare lte %const1 : !db.int<32>, %const1 : !db.int<32>
		db.dump %16 : !db.bool
		//CHECK: bool(true)
		%17 = db.compare lt %const1 : !db.int<32>, %const2 : !db.int<32>
		db.dump %17 : !db.bool
		//CHECK: bool(true)
		%18 = db.compare lte %const1 : !db.int<32>, %const2 : !db.int<32>
		db.dump %18 : !db.bool
		//CHECK: bool(false)
		%19 = db.compare lt %const2 : !db.int<32>, %const1 : !db.int<32>
		db.dump %19 : !db.bool
		//CHECK: bool(false)
		%20 = db.compare lte %const2 : !db.int<32>, %const1 : !db.int<32>
		db.dump %20 : !db.bool
		//CHECK: bool(false)
		%21 = db.compare gt %const1 : !db.int<32>, %const1 : !db.int<32>
		db.dump %21 : !db.bool
		//CHECK: bool(true)
		%22 = db.compare gte %const1 : !db.int<32>, %const1 : !db.int<32>
		db.dump %22 : !db.bool
		//CHECK: bool(false)
		%23 = db.compare gt %const1 : !db.int<32>, %const2 : !db.int<32>
		db.dump %23 : !db.bool
		//CHECK: bool(false)
		%24 = db.compare gte %const1 : !db.int<32>, %const2 : !db.int<32>
		db.dump %24 : !db.bool
		//CHECK: bool(true)
		%25 = db.compare gt %const2 : !db.int<32>, %const1 : !db.int<32>
		db.dump %25 : !db.bool
		//CHECK: bool(true)
		%26 = db.compare gte %const2 : !db.int<32>, %const1 : !db.int<32>
		db.dump %26 : !db.bool
		
		//CHECK: bool(true)
		%30 = db.compare eq %constf1 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %30 : !db.bool
		//CHECK: bool(false)
		%31 = db.compare eq %constf1 : !db.float<32>, %constf2 : !db.float<32>
		db.dump %31 : !db.bool
		//CHECK: bool(false)
		%32 = db.compare neq %constf1 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %32 : !db.bool
		//CHECK: bool(false)
		%33 = db.compare neq %constf1 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %33 : !db.bool
		//CHECK: bool(true)
		%34 = db.compare neq %constf1 : !db.float<32>, %constf2 : !db.float<32>
		db.dump %34 : !db.bool
		//CHECK: bool(false)
		%35 = db.compare lt %constf1 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %35 : !db.bool
		//CHECK: bool(true)
		%36 = db.compare lte %constf1 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %36 : !db.bool
		//CHECK: bool(true)
		%37 = db.compare lt %constf1 : !db.float<32>, %constf2 : !db.float<32>
		db.dump %37 : !db.bool
		//CHECK: bool(true)
		%38 = db.compare lte %constf1 : !db.float<32>, %constf2 : !db.float<32>
		db.dump %38 : !db.bool
		//CHECK: bool(false)
		%39 = db.compare lt %constf2 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %39 : !db.bool
		//CHECK: bool(false)
		%40 = db.compare lte %constf2 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %40 : !db.bool
		//CHECK: bool(false)
		%41 = db.compare gt %constf1 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %41 : !db.bool
		//CHECK: bool(true)
		%42 = db.compare gte %constf1 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %42 : !db.bool
		//CHECK: bool(false)
		%43 = db.compare gt %constf1 : !db.float<32>, %constf2 : !db.float<32>
		db.dump %43 : !db.bool
		//CHECK: bool(false)
		%44 = db.compare gte %constf1 : !db.float<32>, %constf2 : !db.float<32>
		db.dump %44 : !db.bool
		//CHECK: bool(true)
		%45 = db.compare gt %constf2 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %45 : !db.bool
		//CHECK: bool(true)
		%46 = db.compare gte %constf2 : !db.float<32>, %constf1 : !db.float<32>
		db.dump %46 : !db.bool		
 		return
  }
 }