module @querymodule  {
  func @query() {
    %0 = relalg.basetable @R0  {table_identifier = "r0"} columns: {
    	A => @A({type = !db.int<32>})
    }
	%1 = relalg.basetable @R1  {table_identifier = "r1"} columns: {
		A => @A({type = !db.int<32>}),
		B => @B({type = !db.int<32>})
	}
	%2 = relalg.basetable @R2  {table_identifier = "r2"} columns: {
		B => @B({type = !db.int<32>}),
		C => @C({type = !db.int<32>})
	}
	%3 = relalg.basetable @R3  {table_identifier = "r3"} columns: {
		C => @C({type = !db.int<32>})
	}
	%4 = relalg.semijoin left %2,%3 (%arg0 : !relalg.tuple){
		%20 = relalg.getattr %arg0 @R2::@C : !db.int<32>
		%21 = relalg.getattr %arg0 @R3::@C : !db.int<32>
		%22 = db.compare eq %20 : !db.int<32>,%21 : !db.int<32>
		relalg.return %22 : !db.bool
	}
	%5 = relalg.join %1,%4 (%arg0 : !relalg.tuple){
		%20 = relalg.getattr %arg0 @R1::@B : !db.int<32>
		%21 = relalg.getattr %arg0 @R2::@B : !db.int<32>
		%22 = db.compare eq %20 : !db.int<32>,%21 : !db.int<32>
		relalg.return %22 : !db.bool
	}

	%6 = relalg.outerjoin left %0,%5 (%arg0 : !relalg.tuple){
		%20 = relalg.getattr %arg0 @R1::@A : !db.int<32>
		%21 = relalg.getattr %arg0 @R0::@A : !db.int<32>
		%22 = db.compare eq %20 : !db.int<32>,%21 : !db.int<32>
		relalg.return %22 : !db.bool
	}
	%7 = relalg.getscalar @R1::@A %6 : !db.int<32>
    //%36 = relalg.materialize %35 [@supplier::@s_acctbal,@supplier::@s_name,@nation::@n_name,@part::@p_partkey,@part::@p_mfgr,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment] : !db.matcollection<!db.decimal<15,2>,!db.string,!db.string,!db.int<32>,!db.string,!db.string,!db.string,!db.string>
    //return %36 : !db.matcollection<!db.decimal<15,2>,!db.string,!db.string,!db.int<32>,!db.string,!db.string,!db.string,!db.string>
    return
  }
}

