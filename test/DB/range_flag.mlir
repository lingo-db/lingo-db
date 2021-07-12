 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 module {
	func @main (%execution_context:  !util.generic_memref<i8>) {
            %zero = db.constant (0) : !db.int<32>
            %ten = db.constant (10) : !db.int<32>
            %one = db.constant (1) : !db.int<32>
            %five = db.constant (5) : !db.int<32>
            %range = db.range %zero,%ten,%one : !db.int<32> -> !db.range<!db.int<32>>
            %flag = db.createflag
			%total_count=db.for %i in %range : !db.range<!db.int<32>> until %flag iter_args(%count_iter = %zero) -> (!db.int<32>){
    			%curr_count=db.add %count_iter : !db.int<32>, %i : !db.int<32>
    			%cmp = db.compare gte %i : !db.int<32>, %five : !db.int<32>
    			db.setflag %flag, %cmp
                db.yield %curr_count : !db.int<32>
			}
			//CHECK: int(15)
			db.dump %total_count : !db.int<32>
		return
	}
 }