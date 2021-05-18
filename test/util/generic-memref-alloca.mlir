 // RUN: db-run %s | FileCheck %s
 module {

	func @main () {
        %c1 = constant 1 : index
        %c2 = constant 2 : index

        %generic_memref=util.alloca(%c2) : !util.generic_memref<!db.int<32,nullable>>
        %testval1= db.null : !db.int<32,nullable>
        %testval2c= db.constant (42) : !db.int<32>
        %testval2= db.cast %testval2c  : !db.int<32> -> !db.int<32,nullable>

        util.store %testval1:!db.int<32,nullable>,%generic_memref[%c1] :!util.generic_memref<!db.int<32,nullable>>
        util.store %testval2:!db.int<32,nullable>,%generic_memref[] :!util.generic_memref<!db.int<32,nullable>>
        %res1=util.load %generic_memref[%c1] :!util.generic_memref<!db.int<32,nullable>> -> !db.int<32,nullable>
        %res2=util.load %generic_memref[] :!util.generic_memref<!db.int<32,nullable>> -> !db.int<32,nullable>
        //CHECK: int(NULL)
        db.dump %res1 : !db.int<32,nullable>
        //CHECK: int(42)
        db.dump %res2 : !db.int<32,nullable>
		return
	}
 }