 // RUN: db-run %s | FileCheck %s
 module {

	func @main () {
        %c1 = constant 1 : index
        %c10 = constant 10 : index
        %memref=memref.alloc(%c10) : memref<?xi8>
        %generic_memref= util.to_generic_memref %memref : memref<?xi8> -> !util.generic_memref<!db.int<32,nullable>>
        %testval2c= db.constant (42) : !db.int<32>
        %testval2= db.cast %testval2c  : !db.int<32> -> !db.int<32,nullable>

        util.store %testval2:!db.int<32,nullable>,%generic_memref[] :!util.generic_memref<!db.int<32,nullable>>
        %raw_ptr=util.to_raw_ptr %generic_memref :!util.generic_memref<!db.int<32,nullable>>
        %generic_memref2=util.from_raw_ptr %raw_ptr -> !util.generic_memref<!db.int<32,nullable>>
        %res2=util.load %generic_memref2[] :!util.generic_memref<!db.int<32,nullable>> -> !db.int<32,nullable>
        //CHECK: int(42)
        db.dump %res2 : !db.int<32,nullable>
		return
	}
 }