 // RUN: db-run %s | FileCheck %s
 module {

	func @main () {
        %c1 = constant 1 : index
        %c10 = constant 10 : index
		%testbyte = constant 1 : i8
        %memref1=memref.alloc(%c10) : memref<?xi8>
        memref.store %testbyte, %memref1[%c1] :memref<?xi8>
        %generic_memref1= util.to_generic_memref %memref1 : memref<?xi8> -> !util.generic_memref<?x!db.int<32,nullable>>
        %memref_reloaded=util.to_memref %generic_memref1 : !util.generic_memref<?x!db.int<32,nullable>> -> memref<?xi8>
        %reloaded=memref.load %memref_reloaded[%c1] :memref<?xi8>

        %memref=memref.alloc(%c10) : memref<?xi8>
        %generic_memref= util.to_generic_memref %memref : memref<?xi8> -> !util.generic_memref<!db.int<32,nullable>>
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