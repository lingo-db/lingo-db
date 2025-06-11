 // RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
 // we don't run BASELINE since it does not support util.alloc

 module {

	func.func @main () {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c10 = arith.constant 10 : index
		%testbyte = arith.constant 1 : i8
        %memref1=memref.alloc(%c10) : memref<?xi8>
        memref.store %testbyte, %memref1[%c1] :memref<?xi8>
        %generic_memref1= util.to_generic_memref %memref1 : memref<?xi8> -> !util.ref<!db.nullable<i32>>
        %memref_reloaded=util.to_memref %generic_memref1 : !util.ref<!db.nullable<i32>> -> memref<?xi8>
        %reloaded=memref.load %memref_reloaded[%c1] :memref<?xi8>
        memref.dealloc %memref1 : memref<?xi8>

        %generic_memref=util.alloc(%c2) : !util.ref<!db.nullable<i32>>
        %testval1= db.null : !db.nullable<i32>
        %testval2c= db.constant (42) : i32
        %testval2= db.as_nullable %testval2c  : i32 -> !db.nullable<i32>

        util.store %testval1:!db.nullable<i32>,%generic_memref[%c1] :!util.ref<!db.nullable<i32>>
        util.store %testval2:!db.nullable<i32>,%generic_memref[] :!util.ref<!db.nullable<i32>>
        %res1=util.load %generic_memref[%c1] :!util.ref<!db.nullable<i32>> -> !db.nullable<i32>
        %res2=util.load %generic_memref[] :!util.ref<!db.nullable<i32>> -> !db.nullable<i32>
        //CHECK: int(NULL)
        db.runtime_call "DumpValue" (%res1) : (!db.nullable<i32>) -> ()
        //CHECK: int(42)
        db.runtime_call "DumpValue" (%res2) : (!db.nullable<i32>) -> ()
        util.dealloc %generic_memref : !util.ref<!db.nullable<i32>>



		return
	}
 }
