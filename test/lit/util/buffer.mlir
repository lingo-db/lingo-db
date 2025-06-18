 // RUN: run-mlir %s | FileCheck %s
 module {

	func.func @main () {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c10 = arith.constant 10 : index
		%testbyte = arith.constant 1 : i8

        %generic_memref=util.alloc(%c2) : !util.ref<!db.nullable<i32>>
        %buffer=util.buffer_create %generic_memref : !util.ref<!db.nullable<i32>>, %c2 -> !util.buffer<!db.nullable<i32>>
        %buffer_ref=util.buffer_getref %buffer : !util.buffer<!db.nullable<i32>> -> !util.ref<!db.nullable<i32>>
        %buffer_len=util.buffer_getlen %buffer : !util.buffer<!db.nullable<i32>>
        //CHECK: index(2)
        db.runtime_call "DumpValue" (%buffer_len) : (index) -> ()
        %testval1= db.null : !db.nullable<i32>
        %testval2c= db.constant (42) : i32
        %testval2= db.as_nullable %testval2c  : i32 -> !db.nullable<i32>

        util.store %testval1:!db.nullable<i32>,%buffer_ref[%c1] :!util.ref<!db.nullable<i32>>
        util.store %testval2:!db.nullable<i32>,%buffer_ref[] :!util.ref<!db.nullable<i32>>
        %res1=util.load %buffer_ref[%c1] :!util.ref<!db.nullable<i32>> -> !db.nullable<i32>
        %res2=util.load %buffer_ref[] :!util.ref<!db.nullable<i32>> -> !db.nullable<i32>
        //CHECK: int(NULL)
        db.runtime_call "DumpValue" (%res1) : (!db.nullable<i32>) -> ()
        //CHECK: int(42)
        db.runtime_call "DumpValue" (%res2) : (!db.nullable<i32>) -> ()
        util.dealloc %generic_memref : !util.ref<!db.nullable<i32>>



		return
	}
 }