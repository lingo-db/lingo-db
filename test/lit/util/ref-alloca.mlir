// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

 module {

	func.func @main () {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index

        %generic_memref=util.alloca(%c2) : !util.ref<!db.nullable<i32>>
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
		return
	}
 }
