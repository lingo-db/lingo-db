//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

//CHECK: |                             i  |
//CHECK: ----------------------------------
//CHECK: |                             0  |
//CHECK: |                             1  |
//CHECK: |                             2  |
//CHECK: |                             3  |
!result_table_type = !subop.result_table<[ir : index]>
!local_table_type = !subop.local_table<[ir : index],["i"]>
module {
    func.func @main(){
    	%subop_result = subop.execution_group (){
			%heap = subop.create_heap !subop.heap<4,[ih : index]> ["ih"] ([%left],[%right]){
				%lt = arith.cmpi ult, %left, %right : index
				tuples.return %lt : i1
			}
			%generated, %streams = subop.generate [@t::@c1({type=index})] {
				%n = arith.constant 10 : index
				%c0 = arith.constant 0 : index
				%c1 = arith.constant 1 : index
				scf.for %i = %c0 to %n step %c1 {
					subop.generate_emit %i : index
				}
				tuples.return
			}
			subop.materialize %generated {@t::@c1 => ih}, %heap : !subop.heap<4,[ih : index]>
			 %result_table = subop.create !result_table_type
			%stream = subop.scan %heap : !subop.heap<4,[ih : index]> { ih => @scan::@ih({type=index})}

			subop.materialize %stream {@scan::@ih => ir}, %result_table : !result_table_type
			%local_table = subop.create_from ["i"] %result_table : !result_table_type -> !local_table_type
			subop.execution_group_return %local_table : !local_table_type
        } -> !local_table_type
        subop.set_result 0 %subop_result  : !local_table_type
         return
    }
}
