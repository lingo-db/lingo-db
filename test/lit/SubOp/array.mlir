//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: if [ "$(uname)" = "Linux" ]; then env LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s; fi

//CHECK: |                             v  |
//CHECK: ----------------------------------
//CHECK: |                             0  |
//CHECK: |                             1  |
//CHECK: |                             2  |
//CHECK: |                             3  |
//CHECK: |                             4  |
//CHECK: |                             5  |
//CHECK: |                             6  |
//CHECK: |                             7  |
//CHECK: |                             8  |
//CHECK: |                             9  |
!result_table_type = !subop.result_table<[v : index]>
!local_table_type = !subop.local_table<[v : index],["v"]>
module {
    func.func @main(){
		%subop_result = subop.execution_group (){
			%numElements = subop.create_simple_state !subop.simple_state<[numElements: index]> initial: {
				 %c10 = arith.constant 10  : index
				tuples.return %c10 : index
			}
			%vals = subop.create_array %numElements : !subop.simple_state<[numElements: index]> -> !subop.array<[val : index]>

			%generated, %streams = subop.generate [@t::@c1({type=index}),@t::@c2({type=index})] {
				%n = arith.constant 10 : index
				%c0 = arith.constant 0 : index
				%c1 = arith.constant 1 : index
				scf.for %i = %c0 to %n step %c1 {
					subop.generate_emit %i, %i : index, index
				}
				tuples.return
			}
			%stream1 = subop.get_begin_ref %generated %vals : !subop.array<[val : index]> @view::@begin({type=!subop.continous_entry_ref<!subop.continuous_view<!subop.array<[val : index]>>>})
			%stream2 = subop.offset_ref_by %stream1 @view::@begin @t::@c1 @t::@ref({type=!subop.continous_entry_ref<!subop.array<[val : index]>>})
			subop.scatter %stream2 @t::@ref {@t::@c2 => val}

			%result_table = subop.create !result_table_type
			%stream4 = subop.scan_refs %vals : !subop.array<[val : index]> @scan::@ref({type=!subop.continous_entry_ref<!subop.array<[val : index]>>}) {sequential}
			%stream5 = subop.gather %stream4 @scan::@ref { val => @scan::@currval({type=index}) }
			subop.materialize %stream5 {@scan::@currval => v}, %result_table : !result_table_type
			%local_table = subop.create_from ["v"] %result_table : !result_table_type -> !local_table_type
			subop.execution_group_return %local_table : !local_table_type
        } -> !local_table_type
        subop.set_result 0 %subop_result  : !local_table_type
        return
    }
}
