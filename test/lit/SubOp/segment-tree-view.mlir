//RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
//RUN: if [ "$(uname)" = "Linux" ]; then env LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s; fi

//CHECK: |                             v  |                             s  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                             0  |                             0  |
//CHECK: |                             1  |                             1  |
//CHECK: |                             2  |                             3  |
//CHECK: |                             3  |                             6  |
//CHECK: |                             4  |                            10  |
//CHECK: |                             5  |                            15  |
//CHECK: |                             6  |                            21  |
//CHECK: |                             7  |                            28  |
//CHECK: |                             8  |                            36  |
//CHECK: |                             9  |                            45  |

!result_table_type = !subop.result_table<[v : index, s : index]>
!local_table_type = !subop.local_table<[v : index, s : index],["v","s"]>
!c_v = !subop.continuous_view<!subop.buffer<[val : index]>>
!c_v_e_r = !subop.continous_entry_ref<!c_v>
module {
    func.func @main(){
        %subop_result = subop.execution_group (){
			%vals = subop.create!subop.buffer<[val : index]>
			%generated, %streams = subop.generate [@t::@c1({type=index})] {
				%n = arith.constant 10 : index
				%c0 = arith.constant 0 : index
				%c1 = arith.constant 1 : index
				scf.for %i = %c0 to %n step %c1 {
					subop.generate_emit %i : index
				}
				tuples.return
			}
			subop.materialize %generated {@t::@c1 => val}, %vals : !subop.buffer<[val : index]>
			 %result_table = subop.create !result_table_type

			%view = subop.create_continuous_view %vals : !subop.buffer<[val : index]> -> !c_v
			%segment_tree_view = subop.create_segment_tree_view %view :  !c_v -> !subop.segment_tree_view<[from : !c_v_e_r, to : !c_v_e_r],[sum : index]>
									initial["val"]:(%val){
										tuples.return %val : index
									}
									combine: ([%left],[%right]){
										%added = arith.addi %left, %right : index
										tuples.return %added : index
									}
			%stream = subop.scan_refs %view : !c_v @scan::@ref({type=!c_v_e_r})  {sequential}
			%stream1 = subop.gather %stream @scan::@ref { val => @scan::@currval({type=index}) }
			%stream2 = subop.get_begin_ref %stream1 %view : !c_v @view::@begin({type=!c_v_e_r})
			%stream3 = subop.lookup %stream2 %segment_tree_view[@view::@begin,@scan::@ref] :  !subop.segment_tree_view<[from : !c_v_e_r, to : !c_v_e_r],[sum : index]>  @st::@ref({type=!subop.lookup_entry_ref<!subop.segment_tree_view<[from : !c_v_e_r, to : !c_v_e_r],[sum : index]>>})
			%stream4 = subop.gather %stream3 @st::@ref { sum => @st::@sum({type=index})}
			subop.materialize %stream4 {@scan::@currval => v, @st::@sum => s}, %result_table : !result_table_type
			%local_table = subop.create_from ["v","s"] %result_table : !result_table_type -> !local_table_type
        	subop.execution_group_return %local_table : !local_table_type
        } -> !local_table_type
        subop.set_result 0 %subop_result  : !local_table_type        return
    }
}
