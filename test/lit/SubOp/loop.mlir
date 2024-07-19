//RUN: run-mlir %s | FileCheck %s
//CHECK: |                           ctr  |
//CHECK: ----------------------------------
//CHECK: |                             6  |

module{
    func.func @main(){
    	%subop_result = subop.execution_group (){
			%ctri = subop.create_simple_state !subop.simple_state<[ctri:i32]> initial: {
				 %c0 = db.constant(0) : i32
				tuples.return %c0 : i32
			}
			%finalCounter = subop.loop %ctri : !subop.simple_state<[ctri:i32]> (%ctr) -> !subop.simple_state<[ctr:i32]> {
				%newCounter = subop.create_simple_state !subop.simple_state<[ctrn:i32]>

				%20 = subop.scan %ctr : !subop.simple_state<[ctr:i32]> { ctr => @s::@ctr({type=i32})}
				%s21 = subop.lookup %20 %newCounter[] : !subop.simple_state<[ctrn:i32]> @s::@ref({type=!subop.entry_ref<!subop.simple_state<[ctrn:i32]>>})
				%s23 = subop.map %s21 computes: [@m::@p1({type=i32})] input: [@s::@ctr] (%ctrVal : i32){
					 %c1 = db.constant(1) : i32
					 %p1 = arith.addi %c1, %ctrVal : i32
					 tuples.return %p1 : i32
				}
				subop.scatter %s23 @s::@ref {@m::@p1 => ctrn}
				%s1 = subop.scan %ctr : !subop.simple_state<[ctr:i32]> {ctr => @s::@ctr({type=i32})}
				%s2 = subop.map %s1 computes: [@m::@lt5({type=i1})] input: [@s::@ctr] (%ctrVal : i32){
					 %c5 = db.constant(5) : i32
					 %lt5 = arith.cmpi ult, %ctrVal, %c5 : i32
					 tuples.return %lt5 : i1
				}
				%shouldContinue = subop.create_simple_state !subop.simple_state<[shouldContinue:i1]> initial: {
					 %false = arith.constant 0 : i1
					tuples.return %false : i1
				}
				%s3 = subop.lookup %s2 %shouldContinue[] : !subop.simple_state<[shouldContinue:i1]> @ls::@ref({type=!subop.entry_ref<!subop.simple_state<[shouldContinue:i1]>>})
				subop.scatter %s3  @ls::@ref {@m::@lt5 => shouldContinue}
				subop.loop_continue (%shouldContinue:  !subop.simple_state<[shouldContinue:i1]>["shouldContinue"]) %newCounter : !subop.simple_state<[ctrn:i32]>
			}
			%result_table = subop.create !subop.result_table<[ctr2 : i32]>
			%s10 = subop.scan %finalCounter : !subop.simple_state<[ctr:i32]> {ctr => @s::@ctr({type=i32})}
			subop.materialize %s10 {@s::@ctr=>ctr2}, %result_table : !subop.result_table<[ctr2 : i32]>
			%local_table = subop.create_from ["ctr"] %result_table : !subop.result_table<[ctr2 : i32]> -> !subop.local_table<[ctr2 : i32],["ctr"]>
			subop.execution_group_return %local_table : !subop.local_table<[ctr2 : i32],["ctr"]>
        } -> !subop.local_table<[ctr2 : i32],["ctr"]>
        subop.set_result 0 %subop_result  : !subop.local_table<[ctr2 : i32],["ctr"]>
        return
    }
}