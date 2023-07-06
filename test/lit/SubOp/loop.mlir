//RUN: run-mlir %s | FileCheck %s
//CHECK: |                           ctr  |
//CHECK: ----------------------------------
//CHECK: |                             6  |

module{
    func.func @main(){
        %ctri = subop.create_simple_state !subop.simple_state<[ctri:i32]> initial: {
             %c0 = db.constant(0) : i32
            tuples.return %c0 : i32
        }
        %finalCounter = subop.loop %ctri : !subop.simple_state<[ctri:i32]> (%ctr) -> !subop.simple_state<[ctr:i32]> {
            %newCounter = subop.create_simple_state !subop.simple_state<[ctrn:i32]>

            %20 = subop.scan %ctr : !subop.simple_state<[ctr:i32]> { ctr => @s::@ctr({type=i32})}
            %s21 = subop.lookup %20 %newCounter[] : !subop.simple_state<[ctrn:i32]> @s::@ref({type=!subop.entry_ref<!subop.simple_state<[ctrn:i32]>>})
            %s23 = subop.map %s21 computes: [@m::@p1({type=i32})] (%tpl: !tuples.tuple){
                 %ctrVal = tuples.getcol %tpl @s::@ctr : i32
                 %c1 = db.constant(1) : i32
                 %p1 = arith.addi %c1, %ctrVal : i32
                 tuples.return %p1 : i32
            }
            subop.scatter %s23 @s::@ref {@m::@p1 => ctrn}
            %s1 = subop.scan %ctr : !subop.simple_state<[ctr:i32]> {ctr => @s::@ctr({type=i32})}
            %s2 = subop.map %s1 computes: [@m::@lt5({type=i1})] (%tpl: !tuples.tuple){
                 %ctrVal = tuples.getcol %tpl @s::@ctr : i32
                 %c5 = db.constant(5) : i32
                 %lt5 = arith.cmpi ult, %ctrVal, %c5 : i32
                 tuples.return %lt5 : i1
            }
            subop.loop_continue (%s2[@m::@lt5]) %newCounter : !subop.simple_state<[ctrn:i32]>
        }
        %result_table = subop.create_result_table ["ctr"] -> !subop.result_table<[ctr2 : i32]>
        %s10 = subop.scan %finalCounter : !subop.simple_state<[ctr:i32]> {ctr => @s::@ctr({type=i32})}
        subop.materialize %s10 {@s::@ctr=>ctr2}, %result_table : !subop.result_table<[ctr2 : i32]>
        subop.set_result 0 %result_table : !subop.result_table<[ctr2 : i32]>
        return
    }
}