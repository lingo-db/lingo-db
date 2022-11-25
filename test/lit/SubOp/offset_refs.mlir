//RUN: run-mlir %s | FileCheck %s
//CHECK: |                             v  |                             b  |                             e  |                             r  |
//CHECK: -------------------------------------------------------------------------------------------------------------------------------------
//CHECK: |                             0  |                             0  |                             9  |                             0  |
//CHECK: |                             1  |                             0  |                             9  |                             1  |
//CHECK: |                             2  |                             0  |                             9  |                             2  |
//CHECK: |                             3  |                             0  |                             9  |                             3  |
//CHECK: |                             4  |                             0  |                             9  |                             4  |
//CHECK: |                             5  |                             0  |                             9  |                             5  |
//CHECK: |                             6  |                             0  |                             9  |                             6  |
//CHECK: |                             7  |                             0  |                             9  |                             7  |
//CHECK: |                             8  |                             0  |                             9  |                             8  |
//CHECK: |                             9  |                             0  |                             9  |                             9  |
!result_table_type = !subop.result_table<[v : index,b : index, e : index,r:index]>
module {
    func.func @main(){
        %vals = subop.create !subop.buffer<[val : index]>
        %generated = subop.generate [@t::@c1({type=index})] {
            %n = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %i = %c0 to %n step %c1 {
                subop.generate_emit %i : index
            }
            tuples.return
        }
        subop.materialize %generated {@t::@c1 => val}, %vals : !subop.buffer<[val : index]>
         %result_table = subop.create_result_table ["v","b","e","r"] -> !result_table_type

        %view = subop.create_continuous_view %vals : !subop.buffer<[val : index]> -> !subop.continuous_view<!subop.buffer<[val : index]>>
        %stream = subop.scan_refs %view : !subop.continuous_view<!subop.buffer<[val : index]>> @scan::@ref({type=!subop.continous_view_entry_ref<!subop.continuous_view<!subop.buffer<[val : index]>>>})
        %stream1 = subop.gather %stream @scan::@ref { val => @scan::@currval({type=index}) }
        %stream2 = subop.get_begin_ref %stream1 %view : !subop.continuous_view<!subop.buffer<[val : index]>> @view::@begin({type=!subop.continous_view_entry_ref<!subop.continuous_view<!subop.buffer<[val : index]>>>})
        %stream3 = subop.get_end_ref %stream2 %view : !subop.continuous_view<!subop.buffer<[val : index]>> @view::@end({type=!subop.continous_view_entry_ref<!subop.continuous_view<!subop.buffer<[val : index]>>>})
        %stream4 = subop.gather %stream3 @view::@begin { val => @scan::@firstval({type=index}) }
        %stream5 = subop.gather %stream4 @view::@end { val => @scan::@lastval({type=index}) }
        %stream6 = subop.entries_between %stream5 @view::@begin @scan::@ref @scan::@rank({type=index})
        subop.materialize %stream6 {@scan::@currval => v, @scan::@firstval => b, @scan::@lastval => e, @scan::@rank => r}, %result_table : !result_table_type
        subop.set_result 0 %result_table  : !result_table_type
        return
    }
}