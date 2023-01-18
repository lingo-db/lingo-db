//RUN: run-mlir %s | FileCheck %s
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
module {
    func.func @main(){
        %vals = subop.create !subop.array<10,[val : index]>
        %view = subop.create_continuous_view %vals : !subop.array<10,[val : index]> -> !subop.continuous_view<!subop.array<10,[val : index]>>

        %generated = subop.generate [@t::@c1({type=index}),@t::@c2({type=index})] {
            %n = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %i = %c0 to %n step %c1 {
                subop.generate_emit %i, %i : index, index
            }
            tuples.return
        }
        %stream1 = subop.get_begin_ref %generated %view : !subop.continuous_view<!subop.array<10,[val : index]>> @view::@begin({type=!subop.continous_view_entry_ref<!subop.continuous_view<!subop.array<10,[val : index]>>>})
        %stream2 = subop.offset_ref_by %stream1 @view::@begin @t::@c1 @t::@ref({type=!subop.continous_view_entry_ref<!subop.continuous_view<!subop.array<10,[val : index]>>>})
        subop.scatter %stream2 @t::@ref {@t::@c2 => val}

        %result_table = subop.create_result_table ["v"] -> !result_table_type
        %stream4 = subop.scan_refs %view : !subop.continuous_view<!subop.array<10,[val : index]>> @scan::@ref({type=!subop.continous_view_entry_ref<!subop.continuous_view<!subop.array<10,[val : index]>>>})
        %stream5 = subop.gather %stream4 @scan::@ref { val => @scan::@currval({type=index}) }
        subop.materialize %stream5 {@scan::@currval => v}, %result_table : !result_table_type
        subop.set_result 0 %result_table  : !result_table_type
        return
    }
}