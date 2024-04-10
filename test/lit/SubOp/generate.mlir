//RUN: run-mlir %s | FileCheck %s
//CHECK: |                          test  |
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
module{
    func.func @main(){
         %result_table = subop.create !subop.result_table<[test : index]>
        %generated = subop.generate [@t::@c1({type=index})] {
            %n = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %i = %c0 to %n step %c1 {
                subop.generate_emit %i : index
            }
            tuples.return
        }
        subop.materialize %generated {@t::@c1 => test}, %result_table : !subop.result_table<[test : index]>
        %local_table = subop.create_from ["test"] %result_table : !subop.result_table<[test : index]> -> !subop.local_table<[test : index],["test"]>
        subop.set_result 0 %local_table  : !subop.local_table<[test : index],["test"]>
        return
    }
}