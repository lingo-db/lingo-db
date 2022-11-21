//RUN: run-mlir %s | FileCheck %s
//CHECK: |                        const0  |
//CHECK: ----------------------------------
//CHECK: |                             1  |
//CHECK: |                             2  |
//CHECK: |                             2  |
//CHECK: |                             3  |
//CHECK: |                             3  |
//CHECK: |                             3  |

module{
    func.func @main(){
    %12 = subop.create ["const0"] -> !subop.result_table<[const0p0 : i32]>
    %0 = subop.create -> !subop.vector<[membern1 : i32]> initial : {
      %1400 = db.constant(1 : i32) : i32
      tuples.return %1400 : i32
    }, {
      %1400 = db.constant(2 : i32) : i32
      tuples.return %1400 : i32
    }, {
      %1400 = db.constant(3 : i32) : i32
      tuples.return %1400 : i32
    }
    %3 = subop.scan %0 : !subop.vector<[membern1 : i32]> {membern1 => @constrel1::@const0({type = i32})}

    %10 = subop.map %3 computes : [@set2::@repeat({type = index})] (%arg0: !tuples.tuple){
      %14 = tuples.getcol %arg0 @constrel1::@const0 : i32
      %19 = arith.index_cast %14 : i32 to index
      tuples.return %19 : index
    }
    %11 = subop.nested_map %10 [@set2::@repeat] (%arg0, %arg1) {
      %14 = subop.generate[]{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg2 = %c0 to %arg1 step %c1 {
          subop.generate_emit
        }
        tuples.return
      }
      tuples.return %14 : !tuples.tuplestream
    }
    subop.materialize %11 {@constrel1::@const0 => const0p0}, %12 : !subop.result_table<[const0p0 : i32]>
    subop.set_result 0 %12 : !subop.result_table<[const0p0 : i32]>
    return
}
}
    
    