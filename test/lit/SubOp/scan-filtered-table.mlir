// RUN: mlir-db-opt %s --lower-subop-to-cf | FileCheck %s

// Selection-vector entries are row offsets within a morsel (uint16) and must be zero-extended.
// CHECK: util.load %{{.*}} : <i16> -> i16
// CHECK-NEXT: arith.extui %{{.*}} : i16 to i64
// CHECK-NEXT: arith.index_cast %{{.*}} : i64 to index
module attributes {subop.sequential} {
  func.func @main() {
    subop.execution_group (){
      %0 = subop.execution_step [] (){
        %1 = subop.get_external "0000FEFF000006000000000000006E6174696F6E000001000100000000000000FEFF00000D000000000000006E5F726567696F6E6B65792430000001000B000000000000006E5F726567696F6E6B65790100FFFF010002000100000000000000FEFF00000B000000000000006E5F726567696F6E6B657900000100000000000000000001000200000200030001000000000000000300040001000000000000000400050000000000000000000500060000000000000000000600FFFF0200030000000000000000000300040000000000000000000400FFFF0000" : !subop.table<[n_regionkey$0 : i32], true>
        subop.execution_step_return %1 : !subop.table<[n_regionkey$0 : i32], true>
      } -> !subop.table<[n_regionkey$0 : i32], true>
      subop.execution_step %0 : !subop.table<[n_regionkey$0 : i32], true> [false] (%arg0: !subop.table<[n_regionkey$0 : i32], true>){
        %1 = subop.scan_refs %arg0 : !subop.table<[n_regionkey$0 : i32], true> @scan::@ref({type = !subop.table_entry_ref<[]>})
        subop.execution_step_return
      }
      subop.execution_group_return
    }
    return
  }
}
