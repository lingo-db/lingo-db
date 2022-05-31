// RUN: run-mlir %s | FileCheck %s

module  {
  func.func private @_ZN7runtime11DumpRuntime10dumpStringEbNS_8VarLen32E(i1, !util.varlen32)
  func.func private @_ZN7runtime11DumpRuntime9dumpIndexEm(index)

  memref.global "private" constant @db_constant_string0 : memref<16xi8> = dense<[115, 116, 114, 49,115, 116, 114, 49,115, 116, 114, 49,115, 116, 114, 49]> {alignment = 1 : i64}
  func.func @main() {
    //prepare
    %0 = memref.get_global @db_constant_string0 : memref<16xi8>
    %1 = memref.cast %0 : memref<16xi8> to memref<?xi8>
    %2 = util.to_generic_memref %1 : memref<?xi8> -> !util.ref<i8>
    %false_14 = arith.constant false
    %len_1 = arith.constant 11 : i32
    %len_2 = arith.constant 15 : i32

    %varlen_1 = util.varlen32_create %2, %len_1
    %len_res_1 = util.varlen32_getlen %varlen_1
    //CHECK: index(11)
    call @_ZN7runtime11DumpRuntime9dumpIndexEm(%len_res_1) : (index) -> ()

    %varlen_2 = util.varlen32_create %2, %len_2
    %len_res_2 = util.varlen32_getlen %varlen_2
    //CHECK: index(15)
    call @_ZN7runtime11DumpRuntime9dumpIndexEm(%len_res_2) : (index) -> ()
    return
  }
}

