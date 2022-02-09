// RUN: db-run %s | FileCheck %s

module  {
  func private @rt_dump_string(i1, !util.varlen32)
  func private @rt_dump_index(index)

  memref.global "private" constant @db_constant_string0 : memref<16xi8> = dense<[115, 116, 114, 49,115, 116, 114, 49,115, 116, 114, 49,115, 116, 114, 49]> {alignment = 1 : i64}
  func @main() {
    //prepare
    %0 = memref.get_global @db_constant_string0 : memref<16xi8>
    %1 = memref.cast %0 : memref<16xi8> to memref<?xi8>
    %2 = util.to_generic_memref %1 : memref<?xi8> -> !util.ref<? x i8>
    %3 = util.generic_memref_cast %2 : !util.ref<? x i8> -> !util.ref<i8>
    %false_14 = arith.constant false
    %len_1 = arith.constant 11 : i32
    %len_2 = arith.constant 15 : i32

    %varlen_1 = util.varlen32_create %3, %len_1
    %ref_1 = util.varlen32_getref %varlen_1 -> !util.ref<? x i8>
    //CHECK: string("str1str1str")
    call @rt_dump_string(%false_14, %varlen_1) : (i1, !util.varlen32) -> ()
    %len_res_1 = util.varlen32_getlen %varlen_1
    //CHECK: index(11)
    call @rt_dump_index(%len_res_1) : (index) -> ()

    %varlen_2 = util.varlen32_create %3, %len_2
    %ref_2 = util.varlen32_getref %varlen_2 -> !util.ref<? x i8>
    //CHECK: string("str1str1str1str")
    call @rt_dump_string(%false_14, %varlen_2) : (i1, !util.varlen32) -> ()
    %len_res_2 = util.varlen32_getlen %varlen_2
    //CHECK: index(15)
    call @rt_dump_index(%len_res_2) : (index) -> ()
    return
  }
}

