// RUN: db-run %s | FileCheck %s

//CHECK: string("constant string!!!!!")
//CHECK: string("short str")
module  {
  func private @rt_dump_string(i1, !util.varlen32)
  func private @rt_dump_index(index)

  func @main() {
    %varlen_1 = util.varlen32_create_const "constant string!!!!!"
    %false_14 = arith.constant false

    call @rt_dump_string(%false_14, %varlen_1) : (i1, !util.varlen32) -> ()
    %varlen_2 = util.varlen32_create_const "short str"

    call @rt_dump_string(%false_14, %varlen_2) : (i1, !util.varlen32) -> ()
    return
  }
}

