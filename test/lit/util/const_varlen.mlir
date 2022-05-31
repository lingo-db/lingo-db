// RUN: run-mlir %s | FileCheck %s

//CHECK: string("constant string!!!!!")
//CHECK: string("short str")
module  {
  func.func private @_ZN7runtime11DumpRuntime10dumpStringEbNS_8VarLen32E(i1, !util.varlen32)
  func.func private @rt_dump_index(index)

  func.func @main() {
    %varlen_1 = util.varlen32_create_const "constant string!!!!!"
    %false_14 = arith.constant false

    call @_ZN7runtime11DumpRuntime10dumpStringEbNS_8VarLen32E(%false_14, %varlen_1) : (i1, !util.varlen32) -> ()
    %varlen_2 = util.varlen32_create_const "short str"

    call @_ZN7runtime11DumpRuntime10dumpStringEbNS_8VarLen32E(%false_14, %varlen_2) : (i1, !util.varlen32) -> ()
    return
  }
}

